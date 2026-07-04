"""Planner -> Executor -> Critic loop over progressively revealed shards.

Design choices (document these in your group update — they ARE the method):

1. Shards are revealed one per 'user turn' unconditionally (the paper's
   'gradual sharding' simplification; we are not simulating a reactive user).
2. Conversation history is broadcast to every agent — the worst-case regime,
   matching both Tanzila's KLRD pilot and falundafa's MAS runs.
3. The Executor must end with 'FINAL ANSWER:' (math) or a ```python block
   (code) so outcome scoring is mechanical, not judged.
4. Temperature 1.0 by default — the anchor paper shows T=0 does NOT fix
   multi-turn unreliability, and we WANT outcome variance to predict.
"""
import json
import time

SYSTEM_PROMPTS = {
    "planner": (
        "You are the PLANNER in a three-agent team solving a task that the "
        "user reveals gradually. Maintain a running list of ALL requirements "
        "revealed so far, note what is still unknown, and give the team a "
        "short plan for the current best answer. Do not solve the task "
        "yourself. Be concise."
    ),
    "executor": (
        "You are the EXECUTOR in a three-agent team. Using the conversation "
        "so far (user requirements, planner's plan, critic's feedback), "
        "produce the current best COMPLETE answer to the task. "
        "For math: the FINAL LINE of your message must be exactly "
        "'FINAL ANSWER: <number>' with no text after it. "
        "For code: output one complete ```python code block``` implementing "
        "the requested function."
    ),
    "critic": (
        "You are the CRITIC in a three-agent team. Check the executor's "
        "latest answer against EVERY requirement the user has revealed so "
        "far, including early ones. List violated or forgotten requirements "
        "explicitly. Be concise and concrete."
    ),
}

ROLE_ORDER = ["planner", "executor", "critic"]

import re as _re
_FA_OK = _re.compile(r"FINAL ANSWER\s*[:\-]?\s*[\*\_\$\s]*-?\d", _re.I)
_CODE_OK = _re.compile(r"```")


def _parseable(text, task):
    return bool(_CODE_OK.search(text) if task == "code" else _FA_OK.search(text))


def elicit_final(transcript, task, backbone):
    """Post-conversation answer elicitation. This is MEASUREMENT APPARATUS,
    not the phenomenon: it runs at temperature 0 for format compliance, with
    ONE strict retry if the reply is still unparseable. Never appended to
    `turns`, so process metrics see only the natural conversation."""
    elicit = ("All task information has now been provided. Commit to your "
              "final answer now. For math, your entire last line must be "
              "exactly: FINAL ANSWER: <number>. For code, output the single "
              "complete ```python code block```.")
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPTS["executor"]}]
        + [{"role": m2["role"] if m2["role"] == "user" else "assistant",
            "content": m2["content"]} for m2 in transcript]
        + [{"role": "user", "content": elicit}]
    )
    ans = backbone.chat(messages, temperature=0.0)
    if not _parseable(ans, task):
        strict = ("Format violation. Reply with EXACTLY one line and nothing "
                  "else: FINAL ANSWER: <number>" if task != "code" else
                  "Format violation. Reply with ONLY the complete "
                  "```python code block``` and nothing else.")
        retry_messages = messages + [{"role": "assistant", "content": ans},
                                     {"role": "user", "content": strict}]
        ans2 = backbone.chat(retry_messages, temperature=0.0)
        if _parseable(ans2, task):
            ans = ans2
    return ans


def run_record(record, backbone, max_shards=None, sleep_s=0.0):
    """Run one sharded record through the MAS. Returns a trajectory dict."""
    shards = record["shards"][:max_shards] if max_shards else record["shards"]
    transcript = []          # shared, broadcast to everyone
    turns = []               # logged agent outputs

    for shard_idx, sh in enumerate(shards, start=1):
        transcript.append({"role": "user", "content": f"[User, turn {shard_idx}] {sh['shard']}"})
        for role in ROLE_ORDER:
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPTS[role]}]
                + [{"role": m["role"] if m["role"] == "user" else "assistant",
                    "content": m["content"]} for m in transcript]
                + [{"role": "user", "content": f"({role.upper()}, respond now for turn {shard_idx}.)"}]
            )
            out = backbone.chat(messages)
            turns.append({"shard_idx": shard_idx, "role": role, "text": out})
            transcript.append({"role": "assistant", "content": f"[{role.upper()}] {out}"})
            if sleep_s:
                time.sleep(sleep_s)

    final_answer = elicit_final(transcript, record["task"], backbone)

    return {
        "task_id": record["task_id"],
        "task": record["task"],
        "model": backbone.model,
        "n_shards": len(shards),
        "full_brief": " ".join(s["shard"] for s in shards),
        "turns": turns,
        "final_answer": final_answer,
        "gold": record.get("answer") or record.get("test"),
        "record_meta": {k: record[k] for k in record
                        if k in ("prompt", "test", "answer", "question")},
    }


def run_many(records, backbone, out_path, **kw):
    """Run a list of records, streaming trajectories to a JSONL file so a
    crash never loses completed work. Resumes past already-done task_ids."""
    done = set()
    try:
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["task_id"])
    except FileNotFoundError:
        pass
    with open(out_path, "a") as f:
        for i, rec in enumerate(records):
            if rec["task_id"] in done:
                continue
            try:
                traj = run_record(rec, backbone, **kw)
            except Exception as e:
                print(f"[{i}] FAILED {rec['task_id']}: {e}")
                continue
            f.write(json.dumps(traj) + "\n")
            f.flush()
            print(f"[{i}] done {rec['task_id']} ({traj['n_shards']} shards)")
