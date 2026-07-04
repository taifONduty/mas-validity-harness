"""Repair unparseable final answers in an existing trajectory JSONL —
WITHOUT re-running any conversations.

The conversation itself is fine; only the elicitation reply was
format-non-compliant. We rebuild each record's transcript exactly as
run_record constructed it (user shards + [ROLE] outputs, same prefixes),
then re-run elicit_final (temperature 0 + strict retry) only for records
whose current final_answer is unparseable. Repairs in place, keeping a
.bak backup. Cost: one or two API calls per broken record only.

Usage:
    python repair_elicitation.py trajs_math_llama.jsonl data_sharded.json
"""
import json
import shutil
import sys

from backbone import ChatBackbone
from mas_loop import elicit_final, _parseable


def rebuild_transcript(traj, record):
    """Deterministic reconstruction of the broadcast transcript."""
    shards = record["shards"][: traj["n_shards"]]
    turns = list(traj["turns"])
    transcript, ti = [], 0
    for shard_idx, sh in enumerate(shards, start=1):
        transcript.append({"role": "user",
                           "content": f"[User, turn {shard_idx}] {sh['shard']}"})
        while ti < len(turns) and turns[ti]["shard_idx"] == shard_idx:
            t = turns[ti]
            transcript.append({"role": "assistant",
                               "content": f"[{t['role'].upper()}] {t['text']}"})
            ti += 1
    return transcript


def main(traj_path, data_path):
    records = {r["task_id"]: r for r in json.load(open(data_path))}
    trajs = [json.loads(l) for l in open(traj_path)]
    backbone = None
    n_fix = 0
    for tr in trajs:
        if _parseable(tr["final_answer"], tr["task"]):
            continue
        if backbone is None:
            backbone = ChatBackbone(provider="openrouter", model=tr["model"])
        transcript = rebuild_transcript(tr, records[tr["task_id"]])
        old_tail = tr["final_answer"][-60:].replace("\n", " ")
        tr["final_answer"] = elicit_final(transcript, tr["task"], backbone)
        ok = _parseable(tr["final_answer"], tr["task"])
        n_fix += 1
        print(f"repaired {tr['task_id']}: parseable={ok} (was: ...{old_tail})")
    shutil.copy(traj_path, traj_path + ".bak")
    with open(traj_path, "w") as f:
        for tr in trajs:
            f.write(json.dumps(tr) + "\n")
    print(f"\n{n_fix}/{len(trajs)} records re-elicited; "
          f"backup at {traj_path}.bak")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
