"""Trajectories (scored JSONL) -> features.csv, one row per record.

Per agent (planner/executor/critic) we compute Group A baselines and all
Group B candidates, prefix them with the role name, and join into one wide
row together with task family and outcome. That wide row is what the
gauntlet consumes.

Reference distributions for KLRD: built from each role's OWN first-turn
outputs across the corpus (role-pure by construction, matches the spirit of
'seed prompts' without extra API calls). Centroids for role-stability:
mean embedding of the same first-turn corpus.
"""
import json
import sys
import numpy as np
import pandas as pd

from embedder import get_embedder
from metrics_baselines import baseline_features
from metrics_klrd import build_reference, klrd_trajectory, klrd_excess_trajectory, centroid_stability
from metrics_geometry import vcv_trajectory, lz76_normalized, anisotropy

ROLES = ["planner", "executor", "critic"]


def load_trajs(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_role_resources(trajs, embedder, tokenizer):
    """Per-role reference distribution (KLRD) and centroid (role stability)
    from every agent's FIRST turn across the corpus."""
    refs, cents = {}, {}
    for role in ROLES:
        first_turns = []
        for tr in trajs:
            texts = [t["text"] for t in tr["turns"] if t["role"] == role]
            if texts:
                first_turns.append(texts[0])
        refs[role] = build_reference(first_turns, tokenizer) if tokenizer else None
        embs = embedder.encode(first_turns)
        c = embs.mean(axis=0)
        cents[role] = c / (np.linalg.norm(c) + 1e-12)
    return refs, cents


def features_for_traj(traj, refs, cents, embedder, tokenizer):
    row = {"task_id": traj["task_id"], "task": traj["task"],
           "model": traj["model"], "outcome": traj.get("outcome")}
    brief = traj["full_brief"]
    brief_emb = embedder.encode([brief])[0]

    for role in ROLES:
        texts = [t["text"] for t in traj["turns"] if t["role"] == role]
        if len(texts) < 2:
            continue
        embs = embedder.encode(texts)

        f = baseline_features(texts, embs, brief, brief_emb, tokenizer)
        if tokenizer is not None and refs[role] is not None:
            k = klrd_trajectory(texts, refs[role], tokenizer)
            f.update({k2: v for k2, v in k.items() if not k2.endswith("_series")})
            kx = klrd_excess_trajectory(texts, refs[role], tokenizer)
            f.update({k2: v for k2, v in kx.items() if not k2.endswith("_series")})
        f.update({k2: v for k2, v in
                  centroid_stability(embs, cents[role]).items()
                  if not k2.endswith("_series")})
        v = vcv_trajectory(embs)
        f.update({k2: v2 for k2, v2 in v.items() if not k2.endswith("_series")})
        f["lz76"] = lz76_normalized(v["vcv_series"])
        f.update(anisotropy(embs))

        row.update({f"{role}_{k2}": v2 for k2, v2 in f.items()})
    return row


def main(traj_path, out_csv, embedder_kind="sbert"):
    trajs = load_trajs(traj_path)
    embedder = get_embedder(embedder_kind)
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = None
        print("WARNING: tiktoken unavailable, KLRD features skipped")
    refs, cents = build_role_resources(trajs, embedder, tokenizer)
    rows = [features_for_traj(t, refs, cents, embedder, tokenizer) for t in trajs]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {len(df)} rows x {len(df.columns)} cols -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "sbert")
