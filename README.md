# MAS Validity Harness — from-scratch pipeline

Tests whether the team's process metrics (KLRD, VCV, LZ76, anisotropy, role
stability) predict **task outcome** beyond trivial baselines, on trajectories
you generate yourself from the public Lost-in-Conversation data.

## Already verified (by construction, before any API spend)

- `test_synthetic.py` — 14/14 calibration checks pass on frozen / random /
  parrot / drifter agents.
- **Finding baked in:** raw KLRD slope is ~+0.29 for a FROZEN agent — nearly
  identical to the +0.300 mean reported on real corpora. The raw slope is an
  estimator artifact (EMA convergence from a broad reference to any narrow
  distribution). `klrd_excess_slope` is the corrected, null-subtracted
  estimator: 0.000 frozen, 0.008 stationary-random, +0.099 true drifter.
- Gauntlet statistics verified on synthetic features with one planted signal.

## Setup (your machine)

```bash
pip install numpy pandas scipy scikit-learn tiktoken sentence-transformers
export OPENROUTER_API_KEY=sk-or-...        # your own $5 account
```

Data (NOT in the zip — fetch once):
```bash
git clone --depth 1 https://github.com/microsoft/lost_in_conversation
cp lost_in_conversation/data/sharded_instructions_600.json data_sharded.json
```
627 sharded records; math has gold answers, code ships unit tests.

## Run order

```bash
# 0. calibration must pass on YOUR machine first
python test_synthetic.py

# 1. simulate: start with 60 math records (cheap, self-scoring, generative)
python - <<'EOF'
import json
from backbone import ChatBackbone
from mas_loop import run_many
d = json.load(open('data_sharded.json'))
recs = [x for x in d if x['task']=='math'][:60]
run_many(recs, ChatBackbone(provider='openrouter'), 'trajs_math_llama.jsonl', sleep_s=0.3)
EOF

# 2. score outcomes (code scoring executes model code -> run in a VM/container)
python scorers.py trajs_math_llama.jsonl scored_math_llama.jsonl

# 3. extract features (sbert = all-MiniLM-L6-v2, matches the team)
python extract_features.py scored_math_llama.jsonl features_math_llama.csv sbert

# 4. the gauntlet
python gauntlet.py features_math_llama.csv 2000

# 5. repeat step 1-4 with task='code' (100 recs), then with
#    ChatBackbone(provider='openrouter', model=REPLICATION_MODEL) for DeepSeek.
```

Cost: ~1,000-1,500 calls per 60-100 records; llama-3.1-8b ≈ well under $1
per full run; DeepSeek replication a few dollars. $5 covers everything.

## Interpretation rules (v2.1, post-audit — do not bend these)

1. TWO TIERS. PRIMARY hypotheses (default: klrd_excess_slope per role) are
   pre-registered in gauntlet.py BEFORE touching real data and BH-FDR
   corrected among themselves. Everything else is exploratory: it can
   suggest the next primary for a NEW dataset, never a claim on this one.
2. Survival = delta_auc > 0.05 AND bh_q < 0.05 AND PRIMARY tier AND
   replication on the second backbone, per task family.
3. Statistics: repeated stratified 5-fold CV (20 repeats), paired delta-AUC
   with repeat-level SD, permutation p, BH-FDR. LOO was dropped (unstable
   pooled-prediction AUC at small N).
4. Never pool task families.
5. If the success rate is near 0 or 1 in a family, there is no outcome
   variance to predict: add records or switch family. Need minority class
   n >= 8 and total n >= 30 per family; aim for 80-100+.

## Checkpoints to report back

- CP1: test_synthetic output on your machine (14/14).
- CP2: first 10 real trajectories — paste one full trajectory + its score.
       Sanity-read it: did the executor actually attempt the task?
- CP3: success rate per family (want it between 0.2 and 0.8).
- CP4: the two gauntlet tables per family + your pre-registered predictions.

## Files

| file | role |
|---|---|
| `embedder.py` | SBERT (prod) / deterministic mock (tests) |
| `backbone.py` | OpenRouter client + MockBackbone |
| `mas_loop.py` | Planner→Executor→Critic sharded simulation, resumable JSONL |
| `scorers.py` | math exact-match, code unit-test subprocess sandbox |
| `metrics_baselines.py` | Group A trivial baselines (the bar) |
| `metrics_klrd.py` | spec KLRD + corrected KLRD-excess + role stability |
| `metrics_geometry.py` | VCV, LZ76, anisotropy |
| `extract_features.py` | trajectories → wide features.csv |
| `gauntlet.py` | Spearman + repeated-CV paired ΔAUC + permutation + two-tier BH-FDR |
| `test_synthetic.py` | frozen/random/parrot/drifter calibration (run first) |
