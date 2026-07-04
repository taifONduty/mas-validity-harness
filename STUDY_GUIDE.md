# Study Guide: Every Concept in This Project, From Zero

Read this top to bottom once, slowly. Then use it as a reference while you
work. Every section ends with where the concept lives in your own code, so
you can open the file and see the idea in action.

---

## Part 0 — The whole project in one story

Language models do well when you give them a complete task in one message.
But real users don't talk that way — they reveal requirements bit by bit
("build me a scheduler... oh, it must be in Python... oh, and under 200ms").
The anchor paper (*LLMs Get Lost in Multi-Turn Conversation*) proved that
when the same task is revealed gradually, performance drops ~39%, mostly
because models become *inconsistent*: they answer too early, invent missing
details, and never recover from early mistakes.

Your team asks the next question: **can we watch a conversation while it
happens and measure, with numbers, whether it is going wrong?** Not by
checking the final answer (too late), but from the *process* — how the
agents' language moves and changes turn by turn. The team invented several
candidate "process metrics" (KLRD, VCV, LZ76, anisotropy). The recurring
disaster is that each flashy metric, when checked carefully, turned out to
measure something boring (like "the agent copied the prompt"). Your job is
to build the careful check itself: does any metric predict task success
**beyond** the boring explanations? That check is the gauntlet.

---

## Part 1 — The objects we study

### 1.1 What an LLM actually is
A large language model is a next-word predictor. Given text so far, it
outputs a probability for every possible next *token* and samples one.
Everything else — chat, reasoning, code — is built from repeating that step.
"Temperature" controls how adventurous the sampling is: temperature 0 almost
always picks the single most likely token; temperature 1 samples
proportionally to the probabilities, so outputs vary between runs. We run
at temperature 1 **on purpose**: we want run-to-run variation in outcomes,
because variation is what we're trying to predict.

### 1.2 Tokens and tokenizers
Models don't read letters or words; they read tokens — chunks like "sched",
"uling", " the". A tokenizer is the fixed dictionary that chops text into
these chunks. We use `cl100k_base`, a dictionary of 100,277 tokens (this is
where the "100,277-symbol vocabulary" in Tanzila's report comes from).
Why care? Because one family of our metrics (KLRD) treats an agent's output
as a *statistical fingerprint over tokens* — which tokens it uses, how
often. → `metrics_klrd.py`, the `tokenizer.encode(...)` calls.

### 1.3 Sharding, and the three conditions
The anchor paper takes a complete task and splits it into "shards" — atomic
pieces of information. Then it compares: **FULL** (whole task in one
message), **CONCAT** (all shards pasted into one message — a control proving
that splitting itself doesn't lose information), and **SHARDED** (one shard
per turn — the realistic drip-feed). SHARDED is where models fall apart.
Your simulator implements the sharded condition: each "user turn" reveals
one more shard. → `mas_loop.py`, the loop over `record["shards"]`, and
`data_sharded.json` (open it — look at one record, it makes this concrete).

### 1.4 Aptitude vs unreliability
Think of a football player. *Aptitude* = how good their best games are.
*Unreliability* = the gap between their best and worst games. The paper's
key finding: sharded conversation barely hurts aptitude (−15%) but explodes
unreliability (+112%). Models don't get dumber; they get *erratic*. That is
why "predicting which runs fail" is a meaningful research target — the same
model on the same task sometimes succeeds and sometimes doesn't.

### 1.5 Multi-agent systems (MAS) and roles
Instead of one model doing everything, a MAS assigns roles: a **Planner**
(tracks requirements, decides approach), an **Executor** (produces the
actual answer), a **Critic** (checks the answer against the requirements).
Each role is just the same LLM with a different system prompt — a "job
description" prepended to the conversation. **Role drift** is when an agent
gradually stops behaving like its job description (the Critic starts
solving, the Planner starts chatting). The team's metrics all try to detect
drift or dysfunction numerically. → `mas_loop.py`, `SYSTEM_PROMPTS`.

### 1.6 Process vs outcome
An **outcome** metric looks at the end product: did the code pass the unit
tests? Is the math answer right? → `scorers.py`. A **process** metric looks
at the journey: how did the agents' language behave along the way? The
project's whole bet is that process signals can predict outcomes early.
The gauntlet tests that bet honestly.

---

## Part 2 — The measuring instruments (Group B: the candidates)

### 2.1 Embeddings: giving text coordinates
An embedding model turns a piece of text into a list of numbers (a vector)
— think GPS coordinates in a "meaning space," where texts with similar
meaning get nearby coordinates. We use `all-MiniLM-L6-v2`, which outputs
384 numbers per text. We *normalize* every vector to length 1, so each text
becomes an arrow of equal length pointing in some direction; only the
direction (the meaning) differs. → `embedder.py`.

### 2.2 Cosine similarity
The standard way to compare two arrows: the cosine of the angle between
them. Same direction → 1.0. Unrelated → near 0. For unit-length vectors
it's just the dot product. Almost every geometric metric here is built from
cosines and distances between embeddings. → used everywhere;
`metrics_baselines.py` is full of it.

### 2.3 VCV — semantic velocity
Take an agent's turn 3 embedding and turn 4 embedding; the distance between
them is "how far the agent moved in meaning-space in one turn." That's VCV:
a speedometer. Near 0 = the agent is saying the same thing repeatedly.
Large = the agent is jumping around. Neither is automatically bad — that's
exactly why VCV alone was never a headline. → `metrics_geometry.py`,
`vcv_trajectory`.

### 2.4 Probability distributions over tokens
Count every token an agent produced and divide by the total: you get its
"vocabulary fingerprint" — a probability distribution saying "this agent
uses ' plan' 2% of the time, ' SELECT' 0.01%..." Two agents with different
jobs should have different fingerprints; an agent whose fingerprint changes
over time might be drifting. → `metrics_klrd.py`, `_counts` and
`build_reference`.

### 2.5 KL divergence — the distance between fingerprints
KL(P‖Q) measures how different distribution Q is from distribution P.
Intuition: suppose you built a compression scheme optimized for author P's
writing habits, then used it on author Q's text — KL is the extra cost you
pay per token for using the wrong codebook. Properties you must remember:
it's ≥ 0, it's 0 only when P = Q, and it's **asymmetric** (KL(P‖Q) ≠
KL(Q‖P)), which is why the spec fixes the reference on the left. The tiny
"smoothing" constant (1e-6) exists because KL explodes if Q assigns exactly
zero probability to something P uses — smoothing guarantees nothing is ever
exactly zero. → `metrics_klrd.py`, `kl_divergence`.

### 2.6 KLRD — the drift meter (and the artifact we found)
KLRD works like this: first build a **reference distribution** P for each
role — the statistical version of its job description, from "role-pure"
seed text. Then, as the conversation runs, maintain a **running estimate**
Q of what the agent is *currently* sounding like, and plot KL(P‖Q) over
turns. Rising curve = agent drifting from its role. The running estimate
uses an **exponential moving average (EMA)**:

    Q_new = 0.65 · Q_old + 0.35 · (this turn's fingerprint)

An EMA is a leaky memory: recent turns matter most, old turns fade. The
0.35 is α, chosen so memory spans ~3 turns.

**The artifact (our discovery):** Q starts at P (a broad distribution built
from many texts) and the EMA drags it toward the agent's own per-turn
fingerprint — which is always *narrow* (one turn contains few distinct
tokens). So KL(P‖Q) rises mechanically for ANY agent, even one that repeats
the identical sentence forever. Our frozen agent scored slope +0.29; the
team's real-corpus mean was +0.300. The published "universal drift" is
mostly the meter's needle moving on its own.

**The fix (KLRD-excess):** compute the exact curve a perfectly *stationary*
agent would produce (one that always emits its own average fingerprint),
and subtract it. What remains — the excess — is only genuine
*non-stationarity*: real change over time. Frozen agent: 0.000. True
drifter: +0.099. → `metrics_klrd.py`, `klrd_excess_trajectory`, and run
`test_synthetic.py` to watch it happen.

### 2.7 Entropy, Sample Entropy, and LZ76 — the rhythm detectors
*Entropy* measures unpredictability. Sadek's metric (CSE) asked: take the
VCV motion series (how far the conversation moved each turn) — is that
rhythm predictable (agents stuck in a loop), structured-but-varied
(healthy), or random (chaotic flailing)? **Sample Entropy** formalizes
"predictable": if short patterns in the series repeat, do slightly longer
patterns repeat too? Trouble: with only 10–50 turns there are too few
patterns to count, so the estimate fails ("saturates") on 76% of real runs.
**LZ76** is the rescue: it measures how *compressible* the sequence is —
think of zipping a file: repetitive content compresses tiny, random content
doesn't compress. Calibrated bands: ~0.5 stuck, ~1.0–1.2 healthy, >1.3
chaotic. → `metrics_geometry.py`, `lz76_normalized`.

### 2.8 Anisotropy / cone collapse — the shape of the cloud
Take all of one agent's turn embeddings as a cloud of points. A healthy,
varied agent produces a roundish cloud (spread in many directions); a
degenerate agent that keeps rephrasing one idea produces a cigar or a line.
Covariance eigenvalues are the lengths of the cloud's principal axes;
**top-1 explained variance** = fraction of all spread lying along the
single longest axis (→1 means collapsed); **participation ratio** = an
effective count of "how many directions the cloud really uses."
History lesson: this metric predicted privacy leakage at AUC 0.933 — until
Aritra's control showed plain brief-cosine did it at 0.844 and anisotropy
fell to 0.500 (a coin flip) within task. It died to a baseline. It's in our
gauntlet to confirm that death against *outcome* — and as a monument to why
the gauntlet exists. → `metrics_geometry.py`, `anisotropy`.

---

## Part 3 — The boring enemies (Group A: baselines)

A **baseline** is the dumbest explanation that could produce your result.
The team's two dead headlines both died to baselines: "KLRD predicts
leakage" collapsed into "copying the brief means quoting its private
tokens," and cone collapse collapsed into "similarity to the brief."
So we compute the boring features first, and every candidate must prove it
adds something beyond them:

length features (turn count, tokens — verbose agents fail differently);
**brief-cosine** (similarity of each turn to the task brief — the assassin
that killed anisotropy); **brief-jaccard** (word-overlap with the brief — a
verbatim-copying detector); **self-repetition** (cosine between an agent's
consecutive turns); **lexical novelty** (fraction of never-before-seen
words per turn). → `metrics_baselines.py`, with comments.

---

## Part 4 — The statistics (how the gauntlet decides)

### 4.1 Spearman correlation
Measures whether two variables move together — using *ranks*, so it doesn't
care about scale or outliers, only order. ρ = +1 perfect together, −1
perfect opposite, 0 unrelated. We use it for the first look: "does feature
X even move with outcome at all?" → `gauntlet.py`, step 1.

### 4.2 p-values, and the multiple-comparisons trap
A p-value answers: "if there were truly no relationship, how often would
chance alone produce a signal this strong?" p = 0.03 sounds impressive
until you remember we test ~45 features: at the p < 0.05 threshold you
should *expect* about two false alarms from pure chance. You saw this live
in the smoke test — unplanted features showed p < 0.05. Rule: univariate
p-values locate candidates; they never confirm them.

### 4.3 Logistic regression
The simplest model that predicts a probability of success (0–1) from a set
of features — it learns a weight per feature. We deliberately use the
simplest model: the question is whether the *features* carry information,
not whether a clever model can squeeze signal out.

### 4.4 AUC — the honest score
AUC answers: pick one successful run and one failed run at random; how
often does the model rank the success above the failure? 0.5 = coin flip,
1.0 = perfect. It's the standard currency of "does this signal predict?"
— all the team's dead headlines were quoted in AUC (0.933 → 0.500).

### 4.5 Cross-validation and leave-one-out (LOO)
If you fit and score a model on the same data, it "memorizes" and flatters
itself. Cross-validation: hide part of the data, fit on the rest, score on
the hidden part. **Leave-one-out** does this N times, hiding one record
each time — the right choice when N is small. Subtle trap we handle in
code: even feature *scaling* must be fit only on the training part of each
fold, or information leaks and inflates the score. → `gauntlet.py`,
`_loo_auc`, the `Pipeline`.

### 4.6 Incremental validity (ΔAUC) — the only number that matters
Model 1: baselines only. Model 2: baselines + one candidate metric.
ΔAUC = AUC₂ − AUC₁ = **what does the candidate know that the boring
features don't?** A candidate with a great solo AUC but ΔAUC ≈ 0 is just
re-measuring something boring — which is precisely the autopsy of cone
collapse. Our survival bar: ΔAUC > 0.05. → `gauntlet.py`, step 2.

### 4.7 Permutation test
The final lie detector. Shuffle the outcome labels randomly (destroying any
true relationship), recompute ΔAUC, repeat thousands of times. If shuffled
data beats your real ΔAUC 20% of the time, your result is 80% luck. We
require perm_p < 0.05. → `gauntlet.py`, step 3.

### 4.8 Why we never pool task families
Suppose code tasks have high success and low VCV, summary tasks low success
and high VCV — pooled together, VCV "predicts" failure brilliantly while
predicting nothing *within* either family. Pooling manufactures fake
signals from group differences. Every gauntlet run is within one family.
This exact mistake is what made cone collapse look real.

---

## Part 5 — Map: concept → file

| Concept | File | Function |
|---|---|---|
| Sharded MAS simulation | `mas_loop.py` | `run_record` |
| Roles / system prompts | `mas_loop.py` | `SYSTEM_PROMPTS` |
| Outcome scoring | `scorers.py` | `score_math`, `score_code` |
| Embeddings | `embedder.py` | `SBERTEmbedder` |
| Trivial baselines | `metrics_baselines.py` | `baseline_features` |
| Token distributions, KL | `metrics_klrd.py` | `build_reference`, `kl_divergence` |
| KLRD (spec) + our fix | `metrics_klrd.py` | `klrd_trajectory`, `klrd_excess_trajectory` |
| VCV, LZ76, anisotropy | `metrics_geometry.py` | all three |
| Calibration on fakes | `test_synthetic.py` | run it, read the asserts |
| Spearman, ΔAUC, permutation | `gauntlet.py` | `run_family` |

---

## Part 6 — Seven-day active learning plan

Learning = doing, predicting, breaking. One hour of playing with the code
beats five hours of reading.

**Day 1 — the anchor paper, again.** Reread Lost-in-Conversation knowing
what you now know. Then open `data_sharded.json`, read 3 records, and
explain out loud what FULL/CONCAT/SHARDED would look like for each.

**Day 2 — embeddings by hand.** In a Python shell, load `MockEmbedder`,
encode "the budget is 100k" vs "the cost limit is 100000" vs "bananas are
yellow," and compute the dot products. Predict the ordering before you run.

**Day 3 — KL by hand.** Compute KL between two tiny 3-token distributions
with pen and paper (P = [.7,.2,.1], Q = [.5,.3,.2]). Then verify with
`kl_divergence` on padded arrays. Swap P and Q — see the asymmetry.

**Day 4 — break the frozen agent.** Open `test_synthetic.py`. Change the
frozen agent's sentence, rerun, watch the raw KLRD slope stay ~0.3 no
matter what the sentence is. Now you *own* the artifact finding — you can
defend it in front of Tanzila because you've poked it from five angles.

**Day 5 — invent an agent.** Add a fifth synthetic agent: e.g. one that
alternates between two sentences ("oscillator"). Predict every metric's
value before running. Wrong predictions are where learning happens.

**Day 6 — statistics on paper.** Take 10 imaginary runs (5 success, 5
fail), invent a feature, and compute AUC by hand by counting correctly
ranked pairs. Then explain LOO to a rubber duck: why can't we test on
training data?

**Day 7 — teach-back.** Write a half-page explanation of the project as if
to another new member, without opening this guide. Whatever you can't
explain, that's next week's gap. Send it to Claude for correction.

---

## Glossary (one-liners for quick reference)

**Token** — the chunks models read; ~¾ of a word on average. **Tokenizer**
— the fixed dictionary doing the chopping. **Shard** — one atomic piece of
a task's requirements. **Aptitude** — best-case ability. **Unreliability**
— gap between best and worst runs. **MAS** — multi-agent system. **Role
drift** — an agent no longer behaving per its job description.
**Embedding** — a vector giving text coordinates in meaning-space.
**Cosine similarity** — angle-based similarity of two vectors (1 = same
direction). **VCV** — distance moved in meaning-space per turn.
**Distribution** — probabilities summing to 1 over all tokens. **KL
divergence** — asymmetric mismatch cost between two distributions. **EMA**
— running average with fading memory. **Stationary** — statistically
unchanging over time; the opposite of drifting. **KLRD-excess** — KLRD
minus the mechanical part a stationary agent would produce. **SampEn** —
predictability of a time series via repeating patterns. **LZ76** —
complexity via compressibility. **Anisotropy** — how unevenly a point
cloud spreads across directions. **Eigenvalues** — lengths of a cloud's
principal axes. **Baseline** — the dumbest explanation that must be ruled
out. **Spearman ρ** — rank correlation. **p-value** — probability chance
alone produces a signal this strong. **AUC** — probability a random
success outranks a random failure. **LOO** — leave-one-out
cross-validation. **ΔAUC** — added predictive power beyond baselines.
**Permutation test** — shuffle labels to measure luck. **Within-task** —
analyzed inside one task family to avoid pooling artifacts.
