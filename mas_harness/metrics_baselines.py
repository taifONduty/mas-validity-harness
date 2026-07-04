"""Group A: trivial baseline features.

These are the 'boring' explanations. Any candidate metric (KLRD, VCV, LZ76,
anisotropy) only counts as a finding if it predicts outcome BEYOND these.
This is the lesson of the cone-collapse episode: anisotropy looked like
AUC 0.933 until brief-cosine (a Group A feature) explained it away.
"""
import re
import numpy as np


def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _ols_slope(series):
    """Slope of a least-squares line through the series. NaN if < 2 points."""
    s = np.asarray(series, dtype=float)
    if len(s) < 2 or np.all(np.isnan(s)):
        return float("nan")
    x = np.arange(len(s))
    return float(np.polyfit(x, s, 1)[0])


def baseline_features(turn_texts, turn_embs, brief_text, brief_emb, tokenizer=None):
    """Compute all Group A features for one agent's trajectory.

    turn_texts : list[str]   agent's output at each turn
    turn_embs  : (T, d) unit-normalized embeddings of those outputs
    brief_text : the full task brief revealed so far is a per-turn concept;
                 for simplicity we use the FINAL full brief (all shards).
    brief_emb  : (d,) unit-normalized embedding of that brief
    tokenizer  : optional tiktoken encoder for exact token counts
    """
    T = len(turn_texts)
    feats = {}

    # --- length / verbosity ---
    if tokenizer is not None:
        tok_counts = [len(tokenizer.encode(t)) for t in turn_texts]
    else:
        tok_counts = [len(_tokens(t)) for t in turn_texts]
    feats["n_turns"] = T
    feats["total_tokens"] = int(np.sum(tok_counts))
    feats["mean_tokens_per_turn"] = float(np.mean(tok_counts)) if T else float("nan")
    feats["token_slope"] = _ols_slope(tok_counts)

    # --- brief-cosine: the baseline that killed cone collapse ---
    bc = [float(np.dot(e, brief_emb)) for e in turn_embs]
    feats["brief_cosine_mean"] = float(np.mean(bc)) if bc else float("nan")
    feats["brief_cosine_slope"] = _ols_slope(bc)
    feats["brief_cosine_max"] = float(np.max(bc)) if bc else float("nan")

    # --- lexical overlap with brief (verbatim-echo detector) ---
    brief_set = set(_tokens(brief_text))
    jac = []
    for t in turn_texts:
        ts = set(_tokens(t))
        u = ts | brief_set
        jac.append(len(ts & brief_set) / len(u) if u else 0.0)
    feats["brief_jaccard_mean"] = float(np.mean(jac)) if jac else float("nan")
    feats["brief_jaccard_slope"] = _ols_slope(jac)

    # --- repetition: cosine between consecutive turns of the same agent ---
    rep = [float(np.dot(turn_embs[i], turn_embs[i - 1])) for i in range(1, T)]
    feats["self_repetition_mean"] = float(np.mean(rep)) if rep else float("nan")

    # --- lexical novelty: fraction of never-seen-before tokens per turn ---
    seen, novelty = set(), []
    for t in turn_texts:
        ts = _tokens(t)
        if ts:
            novelty.append(sum(1 for x in ts if x not in seen) / len(ts))
        else:
            novelty.append(0.0)
        seen.update(ts)
    feats["novelty_mean"] = float(np.mean(novelty)) if novelty else float("nan")
    feats["novelty_slope"] = _ols_slope(novelty)

    return feats


BASELINE_COLUMNS = [
    "n_turns", "total_tokens", "mean_tokens_per_turn", "token_slope",
    "brief_cosine_mean", "brief_cosine_slope", "brief_cosine_max",
    "brief_jaccard_mean", "brief_jaccard_slope",
    "self_repetition_mean", "novelty_mean", "novelty_slope",
]
