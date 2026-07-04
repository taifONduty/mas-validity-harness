"""KLRD — KL Reference Divergence, implemented per the klrd.pdf spec (Section 2).

Two variants live in the team's work under the SAME name — implement BOTH,
because noticing this discrepancy is itself a contribution:

1. KL-based (Tanzila's spec):
   P_r  = smoothed token distribution from role-pure seed completions
   Q_0  = P_r ;  Q_t = (1-a) Q_{t-1} + a * qhat_t ,  a = 0.35
   report per-turn KL(P_r || Q_t) and its OLS slope. Positive slope = drift.

2. Centroid-cosine "role stability" (falundafa's plots, axis label
   'Cosine Similarity to Role Centroid', values in [0,1]):
   stability_t = cosine(embed(turn_t), role_centroid)

Keep tokenization identical to the spec: cl100k_base (100,277 symbols).
"""
import numpy as np

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

VOCAB_SIZE = 100_277
SMOOTHING = 1e-6
ALPHA = 0.35


def _counts(token_ids, vocab_size=VOCAB_SIZE):
    c = np.zeros(vocab_size, dtype=np.float64)
    ids, cnt = np.unique(np.asarray(token_ids, dtype=np.int64), return_counts=True)
    ids = np.clip(ids, 0, vocab_size - 1)
    c[ids] += cnt
    return c


def build_reference(seed_texts, tokenizer):
    """P_r: smoothed empirical distribution over the vocab from seed texts."""
    c = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for t in seed_texts:
        c += _counts(tokenizer.encode(t))
    c += SMOOTHING                      # additive smoothing -> strict positivity
    return c / c.sum()


def kl_divergence(p, q):
    """KL(P || Q). Both must be strictly positive distributions."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def klrd_trajectory(turn_texts, reference, tokenizer, alpha=ALPHA):
    """Per-turn KL(P || Q_t) series + summary stats for one agent."""
    Q = reference.copy()
    series = []
    for text in turn_texts:
        ids = tokenizer.encode(text)
        if len(ids) == 0:
            series.append(series[-1] if series else 0.0)
            continue
        qhat = _counts(ids)
        qhat = qhat / qhat.sum()
        Q = (1 - alpha) * Q + alpha * qhat   # stays strictly positive (Q0 > 0)
        series.append(kl_divergence(reference, Q))
    s = np.asarray(series)
    x = np.arange(len(s))
    slope = float(np.polyfit(x, s, 1)[0]) if len(s) >= 2 else float("nan")
    return {
        "klrd_series": series,
        "klrd_slope": slope,
        "klrd_final": float(s[-1]) if len(s) else float("nan"),
        "klrd_auc": float(_trapz(s)) if len(s) >= 2 else float("nan"),
    }


def klrd_excess_trajectory(turn_texts, reference, tokenizer, alpha=ALPHA):
    """Null-corrected KLRD ('KLRD-excess').

    Discovery from the frozen-agent calibration: a constant-output agent
    (zero drift by construction) produces raw KLRD slope ~ +0.29, nearly
    identical to the +0.300 reported on real corpora. Reason: with
    Q_t = (1-a) Q_{t-1} + a qhat_t starting at a broad reference P, the EMA
    converges toward the agent's own narrow distribution no matter what the
    agent says, so KL(P||Q_t) rises mechanically.

    Fix: compute the exact null path for a STATIONARY agent that emits its
    own pooled distribution qbar every turn:
        Q_t^null = (1-a)^t P + (1-(1-a)^t) qbar
    and report excess_t = KL(P||Q_t) - KL(P||Q_t^null). A stationary agent
    (frozen OR random-but-stable) gets excess slope ~ 0; only genuine
    NON-STATIONARITY (real drift over time) produces positive excess slope.
    """
    # pooled distribution of the agent's own text
    pooled = np.zeros(VOCAB_SIZE, dtype=np.float64)
    per_turn = []
    for text in turn_texts:
        ids = tokenizer.encode(text)
        c = _counts(ids) if len(ids) else np.zeros(VOCAB_SIZE)
        per_turn.append(c)
        pooled += c
    if pooled.sum() == 0:
        return {"klrd_excess_slope": float("nan"), "klrd_excess_final": float("nan")}
    qbar = pooled / pooled.sum()

    Q = reference.copy()
    excess = []
    for t, c in enumerate(per_turn, start=1):
        if c.sum() > 0:
            Q = (1 - alpha) * Q + alpha * (c / c.sum())
        w = (1 - alpha) ** t
        Q_null = w * reference + (1 - w) * qbar
        excess.append(kl_divergence(reference, Q) - kl_divergence(reference, Q_null))
    e = np.asarray(excess)
    x = np.arange(len(e))
    return {
        "klrd_excess_series": excess,
        "klrd_excess_slope": float(np.polyfit(x, e, 1)[0]) if len(e) >= 2 else float("nan"),
        "klrd_excess_final": float(e[-1]) if len(e) else float("nan"),
    }


def centroid_stability(turn_embs, centroid):
    """falundafa-style role stability: cosine of each turn to the role centroid."""
    c = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = [float(np.dot(e, c)) for e in turn_embs]
    s = np.asarray(sims)
    x = np.arange(len(s))
    return {
        "rolestab_series": sims,
        "rolestab_mean": float(s.mean()) if len(s) else float("nan"),
        "rolestab_min": float(s.min()) if len(s) else float("nan"),
        "rolestab_slope": float(np.polyfit(x, s, 1)[0]) if len(s) >= 2 else float("nan"),
    }
