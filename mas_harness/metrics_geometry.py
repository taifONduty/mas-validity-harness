"""Candidate geometric metrics: VCV, LZ76 complexity, anisotropy (cone collapse).

VCV        : per-turn semantic velocity  ||e_t - e_{t-1}||  on unit vectors.
LZ76       : Lempel-Ziv 1976 complexity of the (binarized) motion series —
             Sadek's replacement for SampEn, which saturated on 76% of real runs.
Anisotropy : eigen-spectrum of the agent's embedding cloud —
             participation ratio and top-1 explained variance (Tanzila's
             cone-collapse numbers). Remember: this one already died once
             against brief-cosine; it is in the gauntlet to confirm that
             death against OUTCOME, not to resurrect it quietly.
"""
import numpy as np


# ------------------------------ VCV ------------------------------
def vcv_trajectory(turn_embs):
    T = len(turn_embs)
    deltas = [float(np.linalg.norm(turn_embs[i] - turn_embs[i - 1])) for i in range(1, T)]
    s = np.asarray(deltas)
    x = np.arange(len(s))
    return {
        "vcv_series": deltas,
        "vcv_mean": float(s.mean()) if len(s) else float("nan"),
        "vcv_max": float(s.max()) if len(s) else float("nan"),
        "vcv_slope": float(np.polyfit(x, s, 1)[0]) if len(s) >= 2 else float("nan"),
    }


# ------------------------------ LZ76 ------------------------------
def _lz76_complexity(binary_seq):
    """Kaspar–Schuster counting of distinct phrases in a binary string."""
    s = "".join("1" if b else "0" for b in binary_seq)
    n = len(s)
    if n == 0:
        return 0
    i, c, l = 0, 1, 1
    k, k_max = 1, 1
    while True:
        if i + k > n - 1 or l + k > n - 1:
            c += 1
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def lz76_normalized(motion_series):
    """Binarize the VCV motion series at its median, then LZ76, normalized by
    the complexity of a random sequence of the same length: c * log2(n) / n.
    ~0.5 stuck | ~1.0-1.2 healthy | >1.3 chaotic (Sadek's calibration)."""
    s = np.asarray(motion_series, dtype=float)
    n = len(s)
    if n < 4:
        return float("nan")
    binary = s > np.median(s)
    c = _lz76_complexity(binary)
    return float(c * np.log2(n) / n)


# ------------------------------ Anisotropy ------------------------------
def anisotropy(turn_embs):
    """Participation ratio and top-1 explained variance of the turn cloud."""
    X = np.asarray(turn_embs)
    if X.shape[0] < 3:
        return {"participation_ratio": float("nan"), "top1_var": float("nan"),
                "effective_rank": float("nan")}
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / (X.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    tot = eig.sum()
    if tot <= 0:
        return {"participation_ratio": 1.0, "top1_var": 1.0, "effective_rank": 1.0}
    pr = float(tot ** 2 / np.sum(eig ** 2))
    p = eig / tot
    p = p[p > 0]
    erank = float(np.exp(-np.sum(p * np.log(p))))
    return {"participation_ratio": pr,
            "top1_var": float(eig.max() / tot),
            "effective_rank": erank}
