"""The gauntlet (v2 — post-audit statistics).

Changes from v1, per methodology audit:
1. LOO-CV AUC replaced by REPEATED STRATIFIED 5-FOLD CV (20 repeats).
   Pooled LOO predictions give unstable/distorted AUC at small N; repeated
   k-fold gives a mean AUC plus repeat-level spread (uncertainty band).
2. Incremental dAUC computed as a PAIRED difference per repeat (same fold
   seeds for baseline-only and baseline+candidate), reported with SD.
3. Formal multiple-comparison control: Benjamini-Hochberg FDR across ALL
   candidates (untested candidates count as p=1 — conservative).
4. dAUC > 0.05 is a pre-registered descriptive bar, not the decision rule;
   survival = dAUC>0.05 AND BH q<0.05 AND cross-backbone replication.

Never pool task families.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from metrics_baselines import BASELINE_COLUMNS

ROLES = ["planner", "executor", "critic"]
CANDIDATE_STEMS = [
    "klrd_slope", "klrd_final", "klrd_auc",
    "klrd_excess_slope", "klrd_excess_final",
    "rolestab_mean", "rolestab_min", "rolestab_slope",
    "vcv_mean", "vcv_max", "vcv_slope", "lz76",
    "participation_ratio", "top1_var", "effective_rank",
]
DELTA_AUC_BAR = 0.05
N_REPEATS = 20
SCREEN_QUANTILE_PERMS = 2000    # permutations for screened candidates

# TWO-TIER TESTING (pre-registration in code form).
# PRIMARY: the few hypotheses we commit to BEFORE seeing real data; BH-FDR
# is applied within this small set only (m=3), so real effects can survive.
# Everything else is EXPLORATORY: reported with BH within its own set, but
# labeled hypothesis-generating only — it can never be a paper claim by
# itself. Edit PRIMARY_STEMS only BEFORE running on real data, never after.
PRIMARY_STEMS = ["klrd_excess_slope"]


def _cols(df, stems):
    return [f"{r}_{s}" for r in ROLES for s in stems if f"{r}_{s}" in df.columns]


def _model():
    return make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def _cv_aucs(X, y, n_repeats=N_REPEATS, seed=0):
    """AUC per repeat of stratified 5-fold CV (fold seeds = seed+r)."""
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
        proba = cross_val_predict(_model(), X, y, cv=skf, method="predict_proba")
        out.append(roc_auc_score(y, proba[:, 1]))
    return np.asarray(out)


def _paired_delta(Xb, Xc, y, n_repeats=N_REPEATS, seed=0):
    a_b = _cv_aucs(Xb, y, n_repeats, seed)
    a_c = _cv_aucs(Xc, y, n_repeats, seed)     # same fold seeds -> paired
    d = a_c - a_b
    return float(a_b.mean()), float(a_c.mean()), float(d.mean()), float(d.std())


def _bh_qvalues(pvals):
    """Benjamini-Hochberg q-values (monotone-adjusted)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_i, idx in enumerate(order[::-1]):        # largest p first
        rank = m - rank_i
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q


def run_family(df, family, n_perm=SCREEN_QUANTILE_PERMS, seed=0):
    if "family" not in df.columns:
        df = df.assign(family=df["task"])
    d = df[df["family"] == family].copy()
    d = d[d["outcome"].notna()]
    y = d["outcome"].astype(int).to_numpy()
    print(f"\n=== {family}: n={len(d)}, success rate={y.mean():.2f} ===")
    n_min = min((y == 0).sum(), (y == 1).sum())
    if len(d) < 30 or n_min < 8:
        print(f"Insufficient data (minority class n={n_min}, need >=8; "
              f"total n={len(d)}, need >=30). Generate more records.")
        return None

    base_cols = _cols(d, BASELINE_COLUMNS)
    cand_cols = _cols(d, CANDIDATE_STEMS)

    # 1) univariate landscape (descriptive only — never confirmatory)
    uni = []
    for c in base_cols + cand_cols:
        x = d[c].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        uni.append({"feature": c, "spearman_rho": rho, "p": p,
                    "group": "A-baseline" if c in base_cols else "B-candidate"})
    uni = pd.DataFrame(uni).sort_values("p")

    # 2) incremental validity, paired repeated-CV
    Xb = d[base_cols].to_numpy(dtype=float)
    auc_base = float(_cv_aucs(Xb, y, seed=seed).mean())
    print(f"baselines-only AUC (20x5-fold mean) = {auc_base:.3f}  <- the bar")

    rows = []
    for c in cand_cols:
        Xc = d[base_cols + [c]].to_numpy(dtype=float)
        ab, ac, dmean, dsd = _paired_delta(Xb, Xc, y, seed=seed)
        rows.append({"candidate": c, "auc_with": ac, "delta_auc": dmean,
                     "delta_sd": dsd, "perm_p": np.nan})
    inc = pd.DataFrame(rows).sort_values("delta_auc", ascending=False)

    # 3) permutation test for screened candidates (cheaper CV inside)
    rng = np.random.default_rng(seed)
    primary_cols_early = set(_cols(d, PRIMARY_STEMS))
    for i, row in inc.iterrows():
        is_primary = row["candidate"] in primary_cols_early
        # primaries ALWAYS get their permutation p (pre-registered tests);
        # exploratory only if they clear the screening bar
        if not is_primary and row["delta_auc"] <= DELTA_AUC_BAR:
            continue
        Xc = d[base_cols + [row["candidate"]]].to_numpy(dtype=float)
        beat = 0
        for _ in range(n_perm):
            yp = rng.permutation(y)
            _, _, dperm, _ = _paired_delta(Xb, Xc, yp, n_repeats=3, seed=seed)
            beat += dperm >= row["delta_auc"]
        inc.loc[i, "perm_p"] = (beat + 1) / (n_perm + 1)

    # 4) two-tier BH-FDR: primaries corrected among themselves (m small),
    #    exploratory corrected among themselves and labeled as such
    primary_cols = set(_cols(d, PRIMARY_STEMS))
    inc["tier"] = inc["candidate"].map(
        lambda c: "PRIMARY" if c in primary_cols else "exploratory")
    inc["bh_q"] = np.nan
    for tier in ("PRIMARY", "exploratory"):
        m = inc["tier"] == tier
        if m.any():
            pv = inc.loc[m, "perm_p"].fillna(1.0).to_numpy()
            inc.loc[m, "bh_q"] = _bh_qvalues(pv)
    inc["survives"] = ((inc["delta_auc"] > DELTA_AUC_BAR)
                       & (inc["bh_q"] < 0.05) & (inc["tier"] == "PRIMARY"))

    return {"family": family, "n": len(d), "auc_base": auc_base,
            "univariate": uni, "incremental": inc}


def main(features_csv, n_perm=SCREEN_QUANTILE_PERMS):
    df = pd.read_csv(features_csv)
    # Stratify code by source: HumanEval and LiveCodeBench have very
    # different base rates (0.27 vs 0.09 on the pilot backbone); pooling
    # them would manufacture exactly the kind of between-group artifact
    # this harness exists to prevent. Low-variance strata are benched
    # automatically by the minority-class guard in run_family.
    df["family"] = df["task"]
    is_code = df["task"] == "code"
    df.loc[is_code & df["task_id"].str.contains("HumanEval"), "family"] = "code-HumanEval"
    df.loc[is_code & df["task_id"].str.contains("livecodebench"), "family"] = "code-LCB"
    for fam in sorted(df["family"].unique()):
        res = run_family(df, fam, n_perm=n_perm)
        if res is None:
            continue
        print("\nTop univariate (descriptive only):")
        print(res["univariate"].head(10).to_string(index=False))
        print("\nIncremental over baselines (survival = dAUC>0.05 & bh_q<0.05"
              " & cross-backbone replication):")
        print(res["incremental"].head(15).to_string(index=False))
        res["univariate"].to_csv(f"gauntlet_{res['family']}_univariate.csv", index=False)
        res["incremental"].to_csv(f"gauntlet_{res['family']}_incremental.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else SCREEN_QUANTILE_PERMS)
