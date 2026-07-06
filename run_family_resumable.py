
import sys, os, time, pandas as pd
import gauntlet

feat_csv, family, n_perm = sys.argv[1], sys.argv[2], int(sys.argv[3])
outdir = sys.argv[4]
inc_path = f"{outdir}/gauntlet_{family}_incremental.csv"
if os.path.exists(inc_path):
    print(f"[skip] {family} already done -> {inc_path}"); sys.exit(0)

# live progress: count permutation evaluations
_orig = gauntlet._paired_delta
state = {"n": 0, "t0": time.time()}
def _counted(*a, **k):
    state["n"] += 1
    if state["n"] % 200 == 0:
        el = time.time() - state["t0"]
        print(f"  ... {state['n']} permutation evals, {el/60:.1f} min elapsed", flush=True)
    return _orig(*a, **k)
gauntlet._paired_delta = _counted

df = pd.read_csv(feat_csv)
df["family"] = df["task"]
is_code = df["task"] == "code"
df.loc[is_code & df["task_id"].str.contains("HumanEval"), "family"] = "code-HumanEval"
df.loc[is_code & df["task_id"].str.contains("livecodebench"), "family"] = "code-LCB"

res = gauntlet.run_family(df, family, n_perm=n_perm)
if res is None:
    print(f"[benched] {family}"); sys.exit(0)
os.makedirs(outdir, exist_ok=True)
res["univariate"].to_csv(f"{outdir}/gauntlet_{family}_univariate.csv", index=False)
res["incremental"].to_csv(inc_path, index=False)
print(f"[done] {family} -> {inc_path}")
print(res["incremental"].head(12).to_string(index=False))
