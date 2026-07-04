"""Calibration on synthetic agents whose behavior we control BY CONSTRUCTION.

Agents (8 turns each):
  frozen   : identical sentence every turn
             -> VCV ~ 0, repetition ~ 1, LZ76 nan/low, KLRD slope ~ 0,
                top1_var high (cloud collapses to a point/line)
  random   : unrelated word salad every turn
             -> VCV high, repetition ~ 0, novelty high, KLRD slope > 0
  parrot   : copies the task brief verbatim every turn
             -> brief_cosine ~ 1, brief_jaccard ~ 1, VCV ~ 0
  drifter  : starts on-topic, vocabulary gradually replaced by off-topic
             -> KLRD slope > 0 and > frozen's, rolestab slope < 0

If any expectation fails, the instrument is broken; fix before real data.
"""
import numpy as np
from embedder import MockEmbedder
from metrics_baselines import baseline_features
from metrics_klrd import build_reference, klrd_trajectory, klrd_excess_trajectory, centroid_stability
from metrics_geometry import vcv_trajectory, lz76_normalized, anisotropy

try:
    import tiktoken
    TOK = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOK = None

rng = np.random.default_rng(7)
VOCAB_ON = ("plan constraint budget python latency requirement step verify "
            "check task answer solve compute schedule design review").split()
VOCAB_OFF = ("banana volcano guitar nebula pickle walrus tango glacier "
             "mosaic pepper lantern comet drizzle saddle marble whistle").split()

BRIEF = ("compute the total project cost given a budget ceiling of 100000 "
         "python implementation latency under 200 ms open source tooling only")
T = 8


def sent(vocab, n=14):
    return " ".join(rng.choice(vocab, size=n))


def make_agents():
    frozen = ["the plan meets the budget constraint and latency requirement"] * T
    random_a = [sent(VOCAB_ON + VOCAB_OFF) for _ in range(T)]
    parrot = [BRIEF] * T
    drifter = []
    for t in range(T):
        k_off = int(round(14 * t / (T - 1)))          # 0 -> 14 off-topic words
        words = list(rng.choice(VOCAB_ON, size=14 - k_off)) + \
                list(rng.choice(VOCAB_OFF, size=k_off))
        rng.shuffle(words)
        drifter.append(" ".join(words))
    return {"frozen": frozen, "random": random_a, "parrot": parrot, "drifter": drifter}


def main():
    emb = MockEmbedder()
    agents = make_agents()
    brief_emb = emb.encode([BRIEF])[0]

    # role-pure reference & centroid: built from ON-TOPIC text (as in prod)
    seed_texts = [sent(VOCAB_ON) for _ in range(16)]
    ref = build_reference(seed_texts, TOK) if TOK else None
    cent = emb.encode(seed_texts).mean(axis=0)

    print(f"{'agent':9s} {'vcv_mean':>8s} {'selfrep':>8s} {'briefcos':>8s} "
          f"{'novelty':>8s} {'klrd_slope':>10s} {'klrd_EXC':>10s} {'rolestab_sl':>11s} "
          f"{'lz76':>6s} {'top1var':>8s}")
    results = {}
    for name, texts in agents.items():
        E = emb.encode(texts)
        b = baseline_features(texts, E, BRIEF, brief_emb, TOK)
        v = vcv_trajectory(E)
        k = klrd_trajectory(texts, ref, TOK) if ref is not None else {"klrd_slope": float("nan")}
        kx = klrd_excess_trajectory(texts, ref, TOK) if ref is not None else {"klrd_excess_slope": float("nan")}
        rs = centroid_stability(E, cent)
        a = anisotropy(E)
        lz = lz76_normalized(v["vcv_series"])
        results[name] = dict(vcv=v["vcv_mean"], rep=b["self_repetition_mean"],
                             bc=b["brief_cosine_mean"], nov=b["novelty_mean"],
                             ks=k["klrd_slope"], kxs=kx["klrd_excess_slope"], rss=rs["rolestab_slope"],
                             lz=lz, t1=a["top1_var"])
        r = results[name]
        print(f"{name:9s} {r['vcv']:8.3f} {r['rep']:8.3f} {r['bc']:8.3f} "
              f"{r['nov']:8.3f} {r['ks']:10.4f} {r['kxs']:10.4f} {r['rss']:11.4f} "
              f"{r['lz']:6.2f} {r['t1']:8.3f}")

    print("\n--- assertions ---")
    checks = [
        ("frozen VCV ~ 0",              results["frozen"]["vcv"] < 0.01),
        ("frozen self-repetition ~ 1",  results["frozen"]["rep"] > 0.99),
        ("frozen raw KLRD slope LARGE (the artifact!)", results["frozen"]["ks"] > 0.1),
        ("frozen KLRD-EXCESS ~ 0 (fix works)", abs(results["frozen"]["kxs"]) < 0.02),
        ("random KLRD-EXCESS ~ 0 (stationary)", abs(results["random"]["kxs"]) < 0.05),
        ("random VCV high",             results["random"]["vcv"] > 0.5),
        ("random novelty > frozen",     results["random"]["nov"] > results["frozen"]["nov"]),
        ("random KLRD slope > 0",       results["random"]["ks"] > 0.02),
        ("parrot brief-cosine ~ 1",     results["parrot"]["bc"] > 0.99),
        ("parrot VCV ~ 0",              results["parrot"]["vcv"] < 0.01),
        ("drifter KLRD-EXCESS > 0 (real drift)", results["drifter"]["kxs"] > 0.02),
        ("drifter EXCESS > frozen EXCESS", results["drifter"]["kxs"] > results["frozen"]["kxs"] + 0.02),
        ("drifter rolestab slope < 0",  results["drifter"]["rss"] < -0.005),
        ("frozen top1_var >= drifter",  not (results["frozen"]["t1"] < results["drifter"]["t1"])),
    ]
    n_fail = 0
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        n_fail += (not ok)
    print(f"\n{len(checks)-n_fail}/{len(checks)} checks passed")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
