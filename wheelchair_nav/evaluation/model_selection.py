"""Multi-criteria model selection over the table produced by
evaluation/eval_depth_comparison.py -- "which depth model is actually the
most worth it?" once error, accuracy, model size, compute and speed are
all on the table at once.

No single number answers that on its own, so four published methods are
reported side by side, from the weakest assumptions to the strongest:

1. PARETO FRONTIER -- assumption-free. Model A dominates B if A is at
   least as good on EVERY criterion and strictly better on at least one;
   the non-dominated set is what remains. This filters out indefensible
   choices without ranking what survives. Applied to DNN benchmarking by
   Bianco et al., "Benchmark Analysis of Representative Deep Neural
   Network Architectures", IEEE Access 6, 2018.

2. INFORMATION DENSITY -- accuracy per parameter, from Canziani, Paszke &
   Culurciello, "An Analysis of Deep Neural Network Models for Practical
   Applications", arXiv:1605.07678, 2016:

       D(N) = a(N) / p(N)

   How well a network exploits its parametric capacity. Ignores compute
   entirely, which is exactly why it must not be read alone -- see the
   RT-MonoDepth case in this project, where few parameters coexist with
   the highest MAC count of all models compared.

3. NETSCORE -- Wong, "NetScore: Towards Universal Metrics for Large-scale
   Performance Analysis of Deep Neural Networks for Practical On-Device
   Edge Usage", arXiv:1806.05512, 2018:

       Omega(N) = 20 * log10( a(N)^alpha / ( p(N)^beta * m(N)^gamma ) )

   with a = accuracy (percent), p = parameters (millions), m = MACs
   (billions). The paper's coefficients are alpha=2 (accuracy weighted
   twice as heavily, in the log domain, as the costs), beta=gamma=0.5.
   A single scalar in decibels balancing accuracy against BOTH
   architectural and computational complexity.

4. TOPSIS -- Hwang & Yoon, "Multiple Attribute Decision Making: Methods
   and Applications", Springer, 1981. The general MCDM answer: rank
   alternatives by relative closeness to the ideal solution,

       C_i = S_i^- / (S_i^+ + S_i^-)

   where S_i^+ / S_i^- are Euclidean distances, in weighted
   vector-normalised criterion space, to the positive-ideal solution
   (best value of every criterion) and negative-ideal solution (worst
   value of every criterion). Unlike (2) and (3), TOPSIS takes explicit
   weights, so it answers "best FOR THIS deployment" rather than "best in
   general" -- and because the answer moves with the weights, it is run
   here over several weight scenarios as a sensitivity analysis.

Reading order matters: use (1) to discard dominated models, (3) for a
defensible single-number ranking that needs no hand-chosen weights, and
(4) to justify the final pick against this project's actual priorities.
Report (2) only alongside compute, never on its own.

Usage:
    python -m wheelchair_nav.evaluation.model_selection \
        --csv ./wheelchair_nav/log_sunrgbd/depth_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# criterion -> is a HIGHER value better?
CRITERIA = {
    "abs_rel": False,
    "a1": True,
    "params_m": False,
    "macs_g": False,
    "fps": True,
}

# Weight scenarios for the TOPSIS sensitivity analysis, over CRITERIA's
# key order. Each must sum to 1.
WEIGHT_SCENARIOS = {
    "A. Safety-critical (accuracy-led)": [0.30, 0.30, 0.10, 0.10, 0.20],
    "B. Balanced (equal weights)":       [0.20, 0.20, 0.20, 0.20, 0.20],
    "C. Tight embedded budget":          [0.15, 0.15, 0.25, 0.25, 0.20],
    "D. Throughput-led":                 [0.15, 0.15, 0.10, 0.10, 0.50],
}


def read_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    missing = [c for c in CRITERIA if c not in rows[0]]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing column(s): {', '.join(missing)}. "
            "Regenerate it with evaluation/eval_depth_comparison.py --out_csv "
            "(the macs_g column needs thop installed)."
        )

    usable, skipped = [], []
    for r in rows:
        try:
            values = [float(r[c]) for c in CRITERIA]
        except (TypeError, ValueError):
            skipped.append(r["model"])
            continue
        if any(v <= 0 for v in values):
            skipped.append(r["model"])
            continue
        usable.append((r["model"], np.array(values)))

    if skipped:
        print(f"Skipped (missing/non-positive criterion values): {', '.join(skipped)}\n")
    if len(usable) < 2:
        raise SystemExit("Need at least 2 fully-populated models to compare.")

    names = [n for n, _ in usable]
    X = np.vstack([v for _, v in usable])
    return names, X


def pareto_front(X: np.ndarray, higher_is_better: np.ndarray):
    """Returns, per model, the list of models that strictly dominate it
    (empty list => the model is Pareto-optimal).
    """
    signed = np.where(higher_is_better, X, -X)  # make every column higher-is-better
    dominators = []
    for i in range(len(X)):
        dom = [
            j for j in range(len(X))
            if j != i and np.all(signed[j] >= signed[i]) and np.any(signed[j] > signed[i])
        ]
        dominators.append(dom)
    return dominators


def netscore(accuracy_pct, params_m, macs_g, alpha=2.0, beta=0.5, gamma=0.5):
    """Wong (2018), eq. 1. Accuracy in percent, params in millions, MACs
    in billions."""
    return 20.0 * np.log10(accuracy_pct ** alpha / (params_m ** beta * macs_g ** gamma))


def topsis(X: np.ndarray, weights: np.ndarray, higher_is_better: np.ndarray):
    """Hwang & Yoon (1981). Vector normalisation, weighted, then relative
    closeness to the positive-ideal solution. Returns (C, S_plus, S_minus).
    """
    R = X / np.sqrt((X ** 2).sum(axis=0))          # step 2: vector normalisation
    V = R * weights                                 # step 3: weighted matrix
    a_pos = np.where(higher_is_better, V.max(0), V.min(0))   # step 4: PIS
    a_neg = np.where(higher_is_better, V.min(0), V.max(0))   #         NIS
    s_pos = np.sqrt(((V - a_pos) ** 2).sum(1))      # step 5: separation measures
    s_neg = np.sqrt(((V - a_neg) ** 2).sum(1))
    return s_neg / (s_pos + s_neg), s_pos, s_neg    # step 6: closeness coefficient


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="./wheelchair_nav/log_sunrgbd/depth_comparison.csv",
                   help="CSV written by evaluation/eval_depth_comparison.py --out_csv")
    p.add_argument("--alpha", type=float, default=2.0, help="NetScore accuracy exponent")
    p.add_argument("--beta", type=float, default=0.5, help="NetScore parameter exponent")
    p.add_argument("--gamma", type=float, default=0.5, help="NetScore MAC exponent")
    args = p.parse_args()

    names, X = read_rows(args.csv)
    hib = np.array(list(CRITERIA.values()))
    col = {c: i for i, c in enumerate(CRITERIA)}
    w = max(len(n) for n in names) + 2

    print(f"Model selection analysis over {args.csv}")
    print(f"{len(names)} models x {len(CRITERIA)} criteria: "
          f"{', '.join(f'{c}{chr(94)}' if h else f'{c}v' for c, h in CRITERIA.items())}\n")

    print("=" * 78)
    print("1) PARETO FRONTIER  (Bianco et al., IEEE Access 2018)")
    print("=" * 78)
    for name, dom in zip(names, pareto_front(X, hib)):
        verdict = "Pareto-optimal" if not dom else "dominated by " + ", ".join(names[j] for j in dom)
        print(f"  {name:<{w}}{verdict}")

    print("\n" + "=" * 78)
    print("2) INFORMATION DENSITY  (Canziani et al. 2016)   D = a1(%) / params(M)")
    print("=" * 78)
    dens = X[:, col["a1"]] * 100 / X[:, col["params_m"]]
    for i in np.argsort(-dens):
        print(f"  {names[i]:<{w}}{dens[i]:8.2f} %/M")
    print("  NOTE: ignores compute entirely -- read together with GMACs below.")

    print("\n" + "=" * 78)
    print(f"3) NETSCORE  (Wong 2018)   O = 20 log10(a^{args.alpha:g} / "
          f"(p^{args.beta:g} m^{args.gamma:g}))")
    print("=" * 78)
    ns = netscore(X[:, col["a1"]] * 100, X[:, col["params_m"]], X[:, col["macs_g"]],
                  args.alpha, args.beta, args.gamma)
    for i in np.argsort(-ns):
        print(f"  {names[i]:<{w}}{ns[i]:8.2f} dB")

    print("\n" + "=" * 78)
    print("4) TOPSIS  (Hwang & Yoon 1981)   closeness C*, weight sensitivity")
    print("=" * 78)
    header = "".join(f"{n:>16}" for n in names)
    print(f"{'Weight scenario':<36}{header}")
    print("-" * (36 + 16 * len(names)))
    wins = {}
    for label, weights in WEIGHT_SCENARIOS.items():
        C, _, _ = topsis(X, np.array(weights), hib)
        best = names[int(C.argmax())]
        wins[best] = wins.get(best, 0) + 1
        print(f"{label:<36}" + "".join(f"{c:>16.4f}" for c in C) + f"   -> {best}")
    print("\n  Ranked first in:", ", ".join(
        f"{k} ({v}/{len(WEIGHT_SCENARIOS)})" for k, v in sorted(wins.items(), key=lambda t: -t[1])))
    if len(wins) > 1:
        print("  Weight-sensitive: no single model wins every scenario -- state the "
              "weights you adopt, and why, when reporting a winner.")


if __name__ == "__main__":
    main()
