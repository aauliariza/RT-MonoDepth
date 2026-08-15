"""Decision support system: picks the most efficient depth model from the
comparison table using SAW normalisation with Shannon-entropy weights.

METHOD
------
Objective weighting, so the ranking is not steered by hand-picked weights.

Given a decision matrix X (m alternatives x n criteria):

  1. SAW normalisation -- linear scale transformation, direction-aware, so
     every column becomes "higher is better" on a common (0, 1] scale:

         benefit :  r_ij = x_ij / max_i x_ij
         cost    :  r_ij = min_i x_ij / x_ij

     [Fishburn, "Additive Utilities with Incomplete Product Sets",
      Operations Research 15(3), 1967; MacCrimmon, RAND RM-4823-ARPA, 1968]

  2. Projection of each column onto a probability simplex:

         p_ij = r_ij / sum_i r_ij

  3. Shannon entropy of each criterion, k = 1 / ln(m) so e_j lies in [0, 1]:

         e_j = -k * sum_i p_ij * ln(p_ij),      with 0*ln 0 := 0

     [Shannon, Bell System Technical Journal 27, 1948]

  4. Degree of divergence and the resulting weight:

         d_j = 1 - e_j
         w_j = d_j / sum_j d_j

     [Zeleny, "Multiple Criteria Decision Making", McGraw-Hill, 1982]

  5. SAW aggregation:

         V_i = sum_j w_j * r_ij          -- highest V_i wins

NOTE ON STEP 2. Entropy is computed on the ALREADY direction-normalised
matrix R, not on raw X. Running it on raw X would measure the dispersion
of "error" for cost criteria and of "accuracy" for benefit criteria -- two
different things -- and a criterion's weight would then depend on the unit
it happens to be expressed in. Normalising first makes every column a
comparable "goodness" score before its dispersion is measured. State this
choice when reporting: it is a documented variant, not the only one.

TWO THINGS THIS METHOD CANNOT DO, WHICH YOU MUST HANDLE YOURSELF
----------------------------------------------------------------
1. Entropy weights measure DISPERSION, not IMPORTANCE. A criterion on
   which every model scores alike gets weight ~0 even if it is the single
   most safety-critical number in the table. That is correct information
   theory and possibly wrong engineering. Use --subjective_weights to
   multiply in a prior if the purely objective ranking contradicts what
   the application actually requires.

2. Entropy has no notion of redundancy. Feed it abs_rel, sq_rel, rmse and
   rmse_log together and the error family carries roughly four times the
   weight of any single-column family -- not because error matters four
   times more, but because you listed it four times. Worse, fps is exactly
   1000 / lat_mean_ms, so including both counts one quantity twice under
   two names. DEFAULT_CRITERIA therefore takes ONE representative per
   family; override with --criteria only if you know why.

Usage:
    python -m wheelchair_nav.evaluation.dss_saw_entropy \
        --csv ./wheelchair_nav/log_sunrgbd/depth_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# One representative per correlated family -- see the redundancy note above.
#   abs_rel      : error        (stands in for sq_rel / rmse / rmse_log)
#   a1           : accuracy     (stands in for a2 / a3)
#   params_m     : memory cost
#   macs_g_ref   : compute cost, at one common resolution across models
#   lat_p95_ms   : latency tail (NOT fps -- that is 1000/lat_mean_ms)
DEFAULT_CRITERIA = {
    "abs_rel": "cost",
    "a1": "benefit",
    "params_m": "cost",
    "macs_g_ref": "cost",
    "lat_p95_ms": "cost",
}

# Columns whose direction is known, so --criteria can name any of them
# without the user having to also declare benefit/cost.
KNOWN_DIRECTIONS = {
    "abs_rel": "cost", "sq_rel": "cost", "rmse": "cost", "rmse_log": "cost",
    "a1": "benefit", "a2": "benefit", "a3": "benefit",
    "params_m": "cost", "macs_g": "cost", "macs_g_ref": "cost",
    "fps": "benefit",
    "lat_mean_ms": "cost", "lat_std_ms": "cost", "lat_min_ms": "cost",
    "lat_p50_ms": "cost", "lat_p90_ms": "cost", "lat_p95_ms": "cost",
    "lat_p99_ms": "cost", "lat_max_ms": "cost",
    "map50": "benefit", "map50_95": "benefit", "precision": "benefit", "recall": "benefit",
}

# Pairs that are the same quantity twice; warned about rather than blocked,
# since a user may deliberately want a redundancy sensitivity check.
_REDUNDANT_PAIRS = [
    ("fps", "lat_mean_ms", "fps is exactly 1000 / lat_mean_ms"),
    ("macs_g", "macs_g_ref", "both are MACs, differing only in the resolution measured at"),
]


def read_matrix(csv_path: str, criteria: dict):
    """Returns (names, X, directions) for the requested criteria.

    Rows missing or non-positive on any criterion are dropped with a named
    reason: SAW's cost branch divides by x_ij, and entropy takes ln, so a
    zero anywhere yields inf/NaN instead of a ranking.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv_path} has no data rows.")

    missing = [c for c in criteria if c not in rows[0]]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing column(s): {', '.join(missing)}.\n"
            "Regenerate it with evaluation/eval_depth_comparison.py --out_csv "
            "(macs_g_ref needs thop; lat_* columns come from the latency sampler)."
        )

    names, X, skipped = [], [], []
    for r in rows:
        vals, bad = [], None
        for c in criteria:
            try:
                v = float(r[c])
            except (TypeError, ValueError):
                bad = f"{c}=blank"
                break
            if not np.isfinite(v) or v <= 0:
                bad = f"{c}={v:g}"
                break
            vals.append(v)
        if bad:
            skipped.append((r.get("model", "?"), bad))
            continue
        names.append(r.get("model", f"row{len(names)}"))
        X.append(vals)

    if skipped:
        print("Skipped -- every criterion must be finite and > 0 "
              "(SAW divides by cost values, entropy takes ln):")
        for n, why in skipped:
            print(f"  {n}: {why}")
        print()

    if len(names) < 2:
        raise SystemExit(
            f"Only {len(names)} scorable alternative(s); entropy needs at least 2 "
            "(k = 1/ln(m) is undefined for m = 1)."
        )
    return names, np.asarray(X, dtype=np.float64), [criteria[c] for c in criteria]


def saw_normalize(X: np.ndarray, directions) -> np.ndarray:
    """Linear scale transformation. Benefit: x/max. Cost: min/x.
    Every column ends up in (0, 1] with 1.0 = best in that column.
    """
    R = np.empty_like(X)
    for j, d in enumerate(directions):
        col = X[:, j]
        R[:, j] = col / col.max() if d == "benefit" else col.min() / col
    return R


def entropy_weights(R: np.ndarray):
    """Shannon entropy weights over the direction-normalised matrix.

    Returns (w, e, d). A column whose values are all identical carries no
    information: e_j = 1, d_j = 0, w_j = 0 -- correct, it cannot separate
    the alternatives.
    """
    m = R.shape[0]
    col_sums = R.sum(axis=0)
    P = R / col_sums                       # columns now sum to 1
    k = 1.0 / np.log(m)

    # 0*ln0 := 0 by the limit; np.where alone would still evaluate log(0)
    # and emit a warning, so mask the zeros before taking the log.
    safe = np.where(P > 0, P, 1.0)
    e = -k * (P * np.log(safe)).sum(axis=0)
    e = np.clip(e, 0.0, 1.0)               # guard float drift past the bounds

    d = 1.0 - e
    total = d.sum()
    if total <= 0:
        raise SystemExit(
            "Every criterion has zero divergence -- all alternatives score identically "
            "on all criteria, so no weighting can separate them."
        )
    return d / total, e, d


def saw_score(R: np.ndarray, w: np.ndarray) -> np.ndarray:
    """V_i = sum_j w_j * r_ij."""
    return R @ w


def _print_ranking(title, names, V, extra=None):
    order = np.argsort(-V)
    print(title)
    for rank, i in enumerate(order, 1):
        line = f"  {rank}. {names[i]:<24} V = {V[i]:.4f}"
        if extra is not None:
            line += extra(i)
        print(line)
    return order


def main():
    args = parse_args()

    if args.criteria:
        criteria = {}
        for tok in args.criteria.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                name, direction = tok.split(":", 1)
                name, direction = name.strip(), direction.strip().lower()
                if direction not in ("benefit", "cost"):
                    raise SystemExit(f"--criteria: '{direction}' must be benefit or cost")
            else:
                name = tok
                if name not in KNOWN_DIRECTIONS:
                    raise SystemExit(
                        f"--criteria: unknown direction for '{name}'. "
                        f"Write it as '{name}:cost' or '{name}:benefit'."
                    )
                direction = KNOWN_DIRECTIONS[name]
            criteria[name] = direction
        if len(criteria) < 2:
            raise SystemExit("--criteria needs at least 2 columns.")
    else:
        criteria = dict(DEFAULT_CRITERIA)

    for a, b, why in _REDUNDANT_PAIRS:
        if a in criteria and b in criteria:
            print(f"WARNING: '{a}' and '{b}' are both selected, but {why}. "
                  f"Entropy has no notion of redundancy, so this quantity will carry "
                  f"roughly double weight.\n")

    names, X, directions = read_matrix(args.csv, criteria)
    cols = list(criteria)
    m, n = X.shape

    print(f"Decision support: SAW normalisation + Shannon-entropy weights")
    print(f"Source: {args.csv}")
    print(f"{m} alternatives x {n} criteria: "
          + ", ".join(f"{c}{'^' if d == 'benefit' else 'v'}" for c, d in criteria.items()))
    print()

    R = saw_normalize(X, directions)
    w, e, d = entropy_weights(R)

    print("=" * 78)
    print("STEP 1 -- SAW normalisation   benefit: x/max   cost: min/x")
    print("=" * 78)
    head = f"  {'Alternative':<24}" + "".join(f"{c:>13}" for c in cols)
    print(head)
    for i, nm in enumerate(names):
        print(f"  {nm:<24}" + "".join(f"{R[i, j]:>13.4f}" for j in range(n)))

    print()
    print("=" * 78)
    print("STEP 2 -- Shannon entropy weights")
    print("=" * 78)
    print(f"  {'Criterion':<24}{'e_j':>13}{'d_j = 1-e_j':>15}{'w_j':>13}{'w_j (%)':>12}")
    for j, c in enumerate(cols):
        print(f"  {c:<24}{e[j]:>13.6f}{d[j]:>15.6f}{w[j]:>13.4f}{w[j] * 100:>11.2f}%")
    print(f"  {'':<24}{'':>13}{'':>15}{w.sum():>13.4f}{100.0:>11.2f}%")

    if args.subjective_weights:
        prior = json.loads(args.subjective_weights) if args.subjective_weights.strip().startswith("{") \
            else json.load(open(args.subjective_weights))
        missing = [c for c in cols if c not in prior]
        if missing:
            raise SystemExit(f"--subjective_weights is missing: {', '.join(missing)}")
        s = np.array([float(prior[c]) for c in cols], dtype=np.float64)
        if (s <= 0).any():
            raise SystemExit("--subjective_weights must all be > 0.")
        s = s / s.sum()
        if args.synthesis == "multiplicative":
            combined = w * s
            combined = combined / combined.sum()
            formula = "w_j* = (w_j x s_j) / sum(w_j x s_j)"
            label = "combined, multiplicative"
        else:
            lam = args.synthesis_lambda
            combined = lam * w + (1.0 - lam) * s
            combined = combined / combined.sum()
            formula = f"w_j* = {lam:g}*w_j + {1 - lam:g}*s_j"
            label = f"combined, additive (lambda={lam:g})"

        print()
        print(f"  Combined weighting  {formula}")
        print(f"  {'Criterion':<24}{'entropy w_j':>13}{'prior s_j':>13}{'combined':>13}")
        for j, c in enumerate(cols):
            print(f"  {c:<24}{w[j]:>13.4f}{s[j]:>13.4f}{combined[j]:>13.4f}")
        if args.synthesis == "multiplicative" and (combined < 0.02).any():
            print("\n  NOTE: multiplicative synthesis cannot rescue a criterion entropy has")
            print("  driven to ~0 -- the product stays near zero however large the prior.")
            print("  Use --synthesis additive if a low-dispersion criterion must still count.")
        w_used = combined
    else:
        w_used, label = w, "entropy"

    V = saw_score(R, w_used)

    print()
    print("=" * 78)
    print(f"STEP 3 -- SAW aggregation   V_i = sum_j w_j * r_ij   [weights: {label}]")
    print("=" * 78)
    order = _print_ranking("", names, V)
    winner = names[order[0]]
    print(f"\n  => Most efficient by this model: {winner}")

    # Robustness: the same alternatives scored with equal weights. If the
    # winner survives both, the conclusion does not rest on the weighting
    # scheme; if it does not, that is a finding to report, not to hide.
    V_eq = saw_score(R, np.full(n, 1.0 / n))
    order_eq = np.argsort(-V_eq)
    print()
    print("=" * 78)
    print("ROBUSTNESS -- same matrix, equal weights (w_j = 1/n)")
    print("=" * 78)
    _print_ranking("", names, V_eq)
    if names[order_eq[0]] == winner:
        print(f"\n  Winner unchanged under equal weights -- '{winner}' does not depend "
              "on the entropy weighting.")
    else:
        print(f"\n  NOTE: equal weights favour '{names[order_eq[0]]}' instead of '{winner}'. "
              "The ranking IS sensitive to the weighting scheme; report both.")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["model"] + [f"r_{c}" for c in cols] + ["V", "rank", "V_equal_weights"])
            ranks = {int(i): r for r, i in enumerate(order, 1)}
            for i, nm in enumerate(names):
                wr.writerow([nm] + [f"{R[i, j]:.6f}" for j in range(n)]
                            + [f"{V[i]:.6f}", ranks[i], f"{V_eq[i]:.6f}"])
            wr.writerow([])
            wr.writerow(["criterion", "direction", "entropy_e", "divergence_d", "weight_w"])
            for j, c in enumerate(cols):
                wr.writerow([c, criteria[c], f"{e[j]:.6f}", f"{d[j]:.6f}", f"{w_used[j]:.6f}"])
        print(f"\nSaved to {args.out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="./wheelchair_nav/log_sunrgbd/depth_comparison.csv",
                    help="Table from evaluation/eval_depth_comparison.py --out_csv")
    p.add_argument("--criteria", default=None,
                    help="Comma-separated columns, optionally 'name:benefit' / 'name:cost'. "
                         "Directions for known columns are inferred. Default: "
                         + ", ".join(f"{c}:{d}" for c, d in DEFAULT_CRITERIA.items()))
    p.add_argument("--subjective_weights", default=None,
                    help="Optional JSON file or inline JSON of per-criterion priors, combined "
                         "with the entropy weights. Use this when a criterion matters more than "
                         "its dispersion suggests.")
    p.add_argument("--synthesis", choices=["multiplicative", "additive"], default="multiplicative",
                    help="How --subjective_weights combine with the entropy weights. "
                         "multiplicative (default) w*=w.s/sum(w.s) keeps entropy dominant and "
                         "CANNOT lift a criterion entropy zeroed out; additive "
                         "w*=lambda.w+(1-lambda).s can. Ignored without --subjective_weights.")
    p.add_argument("--synthesis_lambda", type=float, default=0.5,
                    help="Share of the entropy weights under --synthesis additive (0..1); "
                         "0.5 weighs objective and subjective equally")
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
