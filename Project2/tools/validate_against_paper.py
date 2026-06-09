#!/usr/bin/env python3
"""Run and validate the Part III spectral PIC teaching examples.

This script does not attempt pixel-level reproduction of the paper figures because
raw figure data are not distributed with the article.  Instead, it checks the
main numerical claims that the paper figures demonstrate:

1. In the Weibel and drifting-cloud examples, the charge-conserving update gives
   much smaller fully discrete Lorenz-gauge error than naive rho redeposition.
2. For the unstaggered BDF methods included here, the charge-conserving update
   also gives much smaller Gauss-law error.
3. The conserving runs satisfy the discrete continuity equation to roundoff.
4. The Weibel magnetic magnitude grows from the initial perturbation.
5. All runs remain finite and subluminal.

The script writes validation_summary.json and prints a compact report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "spectral_pic_part3"

EXAMPLES = [
    "weibel_bdf1_conserving",
    "weibel_bdf1_naive",
    "weibel_bdf2_conserving",
    "weibel_bdf2_naive",
    "cloud_bdf1_conserving",
    "cloud_bdf1_naive",
    "cloud_bdf2_conserving",
    "cloud_bdf2_naive",
]

PAIRS = [
    ("weibel", "bdf1", "weibel_bdf1_conserving", "weibel_bdf1_naive"),
    ("weibel", "bdf2", "weibel_bdf2_conserving", "weibel_bdf2_naive"),
    ("cloud", "bdf1", "cloud_bdf1_conserving", "cloud_bdf1_naive"),
    ("cloud", "bdf2", "cloud_bdf2_conserving", "cloud_bdf2_naive"),
]


def read_rows(prefix: str) -> List[Dict[str, str]]:
    path = ROOT / f"{prefix}_diagnostics.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the executable first or pass --run")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def floats(rows: List[Dict[str, str]], key: str, skip_initial: bool = True) -> List[float]:
    data = rows[1:] if skip_initial else rows
    out: List[float] = []
    for row in data:
        try:
            value = float(row[key])
        except Exception:
            value = math.nan
        out.append(value)
    return out


def finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def max_abs(rows: List[Dict[str, str]], key: str) -> float:
    vals = finite(floats(rows, key))
    return max((abs(v) for v in vals), default=math.nan)


def final_value(rows: List[Dict[str, str]], key: str) -> float:
    vals = finite(floats(rows, key, skip_initial=False))
    return vals[-1] if vals else math.nan


def first_positive(rows: List[Dict[str, str]], key: str) -> float:
    for v in floats(rows, key, skip_initial=False):
        if math.isfinite(v) and v > 0.0:
            return v
    return math.nan


def run_examples() -> None:
    if not EXE.exists():
        subprocess.run(["make"], cwd=ROOT, check=True)
    for name in EXAMPLES:
        deck = ROOT / "examples" / f"{name}.txt"
        print(f"running {deck.relative_to(ROOT)}")
        subprocess.run([str(EXE), str(deck)], cwd=ROOT, check=True)


def build_summary() -> Dict[str, object]:
    summary: Dict[str, object] = {"runs": {}, "comparisons": [], "passed": True, "failures": []}

    for name in EXAMPLES:
        rows = read_rows(name)
        run_summary = {
            "max_gauge_rms": max_abs(rows, "gauge_rms"),
            "max_gauss_rms": max_abs(rows, "gauss_rms"),
            "max_continuity_rms": max_abs(rows, "continuity_rms"),
            "max_speed": max_abs(rows, "max_speed"),
            "final_Bz_l2": final_value(rows, "Bz_l2"),
            "final_total_energy": final_value(rows, "total_energy"),
        }
        summary["runs"][name] = run_summary
        for k, v in run_summary.items():
            if not math.isfinite(float(v)):
                summary["passed"] = False
                summary["failures"].append(f"{name}: non-finite {k}")
        if run_summary["max_speed"] >= 0.95:
            summary["passed"] = False
            summary["failures"].append(f"{name}: max_speed >= 0.95 c")

    # Paper-level conserving-vs-naive comparisons.
    for example, method, conserving, naive in PAIRS:
        cr = read_rows(conserving)
        nr = read_rows(naive)
        gauge_ratio = max_abs(cr, "gauge_rms") / max_abs(nr, "gauge_rms")
        gauss_ratio = max_abs(cr, "gauss_rms") / max_abs(nr, "gauss_rms")
        cont = max_abs(cr, "continuity_rms")
        comp = {
            "example": example,
            "method": method,
            "conserving": conserving,
            "naive": naive,
            "gauge_ratio_conserving_over_naive": gauge_ratio,
            "gauss_ratio_conserving_over_naive": gauss_ratio,
            "conserving_continuity_rms": cont,
        }
        summary["comparisons"].append(comp)

        if not (gauge_ratio < 1.0e-6):
            summary["passed"] = False
            summary["failures"].append(f"{example}/{method}: gauge ratio {gauge_ratio:.3e} is not < 1e-6")
        if not (gauss_ratio < 1.0e-6):
            summary["passed"] = False
            summary["failures"].append(f"{example}/{method}: gauss ratio {gauss_ratio:.3e} is not < 1e-6")
        if not (cont < 1.0e-10):
            summary["passed"] = False
            summary["failures"].append(f"{example}/{method}: continuity residual {cont:.3e} is not < 1e-10")

    # Weibel magnetic growth check.  The paper's Fig. 1 focuses on growth of the
    # magnetic magnitude; the exact growth rate depends on resolution, particle
    # count, and normalization here, so we only require clear growth.
    for name in ["weibel_bdf1_conserving", "weibel_bdf2_conserving"]:
        rows = read_rows(name)
        initial = first_positive(rows, "Bz_l2")
        final = final_value(rows, "Bz_l2")
        growth = final / initial if initial and math.isfinite(initial) else math.nan
        summary["runs"][name]["Bz_growth_from_first_nonzero"] = growth
        if not (math.isfinite(growth) and growth > 10.0):
            summary["passed"] = False
            summary["failures"].append(f"{name}: Bz growth {growth:.3e} is not > 10")

    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\nValidation summary")
    print("==================")
    for comp in summary["comparisons"]:  # type: ignore[index]
        print(
            f"{comp['example']:6s} {comp['method']:4s}: "
            f"gauge ratio={comp['gauge_ratio_conserving_over_naive']:.3e}, "
            f"Gauss ratio={comp['gauss_ratio_conserving_over_naive']:.3e}, "
            f"continuity={comp['conserving_continuity_rms']:.3e}"
        )
    for name in ["weibel_bdf1_conserving", "weibel_bdf2_conserving"]:
        growth = summary["runs"][name]["Bz_growth_from_first_nonzero"]  # type: ignore[index]
        print(f"{name}: Bz growth={growth:.3e}")

    if summary["passed"]:
        print("\nPASS: teaching-code validation checks match the paper-level claims.")
    else:
        print("\nFAIL:")
        for item in summary["failures"]:  # type: ignore[index]
            print(f"  - {item}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run all example input decks before validating")
    args = parser.parse_args()

    if args.run:
        run_examples()

    summary = build_summary()
    out = ROOT / "validation_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print_summary(summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
