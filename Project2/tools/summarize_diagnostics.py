#!/usr/bin/env python3
"""Summarize diagnostics from one Part III spectral PIC teaching-code run.

Examples
--------
python3 tools/summarize_diagnostics.py weibel_bdf1_conserving
python3 tools/summarize_diagnostics.py cloud_bdf2_naive
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def max_abs(rows: List[Dict[str, str]], key: str) -> float:
    vals = finite(to_float(r.get(key, "nan")) for r in rows[1:])
    return max((abs(v) for v in vals), default=math.nan)


def final(rows: List[Dict[str, str]], key: str) -> float:
    vals = finite(to_float(r.get(key, "nan")) for r in rows)
    return vals[-1] if vals else math.nan


def first_positive(rows: List[Dict[str, str]], key: str) -> float:
    for r in rows:
        v = to_float(r.get(key, "nan"))
        if math.isfinite(v) and v > 0.0:
            return v
    return math.nan


def summarize(prefix: str) -> None:
    path = Path(f"{prefix}_diagnostics.csv")
    if not path.exists():
        raise SystemExit(f"missing {path}; run the C++ executable first")
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"{path} is empty")

    print(f"Diagnostics prefix: {prefix}")
    print(f"  rows:                         {len(rows)}")
    print(f"  final time:                   {final(rows, 'time'):.6e}")
    print(f"  max Lorenz-gauge RMS:         {max_abs(rows, 'gauge_rms'):.6e}")
    print(f"  max Gauss-law RMS:            {max_abs(rows, 'gauss_rms'):.6e}")
    print(f"  max continuity RMS:           {max_abs(rows, 'continuity_rms'):.6e}")
    print(f"  final Bz RMS:                 {final(rows, 'Bz_l2'):.6e}")
    first_b = first_positive(rows, 'Bz_l2')
    if math.isfinite(first_b) and first_b > 0.0:
        print(f"  Bz growth from first nonzero: {final(rows, 'Bz_l2') / first_b:.6e}")
    print(f"  final total energy:           {final(rows, 'total_energy'):.6e}")
    print(f"  max particle speed / c:       {max_abs(rows, 'max_speed'):.6e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="output prefix, for example weibel_bdf1_conserving")
    args = parser.parse_args()
    summarize(args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
