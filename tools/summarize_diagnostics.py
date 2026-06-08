#!/usr/bin/env python3
"""Summarize diagnostics from the serial energy-conserving PIC teaching code.

This script intentionally uses only the Python standard library so it can be run
on a minimal system after the C++ example finishes.

Examples
--------
python3 tools/summarize_diagnostics.py two_stream_smoke
python3 tools/summarize_diagnostics.py landau_weak --fit-min 0.0 --fit-max 3.0
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Dict


def _float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def _linear_fit(xs: List[float], ys: List[float]) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0.0:
        return None
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
    intercept = ybar - slope * xbar
    return slope, intercept


def summarize(prefix: str, fit_min: float | None, fit_max: float | None) -> None:
    diag_path = Path(f"{prefix}_diagnostics.csv")
    step_path = Path(f"{prefix}_step_diagnostics.csv")
    if not diag_path.exists():
        raise SystemExit(f"missing {diag_path}; run the C++ executable first")

    diag = _read_csv(diag_path)
    if not diag:
        raise SystemExit(f"{diag_path} is empty")

    total_energy = [_float(r.get("total_energy", "nan")) for r in diag]
    e0 = total_energy[0]
    rel_drift = [abs((e - e0) / e0) for e in total_energy if math.isfinite(e) and e0 != 0.0]
    gauss = _finite(_float(r.get("gauss_rms", "nan")) for r in diag)
    gauge = _finite(_float(r.get("gauge_rms", "nan")) for r in diag)

    print(f"Diagnostics prefix: {prefix}")
    if rel_drift:
        print(f"  max relative total-energy drift: {max(rel_drift):.6e}")
    if gauss:
        print(f"  max Gauss RMS residual:          {max(gauss):.6e}")
    if gauge:
        print(f"  max Lorenz-gauge RMS residual:   {max(gauge):.6e}")

    if step_path.exists():
        step = _read_csv(step_path)
        a_chain = _finite(_float(r.get("A_chain_abs_rms", "nan")) for r in step)
        de_step = _finite(_float(r.get("delta_total_energy", "nan")) for r in step)
        particle_res = _finite(_float(r.get("particle_energy_residual", "nan")) for r in step)
        field_res = _finite(_float(r.get("field_energy_residual", "nan")) for r in step)
        if a_chain:
            print(f"  max A-chain RMS residual:        {max(a_chain):.6e}")
        if de_step:
            print(f"  max abs step total-energy error: {max(abs(x) for x in de_step):.6e}")
        if particle_res:
            print(f"  max particle-energy residual:    {max(abs(x) for x in particle_res):.6e}")
        if field_res:
            print(f"  max field-energy residual:       {max(abs(x) for x in field_res):.6e}")

    # Landau theory comparison: the C++ code writes theory_E_abs and
    # E_over_theory when theory_gamma/theory_omega are supplied in the input.
    ratios = _finite(_float(r.get("E_over_theory", "nan")) for r in diag)
    if ratios:
        print(f"  final |E_k| / linear envelope:   {ratios[-1]:.6e}")
        print(f"  min/max |E_k| / envelope:        {min(ratios):.6e} / {max(ratios):.6e}")

    # Optional measured damping/growth estimate from log |E_k|.  We only do this
    # when a fit window is supplied, because a nonlinear or unstable run may not
    # have a single meaningful exponential rate over the whole output interval.
    if fit_min is not None or fit_max is not None:
        times: List[float] = []
        logs: List[float] = []
        for r in diag:
            t = _float(r.get("time", "nan"))
            e = _float(r.get("E_mode_abs", "nan"))
            if not (math.isfinite(t) and math.isfinite(e) and e > 0.0):
                continue
            if fit_min is not None and t < fit_min:
                continue
            if fit_max is not None and t > fit_max:
                continue
            times.append(t)
            logs.append(math.log(e))
        fit = _linear_fit(times, logs)
        if fit is not None:
            slope, intercept = fit
            print(f"  fitted gamma from log |E_k|:     {slope:.6e}")
            print(f"  fitted log-amplitude intercept:  {intercept:.6e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix", help="output prefix, e.g. landau_weak")
    ap.add_argument("--fit-min", type=float, default=None, help="minimum time used for log-amplitude fit")
    ap.add_argument("--fit-max", type=float, default=None, help="maximum time used for log-amplitude fit")
    args = ap.parse_args()
    summarize(args.prefix, args.fit_min, args.fit_max)


if __name__ == "__main__":
    main()
