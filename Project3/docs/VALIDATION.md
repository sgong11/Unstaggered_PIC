# Local validation of the documented revision

Validation date: 2026-09-25. Serial C++17 build using Apple clang, optimization O2.
No Slurm jobs or full-resolution production simulations were run.

This report records the validation performed before cleanup. Generated run data,
executed notebooks, and the temporary Python environment were removed from the
clean package. The validation scripts are retained so the checks can be repeated.

## Completed checks

- Compiled the revised solver and focused C++ regression harness successfully.
- All **16 built-in numerical regression tests passed**, including spectral
  Nyquist compatibility, field/continuity consistency, HC static-field tests,
  canonical/mechanical equivalence, orbit adjointness and chain rules,
  split-knot crossings, Gauss law, energy conservation, instability
  initialization/growth, and selective mixed-layout conservation.
- Two additional regressions passed: projection off/identity and on/removal,
  idempotence and self-adjointness; and forced local failure through the public
  timestep dispatcher. The latter verifies off-mode failure, on-mode refinement,
  unchanged field dt, and unchanged accepted input state.
- All eight manuscript decks completed six reduced steps and passed the
  conservation summary. The reduced decks use 4³ cells, two pairs/cell,
  initial-only full-particle snapshots, and reduced probe/output settings.
  Physical parameters, dt, tolerances, spline degree, and projection/refinement
  switches remain those of the respective manuscript deck.
- The projection pair differs only in projection/output prefix; the spline pair
  differs only in degree/output prefix. Both paired analyses passed matching
  configuration and completion checks on the reduced outputs.
- Deliberately duplicated-setting, mismatched-dt, incomplete, and nonfinite
  pairs were rejected. Failed-run NaN data produced a valid strict JSON failure
  report. An injected recoverable local trial was allowed only when selective
  refinement was explicitly enabled and the final inner evaluation converged.
- The notebook executed in five fresh Jupyter kernels with **no cell errors**:
  no selected data; TSI; Weibel; missing optional tables; and both paired
  comparisons. Required malformed or missing inputs intentionally remain errors.
  The checker saves executed notebooks in the validation directory on each run. Representative
  growth/energy and spline-comparison PNG plots were visually inspected.
- Reduced TSI-dt0.1 and Weibel-dt0.04 runs were also executed with the preserved
  original source. Every non-timing value in endpoint, step, velocity-moment,
  and inner-history CSVs matched the revised full-step solver exactly.
- Python syntax checks and Bash syntax checks passed for the analysis, summary,
  validation, and submission scripts. Package hashes were refreshed and checked.

## Reduced-run residuals

These values demonstrate local numerical consistency; they are not manuscript
production measurements.

| Reduced case | Max relative energy error | Max Gauss RMS | Max gauge RMS |
|---|---:|---:|---:|
| TSI main dt=0.1 | 2.831e-15 | 1.672e-18 | 2.316e-19 |
| TSI main dt=0.3 | 1.415e-15 | 2.588e-18 | 2.489e-18 |
| TSI projection off | 2.831e-15 | 1.716e-18 | 2.373e-19 |
| TSI projection on | 2.831e-15 | 1.672e-18 | 2.316e-19 |
| TSI spline r=1 | 2.123e-15 | 2.015e-18 | 1.247e-19 |
| TSI spline r=2 | 2.831e-15 | 1.672e-18 | 2.316e-19 |
| Weibel main dt=0.04 | 4.781e-15 | 4.747e-21 | 1.195e-20 |
| Weibel main dt=0.2 | 3.187e-15 | 1.832e-20 | 1.635e-20 |

## Repeating the checks

From this repository, with notebook requirements installed:

```bash
python3 validate.py
python tests/check_analysis.py validation/NEW_TIMESTAMP/manifest.json
python tests/check_notebook.py validation/NEW_TIMESTAMP/manifest.json
```

Use `python3 validate.py --cxx g++ --openmp` on a suitable GNU/OpenMP system.
The local Jupyter checks require loopback sockets for the kernel. The supplied
working notebook has cleared outputs so old figures are not mistaken for a fresh
analysis. Historical notebook outputs were removed during cleanup.

Full-grid, long-time, OpenMP, and cluster integration tests remain outstanding;
see [manuscript notes](README.md) for the publication implications.

## Clean-package runner check

After cleanup, `run_manuscript.sh all` was tested in a temporary copy with all
eight inputs reduced to 4³ grids, two pairs/cell, and six steps. Every solver,
conservation summary, and snapshot summary returned zero. The temporary copy
was removed. Bash syntax and Python/notebook cell syntax checks also passed.
The numerical solver and notebook logic were unchanged during cleanup.
