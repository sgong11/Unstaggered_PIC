# Code not needed for manuscript runs

These labels describe scope; they are not instructions to delete shared code.
All implementations were retained so historical behavior and tests remain
inspectable. Search the C++ source for `NOT NEEDED` and `VALIDATION ONLY`.

## Uncalled legacy/reference helpers

The following functions have no active caller outside this unused group (the
recursive cardinal spline helper calls itself):

- `sqr`, `make_vec3`, `transpose_matvec3`, `zero_field`;
- `cardinal_bspline`, `cardinal_bspline_derivative` (the production kernel uses
  the explicit `centered_bspline` and its derivative);
- `gather_shape_gradient_vector_at` (the interpolated variant is used instead);
- `evaluate_picard_map_orbit` and its `PicardTrial` container (legacy simultaneous
  orbit-velocity map, superseded by the nested current/particle solve);
- `physical_velocity_vector`, `enforce_initial_velocity_bound`.

These are marked `NOT NEEDED` at their function headers. Compiler
`[[maybe_unused]]` alone is not proof of dead code: for example,
`gather_scalar_at` is used by the field-probe diagnostics and must be kept.

## Retained optional branches outside manuscript scope

- Landau initialization, normal-quantile sampling, Landau-only configuration and
  theory settings. The manuscript experiments initialize TSI or Weibel.
- Adaptive field-step shrink/regrowth and associated configuration. The driver
  rejects `adaptive_dt=true`. Some helpers still produce config/log values;
  deleting them alone would break compilation.
- Uniform particle subcycling and its scheduling/refinement controls. The driver
  rejects `uniform_particle_subcycling=true`. The non-selective uniform-map
  evaluation branch is retained but not used by manuscript dispatch.
- Degree-three/four spline options, the artificial transverse TSI seed, and
  unsplit-orbit mode are optional experiments, not the supplied manuscript decks.
- Legacy relaxed-Picard input aliases are accepted/ignored for compatibility.

**Keep the shared functions despite their historical names:**
`nonlinear_step_uniform_subcycled` supplies the outer current iteration for both
the full-dt main runs and the selective auxiliary runs.
`evaluate_selective_current_map_at_layout` is also required by full-dt production:
its layout consists entirely of ones in that case. The local substep solve,
stored particle paths, and subcycled diagnostic machinery are shared kernels.
The restored selective fallback is required for the spline comparison.

## Validation-only and optional outputs

`test_*` functions run only under `run_unit_tests=true`. They are unnecessary
for each expensive production run, but remain useful evidence for the numerical
implementation. Do not classify them as dead code.

Full field-grid snapshots are off in the manuscript decks. Spectra, line probes,
particle snapshots, and all-particle moments are diagnostics rather than time
integration, but the comparisons need the relevant diagnostic files. In
particular, keep the shape-gradient reconstruction and knot-jump writers for
the spline comparison.

## Package contents

The active suite is `inputs/manuscript/*.txt`. `run_manuscript.sh` runs it locally;
`submit_manuscript.sh` submits it to Slurm. `tests/` and `validate.py` verify the
solver and analysis. Historical inputs, archives, and generated results were
removed during package cleanup. New runs create their own output directories.
