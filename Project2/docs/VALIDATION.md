# Validation performed for this package

Validation command:

```bash
make clean
make validate
```

Compiler:

```text
g++ -std=c++17 -O2 -Wall -Wextra -pedantic
```

The build completed without compiler warnings.

## What is validated

The Part III paper emphasizes that updating charge with the continuity equation consistent with the field solver greatly improves the fully discrete Lorenz gauge error, and for the BDF/CDF methods also improves Gauss's law. This teaching package validates the same claim for the included unstaggered BDF1 and BDF2 solvers.

The validation script runs all eight input decks:

```text
weibel_bdf1_conserving    weibel_bdf1_naive
weibel_bdf2_conserving    weibel_bdf2_naive
cloud_bdf1_conserving     cloud_bdf1_naive
cloud_bdf2_conserving     cloud_bdf2_naive
```

It checks:

1. The conserving run has Lorenz-gauge error at least `1e6` times smaller than the matching naive run.
2. The conserving run has Gauss-law error at least `1e6` times smaller than the matching naive run.
3. The conserving run satisfies `d_t rho + div(J)` to less than `1e-10` RMS.
4. The Weibel magnetic field grows by more than a factor of 10.
5. All diagnostics are finite and all particle speeds stay below `0.95 c`.

## Summary from the generated diagnostics

```text
weibel bdf1: gauge ratio=4.394e-13, Gauss ratio=2.020e-13, continuity=3.473e-17
weibel bdf2: gauge ratio=4.306e-13, Gauss ratio=1.748e-13, continuity=8.578e-17
cloud  bdf1: gauge ratio=8.718e-13, Gauss ratio=8.260e-13, continuity=3.046e-17
cloud  bdf2: gauge ratio=5.001e-12, Gauss ratio=7.125e-12, continuity=2.612e-16

weibel_bdf1_conserving: Bz growth=2.552e+02
weibel_bdf2_conserving: Bz growth=5.018e+02

PASS: teaching-code validation checks match the paper-level claims.
```

Here the gauge and Gauss ratios are

```text
max_error(conserving) / max_error(naive)
```

for the same example and method. Smaller is better.

## Notes on comparison to the published plots

The paper's figures show full-resolution production calculations, typically on a `128 x 128` mesh. The default decks here are deliberately smaller so they run quickly in serial on a laptop. The validation therefore checks the robust qualitative and algebraic claims shown in the paper figures rather than matching every plotted data point.

The Weibel decks use normalized versions of the Table 2 values: `c=1`, `n0=1`, `v_x=c/2`, and `v_y` sampled from `[-c/100,c/100)`. The drifting-cloud decks use normalized versions of the Table 3 values: a periodic `[-8,8]^2` domain represented internally as `[0,16)^2`, with electron drift `c/100` in x and y.
