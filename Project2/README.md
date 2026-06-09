# Spectral Generalized-Momentum PIC: Part III Teaching Code

This repository contains a **small serial C++17 teaching implementation** for the spectral generalized-momentum particle-in-cell ideas used in

> Andrew J. Christlieb, William A. Sands, and Stephen R. White,  
> *A Particle-in-cell Method for Plasmas with A Generalized Momentum Formulation, Part III: A family of Gauge Conserving Methods*,  
> Journal of Scientific Computing 104:38, 2025. DOI: `10.1007/s10915-025-02953-7`.

The code is intentionally self contained. It uses no MPI, no OpenMP, and no FFTW. The file `spectral_pic_part3.cpp` includes its own radix-2 FFT and is heavily commented so the method can be read and modified by students.

## What this code demonstrates

The paper's central numerical message is that the field update, the continuity equation, and the Lorenz gauge must be time-discretized consistently. This teaching code demonstrates that point with the two paper examples:

1. **Weibel instability** in a periodic 2D domain.
2. **Drifting cloud of electrons** in a periodic 2D domain.

For each example, the repository includes two variants:

- `charge_update = conserving`: deposit current and update charge from the same BDF continuity equation used by the field solver.
- `charge_update = naive`: redeposit charge directly from particle locations. This is intentionally included as the comparison case, mirroring the left/right comparisons in the paper figures.

For the unstaggered teaching version, the code includes **BDF1** and **BDF2** spectral wave solvers. The paper also studies CDF2 and DIRK2; those are not included here so the code remains short and focused on the requested unstaggered serial example.

## Files

```text
spectral_pic_part3.cpp                      # self-contained C++17 source with internal FFT
Makefile                                    # build and run targets
examples/smoke_weibel_bdf1_conserving.txt   # fast smoke test
examples/weibel_bdf1_conserving.txt         # Weibel, BDF1, continuity rho
examples/weibel_bdf1_naive.txt              # Weibel, BDF1, redeposited rho
examples/weibel_bdf2_conserving.txt         # Weibel, BDF2, continuity rho
examples/weibel_bdf2_naive.txt              # Weibel, BDF2, redeposited rho
examples/cloud_bdf1_conserving.txt          # drifting cloud, BDF1, continuity rho
examples/cloud_bdf1_naive.txt               # drifting cloud, BDF1, redeposited rho
examples/cloud_bdf2_conserving.txt          # drifting cloud, BDF2, continuity rho
examples/cloud_bdf2_naive.txt               # drifting cloud, BDF2, redeposited rho
tools/validate_against_paper.py             # automated validation of paper-level claims
tools/summarize_diagnostics.py              # compact summary of one run
notebooks/plot_results.ipynb                # commented notebook for reading/plotting CSV output
docs/PAPER_CONSISTENCY_NOTES.md             # mapping between paper equations and this code
docs/VALIDATION.md                          # validation results from this package
```

## Build

Requirements: a C++17 compiler and `make`.

```bash
make
```

This creates

```bash
./spectral_pic_part3
```

The executable has no runtime dependencies beyond the C++ standard library.

## Quick smoke test

```bash
make smoke
```

or explicitly:

```bash
./spectral_pic_part3 examples/smoke_weibel_bdf1_conserving.txt
```

The smoke test also runs a small built-in FFT/spectral-derivative unit test.

## Run the paper examples

Run the Weibel comparison set:

```bash
make weibel
```

Run the drifting-cloud comparison set:

```bash
make cloud
```

Run everything and validate the paper-level claims:

```bash
make validate
```

The validation script runs all eight example decks and checks that the charge-conserving runs have much smaller Lorenz-gauge and Gauss-law errors than the naive redeposition runs. It writes:

```text
validation_summary.json
```

## Manual run syntax

Every run uses a plain text input deck:

```bash
./spectral_pic_part3 examples/weibel_bdf1_conserving.txt
```

A deck contains key-value pairs:

```text
example = weibel
method = bdf1
charge_update = conserving
nx = 32
ny = 32
n_steps = 160
dt = 0.05
output_prefix = weibel_bdf1_conserving
```

Unknown keys are ignored with a warning, so it is safe to add notes to an input file.

## Output files

For `output_prefix = weibel_bdf1_conserving`, the code writes:

```text
weibel_bdf1_conserving_diagnostics.csv
weibel_bdf1_conserving_particle_sample.csv
weibel_bdf1_conserving_config_echo.txt
```

The main diagnostics columns are:

```text
gauge_rms        RMS of (1/c^2) d_t phi + div(A)
gauss_rms        RMS of div(E) - rho/eps0
continuity_rms   RMS of d_t rho + div(J)
Bz_l2            RMS magnetic field component B_z
B_l2             RMS magnetic magnitude
E_l2             RMS electric magnitude
rho_l2           RMS charge density
J_l2             RMS current density
field_energy     electromagnetic field energy
kinetic_energy   mobile electron kinetic energy
total_energy     field_energy + kinetic_energy
max_speed        maximum particle speed
mean_gamma       mean relativistic gamma over mobile particles
```

## How the examples map to the paper

### Weibel instability

The paper's Weibel parameter table uses a periodic square domain, two counter-streaming electron sheets, a neutralizing stationary ion background, drift in the x direction, and a small y-velocity perturbation. The default teaching deck uses normalized units:

```text
c = 1
n0 = 1
weibel_vx = 0.5          # c/2
weibel_vy_max = 0.01     # c/100
```

The code also adds a tiny smooth sinusoidal y-velocity seed so the magnetic mode grows clearly even with a small serial particle count:

```text
weibel_sine_perturb = 0.002
```

### Drifting cloud of electrons

The paper's cloud example uses a domain enlarged to `[-8,8]^2`, a stationary Gaussian ion cloud, and a matching mobile Gaussian electron cloud with drift `c/100` in x and y. The teaching code stores positions internally in `[0,16)^2`, which is equivalent under periodic boundary conditions, and writes centered coordinates in the particle sample file.

The Gaussian width is not tabulated in the paper, so it is exposed as an input parameter:

```text
cloud_sigma = 1.0
cloud_drift = 0.01       # c/100
cloud_thermal = 0.005
```

## Plotting results

After running examples, open:

```text
notebooks/plot_results.ipynb
```

The notebook reads the generated CSV files and plots:

- Weibel magnetic growth, corresponding to the role of Fig. 1 in the paper.
- Lorenz-gauge error for naive vs charge-conserving updates, corresponding to Figs. 2 and 5.
- Gauss-law error for naive vs charge-conserving updates, corresponding to Figs. 3 and 6.
- Final particle locations for the drifting-cloud run.

## Important implementation notes

- The code uses a collocated periodic grid and computes all spatial derivatives spectrally.
- The same discrete wave numbers are used for the wave solve, gradients, divergence, gauge residual, and Gauss residual.
- The Nyquist wave number is zeroed. This is a standard real-grid convention that avoids an ambiguous first derivative at the Nyquist mode and keeps the discrete identities consistent.
- Current is deposited at `x^{n+1}` using `v^n`, matching the paper's algorithm outline.
- The IAEM momentum update uses `v* = 2 v^n - v^{n-1}`.
- Initial BDF histories are filled with repeated copies of the initial fields. This is a simple teaching start-up procedure, not a high-order production initializer.

## Clean generated files

```bash
make clean
```
