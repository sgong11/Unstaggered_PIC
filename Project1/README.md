# Energy-Conserving Unstaggered Potential PIC - Serial Teaching Code

This repository contains a **serial C++17 teaching implementation** of the energy-conserving unstaggered potential particle-in-cell method described in the accompanying paper: An Energy-Conserving Unstaggered Electromagnetic-Potential Particle-in-Cell Method, Part I: Non-relativistic Generalized-Momentum Formulation (with the manuscript released on the arxiv within a few weeks).

The code is serial on purpose. There is no MPI, no OpenMP, and no external FFT library. The goal is to make the algorithm readable for students and community users. A parallel implementation can be released later; this version is meant to teach the ideas and provide a compact reference implementation.

## What the code demonstrates

The algorithm follows the paper's conservative potential formulation:

1. Particles carry canonical momentum `P = m v + q A_h(x)`.
2. The current `J^{n+1/2}` is deposited from orbit-averaged particle paths.
3. The charge density is advanced from the discrete continuity equation, not redeposited directly after the initial condition.
4. The scalar and vector potentials are advanced with a Crank-Nicolson first-order wave solve.
5. The particle push uses an orbit-discrete-gradient of the mesh-interpolated vector potential `A_h`.
6. Particle orbits are split at crossed spline knots so the orbit integrals satisfy the discrete chain rule used by the energy proof.

The implementation writes diagnostics for Gauss's law, the Lorenz gauge, particle/field/total energy, the vector-potential chain-rule residual, and particle-mesh work balance.

## Files

```text
ec_pic_serial_demo.cpp                 # self-contained serial C++17 source
Makefile                               # build and example targets
examples/input_two_stream_smoke.txt     # quick two-stream smoke test
examples/input_two_stream_paper_demo.txt# 3D two-stream demo based on paper parameters
examples/input_two_stream_unsplit_comparison.txt
examples/input_landau_weak.txt          # weak Landau damping example
examples/input_landau_strong.txt        # strong Landau damping example
tools/summarize_diagnostics.py          # optional CSV summary helper
docs/VALIDATION.md                      # validation runs performed for this package
```

## Build

Requirements: a C++17 compiler and `make`.

```bash
make
```

This creates

```bash
./ec_pic_serial_demo
```

The executable has no runtime dependencies beyond the standard C++ library.

## Quick run

Run the fastest end-to-end test:

```bash
make smoke
```

or explicitly:

```bash
./ec_pic_serial_demo examples/input_two_stream_smoke.txt
```

This run also executes the built-in unit tests:

- orbit chain-rule residual `R_A,p`,
- split versus unsplit orbit crossing check,
- initial Gauss-law check,
- one-step Gauss-law and energy-coupling check.

## Two-stream instability example

The paper demonstration is a 3D cold two-stream instability with periodic domain `[0,2*pi]^3`, two counter-streaming electron populations, a fixed neutralizing ion background, and an `m=1` sinusoidal velocity perturbation.

Run a shortened serial demonstration:

```bash
./ec_pic_serial_demo examples/input_two_stream_paper_demo.txt
```

The input deck uses the paper-style parameters but shortens the run for a teaching code:

```text
nx = ny = nz = 16
dt = 0.025
v0 = 0.3
perturbation = 0.02
perturbation_mode = 1
particles_per_cell_pair = 16
split_orbit_at_knots = true
```

To reproduce the paper final time `T=30`, change

```text
n_steps = 1200
```

in `examples/input_two_stream_paper_demo.txt`. This will be much slower in the serial version.

### Split versus unsplit orbit comparison

For teaching, an intentionally nonconservative input is included:

```bash
./ec_pic_serial_demo examples/input_two_stream_unsplit_comparison.txt
```

This sets

```text
split_orbit_at_knots = false
```

The run is expected to keep Gauss and gauge residuals small, but the vector-potential chain-rule residual and energy drift should be worse. This demonstrates why the split orbit is required for the energy theorem.

## Weak Landau damping example

Run:

```bash
./ec_pic_serial_demo examples/input_landau_weak.txt
python3 tools/summarize_diagnostics.py landau_weak --fit-min 0.0 --fit-max 3.0
```

This is a 1D-1V Landau damping problem embedded in the same 3D code path with `ny=nz=1`. It uses

```text
Lx = 4*pi
perturbation_mode = 1     # k = 2*pi/Lx = 0.5
thermal_velocity = 1.0
landau_alpha = 0.01
```

The input deck includes the standard linear Vlasov-Poisson reference for `k=0.5`, `v_th=1`, `omega_p=1`:

```text
theory_omega = 1.4156618886
theory_gamma = -0.1533594669
```

The diagnostics file writes `E_mode_abs`, `theory_E_abs`, and `E_over_theory`, where

```text
theory_E_abs = E_mode_abs(t=0) * exp(theory_gamma * t)
```

## Strong Landau damping example

Run:

```bash
./ec_pic_serial_demo examples/input_landau_strong.txt
python3 tools/summarize_diagnostics.py landau_strong --fit-min 0.0 --fit-max 2.0
```

This uses the same setup as the weak Landau case but with

```text
landau_alpha = 0.5
```

The same linear theory columns are written as an **early-time reference only**. The strong case is nonlinear, so trapping and phase-space distortion should not be expected to follow the linear exponential envelope at late time.

## Output files

For an input with

```text
output_prefix = landau_weak
```

the code writes:

```text
landau_weak_diagnostics.csv
landau_weak_step_diagnostics.csv
landau_weak_particle_sample.csv
landau_weak_unit_tests.txt
landau_weak_config_echo.txt
```

### Main diagnostics CSV

`*_diagnostics.csv` contains one row per time level. Important columns include:

```text
gauss_rms                 RMS of div(E) - rho/eps0
gauge_rms                 RMS of psi/c^2 + div(A)
field_energy              electromagnetic field energy
kinetic_energy            particle kinetic energy
total_energy              field + kinetic energy
rho_mode                  charge-density Fourier amplitude for perturbation_mode
E_mode_abs                |E_x(k,0,0)| for perturbation_mode
theory_E_abs              Landau linear envelope, if theory_gamma is supplied
E_over_theory             E_mode_abs / theory_E_abs
picard_iterations         number of Picard iterations used on that step
crossing_particles        particles whose path crossed at least one spline knot
```

### Step diagnostics CSV

`*_step_diagnostics.csv` contains per-step conservation checks:

```text
A_chain_abs_rms                    RMS of the vector-potential chain-rule residual
delta_total_energy                 total energy change over the step
particle_work                      orbit-averaged particle work
grid_work                          mesh J dot E work
deposit_gather_work_residual       particle work minus grid work
particle_energy_residual           delta_K minus particle_work
field_energy_residual              delta_W plus grid_work
```

For a converged split-orbit run, these should be close to Picard tolerance, quadrature error, and roundoff.

## Where to change the code for a new initial condition

Most users should start by copying an input deck in `examples/` and changing parameters. To add a genuinely new initial condition in C++:

1. Open `ec_pic_serial_demo.cpp`.
2. Add any new input parameters to `struct Config`.
3. Parse the new keys in `read_config()`.
4. Add a new function near the existing initializers, for example:

   ```cpp
   static State initialize_my_problem(const Config& cfg, const Grid& g) {
       State st;
       // Fill st.x, st.v, st.q, st.m.
       // Deposit initial electron charge, add the fixed ion background, and subtract the mean.
       // Solve the periodic Poisson equation for phi.
       // Set psi, A, U, P, and v_prev consistently.
       return st;
   }
   ```

5. Add a branch in `initialize_state()`:

   ```cpp
   if (tc == "my_problem") return initialize_my_problem(cfg, g);
   ```

6. Create `examples/input_my_problem.txt` with

   ```text
   test_case = my_problem
   output_prefix = my_problem
   ```

The rest of the code - Picard iteration, current deposition, continuity update, Crank-Nicolson potential solve, orbit-discrete-gradient pusher, and diagnostics - is independent of the initial condition.

## Notes on normalization

The code uses SI-like nondimensional constants `c`, `eps0`, and `mu0`. Keep

```text
c^2 * mu0 = 1 / eps0
```

when changing these constants, because this compatibility is what makes the field update, Gauss-law propagation, and energy diagnostics consistent.

## Clean generated files

```bash
make clean
```

This removes the executable and generated CSV/text output files in the repository root.
