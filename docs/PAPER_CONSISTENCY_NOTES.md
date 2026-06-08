# Paper consistency notes

This file maps the main implementation choices in `ec_pic_serial_demo.cpp` to the paper draft.

## Field update

The code evolves the first-order potential variables `phi`, `psi`, `A`, and `U` with a Crank-Nicolson wave update. The physical fields are recovered as

```text
E = -grad(phi) - U
B = curl(A)
```

The charge density is not redeposited during the step. Instead, the code deposits the orbit-averaged current first and advances charge by

```text
rho^{n+1} = rho^n - dt div(J^{n+1/2}).
```

This is the source ordering used for the discrete Lorenz-gauge and Gauss-law preservation mechanism.

## Particle update

The code stores canonical momentum

```text
P = m v + q A_h(x)
```

and updates it with

```text
P^{n+1} = P^n + dt [ -q grad(phi)_bar + q D_A^T v_bar ].
```

`D_A` is the orbit-discrete-gradient of the same mesh interpolant `A_h` used in the canonical momentum. It is not a spectral gradient gathered to a particle.

## Orbit splitting

When `split_orbit_at_knots = true`, every particle path is split at crossed spline knots before quadrature. This is the implementation detail that makes the finite-difference chain rule for `A_h` hold to quadrature/roundoff accuracy.

The intentionally unsplit comparison input is retained only to show the failure mode.

## Diagnostics

The implementation writes the diagnostics used by the proof:

```text
Gauss residual:             div(E) - rho/eps0
Lorenz-gauge residual:      psi/c^2 + div(A)
A-chain residual:           A_h^{n+1}(x^{n+1}) - A_h^n(x^n) - dt(U_bar + D_A v_bar)
particle work residual:     delta_K - particle_work
field work residual:        delta_W + grid_work
total energy residual:      delta_K + delta_W
```

For split-orbit converged runs, these should be near Picard tolerance, quadrature error, and roundoff.
