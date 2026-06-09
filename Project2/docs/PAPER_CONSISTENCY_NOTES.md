# Paper consistency notes

This file maps the implementation in `spectral_pic_part3.cpp` to the Part III paper.

## Scope of this teaching code

The paper introduces a family of gauge-conserving methods and compares BDF1, BDF2, CDF2, and DIRK2. This repository implements the **unstaggered BDF1 and BDF2** members of that family. CDF2 and DIRK2 are intentionally omitted to keep one compact, readable serial C++ file.

The two paper examples are both included:

1. Weibel instability.
2. Drifting cloud of electrons.

For both examples the code includes a charge-conserving run and a naive redeposition run so the same comparison logic as the paper figures can be reproduced.

## Potential formulation

The code evolves the Lorenz-gauge potential form of Maxwell's equations,

```text
(1/c^2) phi_tt - Delta phi = rho/eps0
(1/c^2) A_tt   - Delta A   = mu0 J
(1/c^2) phi_t + div(A) = 0
```

and computes physical fields from

```text
E = -grad(phi) - A_t
B = curl(A)
```

All spatial derivatives are computed spectrally with the in-file FFT.

## BDF field update

For BDF-k, the first derivative at the new time is

```text
D_t u^{n+1} = (a0 u^{n+1} + a1 u^n + ... + ak u^{n+1-k}) / dt.
```

The wave equation uses the nested derivative `D_t(D_t u)`, giving convolution coefficients of the BDF coefficients. The resulting Helmholtz problem is diagonal in Fourier space:

```text
[(b0)/(c^2 dt^2) + |k|^2] u_hat^{n+1}
  = source_hat^{n+1} - sum_{ell>=1} b_ell u_hat^{n+1-ell}/(c^2 dt^2).
```

The implemented coefficient sets are:

```text
BDF1: a = [1, -1]
BDF2: a = [3/2, -2, 1/2]
```

## Continuity update

The charge-conserving option advances total charge density with the same BDF first-derivative operator as the field update:

```text
D_t rho^{n+1} + div(J^{n+1}) = 0.
```

In code this is

```text
rho^{n+1} = -(dt div(J^{n+1}) + a1 rho^n + ... + ak rho^{n+1-k}) / a0.
```

The naive comparison option instead redeposits charge from the particle locations. It is expected to have much larger fully discrete gauge and Gauss errors.

## Improved asymmetric Euler method (IAEM)

The particle step follows the paper's IAEM outline:

```text
x^{n+1} = x^n + v^n dt
v*      = 2 v^n - v^{n-1}
P^{n+1} = P^n + q[-grad(phi^{n+1}) + grad(A^{n+1}) dot v*] dt
v^{n+1} = c^2(P^{n+1} - q A^{n+1})
          / sqrt(c^2 |P^{n+1} - q A^{n+1}|^2 + (m c^2)^2)
```

The generalized momentum stored in each particle is

```text
P = m gamma v + q A.
```

The code is 2D in space but keeps three velocity, momentum, current, and vector-potential components. In the provided examples `v_z` remains zero.

## Particle weighting

Current and charge scatter/gather use quadratic tensor-product B-spline weights. Current is deposited at the updated particle position `x^{n+1}` using the old velocity `v^n`, matching the paper's algorithm outline.

Unlike the energy-conserving orbit-integral codes, this explicit teaching code does not split particle paths at spline knots. That split-orbit machinery belongs to the earlier implicit energy-conserving examples, not to this compact Part III spectral teaching code.

## Spectral consistency and Nyquist mode

The FFT is a standard complex radix-2 transform. The collocated real-grid Nyquist mode is zeroed in the wave-number map. This avoids an ambiguous real-grid first derivative at the Nyquist frequency and ensures that the wave solve, gradient, divergence, gauge diagnostic, and Gauss diagnostic all use the same discrete spectral operator.

## Validation philosophy

The paper figures do not provide raw numeric data. The validation script therefore checks the paper-level claims rather than trying to digitize curves:

- With the continuity update, Lorenz-gauge residuals are orders of magnitude smaller than with naive rho redeposition.
- With BDF1/BDF2, Gauss-law residuals are also orders of magnitude smaller.
- The conserving runs satisfy the discrete continuity equation to roundoff.
- The Weibel magnetic magnitude grows from the initial perturbation.
- Particle speeds stay below `c`.
