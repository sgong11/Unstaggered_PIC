# Validation performed for this package

The package was built and smoke-tested with:

```bash
make clean
make test
```

Compiler:

```text
g++ -std=c++17 -O2 -Wall -Wextra -pedantic
```

The build completed without compiler warnings after marking intentionally retained helper functions as `[[maybe_unused]]`.

## Two-stream smoke test

Command:

```bash
./ec_pic_serial_demo examples/input_two_stream_smoke.txt
python3 tools/summarize_diagnostics.py two_stream_smoke
```

Summary from the generated diagnostics:

```text
max relative total-energy drift: 1.587867e-16
max Gauss RMS residual:          7.666467e-20
max Lorenz-gauge RMS residual:   5.989427e-22
max A-chain RMS residual:        1.322110e-21
max abs step total-energy error: 1.512282e-15
max particle-energy residual:    1.512248e-15
max field-energy residual:       6.776264e-21
```

## Weak Landau damping example

Command:

```bash
./ec_pic_serial_demo examples/input_landau_weak.txt
python3 tools/summarize_diagnostics.py landau_weak --fit-min 0.0 --fit-max 3.0
```

Summary from the generated diagnostics:

```text
max relative total-energy drift: 1.470409e-16
max Gauss RMS residual:          1.369485e-15
max Lorenz-gauge RMS residual:   2.079016e-16
max A-chain RMS residual:        1.551217e-17
max abs step total-energy error: 9.168556e-16
max particle-energy residual:    9.150040e-16
max field-energy residual:       1.138412e-18
final |E_k| / linear envelope:   2.874823e-01
min/max |E_k| / envelope:        2.661076e-03 / 1.009212e+00
fitted gamma from log |E_k|:     -2.724027e-01
```

The fitted damping rate depends on the short serial teaching resolution, particle quiet start, and chosen fit window. The theory columns in the CSV provide the reference envelope used for comparison.

## Strong Landau damping example

Command:

```bash
./ec_pic_serial_demo examples/input_landau_strong.txt
python3 tools/summarize_diagnostics.py landau_strong --fit-min 0.0 --fit-max 2.0
```

Summary from the generated diagnostics:

```text
max relative total-energy drift: 6.406173e-15
max Gauss RMS residual:          6.069848e-14
max Lorenz-gauge RMS residual:   9.363399e-15
max A-chain RMS residual:        7.896088e-16
max abs step total-energy error: 5.218048e-15
max particle-energy residual:    4.593548e-15
max field-energy residual:       3.011480e-15
final |E_k| / linear envelope:   1.726235e-01
min/max |E_k| / envelope:        1.582008e-03 / 1.009211e+00
fitted gamma from log |E_k|:     -1.397975e+00
```

For the strong Landau case, the linear theory comparison is an early-time reference only; the run is nonlinear by construction.
