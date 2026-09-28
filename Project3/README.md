# Manuscript PIC simulations and analysis

Clean C++17 package for the relativistic energy-conserving GM–HC–CN PIC method.
It contains all eight manuscript case inputs, local and Slurm runners, diagnostic
checks, and an analysis notebook. The reference manuscript is in
[docs/manuscript.pdf](docs/manuscript.pdf).

## Folder contents

| File or folder | Purpose |
|---|---|
| `ec_pic_nyquist_projected.cpp` | Documented solver and built-in numerical tests |
| `inputs/manuscript/` | Eight complete manuscript input decks |
| `run_manuscript.sh` | Compile once and run a suite locally |
| `submit_manuscript.sh`, `run_case.sb` | Submit and execute independent Slurm jobs |
| `summarize_hpc_run.py` | Check completion, convergence, and conservation |
| `summarize_snapshots.py` | Compute all-particle transverse RMS from snapshots |
| `plot_pic_saved_analysis.ipynb`, `pic_analysis.py` | Single-run plots and paired comparisons |
| `requirements.txt` | Python packages for notebook/analysis |
| `validate.py`, `tests/` | Repeatable reduced numerical and analysis checks |
| `docs/` | Manuscript reference, limitations, code notes, prior validation report |
| `SHA256SUMS` | Checksums of the distributed files |

Old inputs, historical job logs, duplicate archives, generated results, caches,
and the machine-specific Python environment have been removed. The solver's
marked legacy branches remain in the source; this cleanup removes unused files,
not numerical implementation code. Future execution creates `runs/`,
`analysis/`, `validation/`, or `submissions/` as needed.

## Cases

| Input stem | dt | Steps | End time | Spline degree | Projection | Selective refinement |
|---|---:|---:|---:|---:|---|---|
| `tsi_main_dt01_T60` | 0.1 | 600 | 60 | 2 | on | off |
| `tsi_main_dt03_T60` | 0.3 | 200 | 60 | 2 | on | off |
| `weibel_main_dt004_T100` | 0.04 | 2500 | 100 | 2 | on | off |
| `weibel_main_dt02_T100` | 0.2 | 500 | 100 | 2 | on | off |
| `tsi_projection_off_dt01_T60` | 0.1 | 600 | 60 | 2 | off | off |
| `tsi_projection_on_dt01_T60` | 0.1 | 600 | 60 | 2 | on | off |
| `tsi_spline_r1_dt01_T60` | 0.1 | 600 | 60 | 1 | on | on |
| `tsi_spline_r2_dt01_T60` | 0.1 | 600 | 60 | 2 | on | on |

All use a 32³ grid, v0=0.9, κ=σ1=σ2=ρ0=1, eight-point Gauss quadrature on
knot-split orbits, fixed field dt, outer tolerances 1e-11/1e-13, outer limit 80,
and inner limit 64. TSI uses a box of side 8π, 10 pairs/cell, density eigenmode
amplitude 0.005, and inner tolerances 1e-11/1e-13. Weibel uses a box of side 2π,
25 pairs/cell, Bz seed 1e-4, and inner tolerances 1e-12/1e-14.

The projection pair differs only in projection/output prefix. The spline pair
differs only in degree/output prefix. Both spline runs allow the same selective
refinement on local convergence failure; field dt stays fixed. See
[docs/README.md](docs/README.md) for the reproduction limits and policy details.

## Run locally

Requirements: a C++17 compiler, Bash, and Python 3.9 or newer. No external FFT
library or Python packages are needed to run the solver and summary scripts.

```bash
# From this folder: compile and run all eight cases, sequentially.
bash run_manuscript.sh all

# Or select one suite:
bash run_manuscript.sh main
bash run_manuscript.sh projection
bash run_manuscript.sh spline

# GNU/OpenMP build on a suitable machine:
CXX=g++ OPENMP=1 OMP_NUM_THREADS=16 bash run_manuscript.sh all
```

The default is a serial build compatible with Apple clang. Full 32³ cases can
be expensive; for a quick check use `python3 validate.py` instead. Each local
batch gets a unique `runs/local_<timestamp>_<pid>/` directory containing the
source, executable, compiler version, copied summary scripts, and one directory
per case. Every case retains its input, simulation log, diagnostics, and summary.
Inspect `status.tsv`: all three exit-status columns must be zero for success.
Failures are recorded while the remaining independent cases continue.

To compile and run just one input manually:

```bash
c++ -std=c++17 -O2 ec_pic_nyquist_projected.cpp -o /tmp/pic
mkdir -p runs/my_tsi
cd runs/my_tsi
/tmp/pic ../../inputs/manuscript/tsi_main_dt01_T60.txt
python3 ../../summarize_hpc_run.py tsi_main_dt01_T60
cd ../..
```

The solver itself can overwrite an existing output prefix; use a fresh directory.
The suite runner creates fresh directories automatically.

## Run on Slurm

Load your cluster's GNU C++/OpenMP environment and submit from this folder:

```bash
bash submit_manuscript.sh all --account=YOUR_ACCOUNT --partition=YOUR_PARTITION
# Replace all with main, projection, or spline to submit a subset.
```

Each job requests one node, 16 CPUs, 32 GB, and 24 hours; adjust `run_case.sb`
for your cluster. Outputs go to `runs/<case>/<job_id>/`. Each job records its
source/input hashes, compiler, environment, logs, and diagnostics. Per-case locks
and submission records under `submissions/` prevent duplicate suite submission.
Inspect these records after a partial submission before retrying missing jobs.
No production runs or cluster submissions are performed by package cleanup.

## Analyze results

Create a fresh Python environment for the notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open `plot_pic_saved_analysis.ipynb` in a Jupyter-capable editor and select this
Python environment. Run the notebook from this package directory so that
`pic_analysis.py` is importable. Set the first cell's `DATA_DIR` to one case's
output directory and `PREFIX` to its filename prefix, for example:

```python
DATA_DIR = Path("runs/local_TIMESTAMP_PID/tsi_main_dt01_T60").resolve()
PREFIX = "tsi_main_dt01_T60"
```

Replace `local_TIMESTAMP_PID` with the directory printed by the runner. With no
data, the notebook gives setup guidance and skips plots. Missing optional files
are reported; invalid required data raise descriptive errors.

For paired comparisons, configure `COMPARISONS` in the same cell:

```python
batch = Path("runs/local_TIMESTAMP_PID")
COMPARISONS = {
    "projection": [str(batch / name / name) for name in (
        "tsi_projection_off_dt01_T60", "tsi_projection_on_dt01_T60")],
    "spline": [str(batch / name / name) for name in (
        "tsi_spline_r1_dt01_T60", "tsi_spline_r2_dt01_T60")],
}
```

Slurm runs use `runs/<case>/<job_id>/<prefix>` instead. The paired analysis
requires completed runs with matching recorded controls. Figures and tables
are saved in new timestamped `analysis/` directories under `DATA_DIR`; inputs
remain unchanged. Growth fits use [0,2] for TSI and [0,1.6] for Weibel and need
at least five finite positive samples.

## Important diagnostic files

| Suffix | Contents |
|---|---|
| `_config_echo.txt` | Effective settings and derived parameters |
| `_diagnostics.csv` | Energies, modes, Gauss/gauge residuals, final solver counts |
| `_step_diagnostics.csv` | Orbit chain-rule and work/energy residuals |
| `_particle_inner_history.csv` | Inner-map evaluations, including failed refinement trials |
| `_adaptive_diagnostics.csv` | Accepted/rejected field-step attempts; historical filename |
| `_velocity_moments.csv` | All-particle velocity statistics |
| `_field_spectrum.csv` | Fourier powers |
| `_field_line_probe.csv`, `_field_knot_jumps.csv` | Field reconstructions and spline regularity |
| `_particle_sample.csv`, `_particles_t*.csv` | Final particle sample and full snapshots |
| `_verification_summary.json`, `_verification_summary.md` | Completion and conservation checks |

Screening thresholds are relative energy 1e-9, Gauss/gauge RMS 1e-9, orbit chain
RMS 1e-10, and energy-normalized work residuals 1e-10. Recovered local failures
are counted separately for selective-refinement runs. These screens do not
establish physical accuracy or reproduce the manuscript's quoted numbers.

## Validate the installation

```bash
python3 validate.py
# Use the newly printed validation directory:
python tests/check_analysis.py validation/TIMESTAMP/manifest.json
python tests/check_notebook.py validation/TIMESTAMP/manifest.json
```

The first command compiles the solver and checks all eight decks on reduced
4³ grids with six steps. The latter commands require the notebook environment;
fresh Jupyter kernels require local loopback sockets. A prior validation record
is in [docs/VALIDATION.md](docs/VALIDATION.md). Function documentation and retained
optional code are described in [docs/CODE_NOTES.md](docs/CODE_NOTES.md).
