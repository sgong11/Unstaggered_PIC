"""Compile and run small numerical regressions; never submit an HPC job."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Execute a validation stage and retain its full combined output
# Inputs:
#   command : list[str], executable and arguments
#   cwd     : Path, isolated working directory
#   log     : Path, log destination
# Output:
#   None; raises RuntimeError on failure, preserving the diagnostic log
# Dependencies:
#   - subprocess, os; C++ compiler/solver or current Python interpreter
# ---------------------------------------------------------------------------
def run(command, cwd, log):
    env = dict(os.environ, OMP_NUM_THREADS='2', MPLBACKEND='Agg')
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'Validation failed ({result.returncode}); inspect {log}')


# ---------------------------------------------------------------------------
# Build reduced runs from each manuscript deck and check numerical invariants
# Inputs:
#   CLI --cxx : compiler (default CXX or c++); --openmp enables OpenMP
# Output:
#   timestamped validation directory, smoke outputs, logs, and manifest.json
# Dependencies:
#   - run, pathlib, subprocess, standard Python library; C++17 compiler
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    parser.add_argument('--openmp', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'validation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    flags = ['-std=c++17', '-O2'] + (['-fopenmp'] if args.openmp else [])
    binary = out / 'pic'
    run([args.cxx, *flags, str(ROOT/'ec_pic_nyquist_projected.cpp'), '-o', str(binary)], ROOT, out/'build.log')
    test_binary = out / 'test_solver'
    run([args.cxx, *flags, str(ROOT/'tests/test_solver.cpp'), '-o', str(test_binary)], ROOT, out/'build_tests.log')
    run([str(test_binary)], out, out/'focused_tests.log')
    manifest = {'purpose': 'Reduced numerical smoke tests, NOT manuscript results', 'openmp': args.openmp, 'runs': {}}
    for deck in sorted((ROOT/'inputs/manuscript').glob('*.txt')):
        name = deck.stem
        folder = out / name; folder.mkdir()
        # Six accepted steps exercise the growth fit and optional diagnostics.
        # Preserve physical parameters, dt, spline, projection and tolerances.
        text = deck.read_text()
        updates = dict(nx=4, ny=4, nz=4, particles_per_cell_pair=2, n_steps=6,
                       particle_snapshot_times='0', field_diagnostics_interval='0',
                       transverse_spectrum_interval='0.1', field_probe_points_per_cell=4)
        for key, value in updates.items():
            text, count = re.subn(r'^'+key+r'\s*=.*$', key+' = '+str(value), text, flags=re.M)
            if not count: text += key+' = '+str(value)+'\n'
        (folder/'input.txt').write_text('# REDUCED SMOKE TEST ONLY\n'+text)
        run([str(binary), 'input.txt'], folder, folder/'simulation.log')
        run([sys.executable, str(ROOT/'summarize_hpc_run.py'), name], folder, folder/'screen.log')
        run([sys.executable, str(ROOT/'summarize_snapshots.py'), name], folder, folder/'snapshot_rms.csv')
        manifest['runs'][name] = str(folder/name)
        print('PASS:', name, flush=True)
    unit = out/'unit_tests'; unit.mkdir()
    (unit/'input.txt').write_text('nx=4\nny=4\nnz=4\nn_steps=0\nrun_unit_tests=true\nadaptive_dt=false\noutput_prefix=unit\n')
    run([str(binary), 'input.txt'], unit, unit/'simulation.log')
    print('PASS: built-in solver tests', flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Validation outputs:', out)


if __name__ == '__main__':
    main()
