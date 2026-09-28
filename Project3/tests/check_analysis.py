"""Regression checks for paired-run rejection and failure-report serialization."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pic_analysis import compare_runs


# ---------------------------------------------------------------------------
# Require a deliberately invalid comparison to raise a descriptive error
# Inputs:
#   prefixes : pair of saved output prefixes; kind : projection or spline
# Output:
#   None; raises AssertionError if invalid inputs are accepted
# Dependencies:
#   - pic_analysis.compare_runs
# ---------------------------------------------------------------------------
def rejected(prefixes, kind):
    try:
        compare_runs(prefixes, kind)
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid comparison was accepted')


# ---------------------------------------------------------------------------
# Test real reduced outputs, then mutate temporary copies into failure cases
# Inputs:
#   CLI manifest : Path, validation manifest produced by validate.py
# Output:
#   console pass/fail; temporary copies are removed automatically
# Dependencies:
#   - compare_runs, rejected, standard library; numpy/pandas via analysis module
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    runs = json.loads(args.manifest.read_text())['runs']
    projection = [runs[f'tsi_projection_{v}_dt01_T60'] for v in ('off','on')]
    spline = [runs[f'tsi_spline_r{v}_dt01_T60'] for v in (1,2)]
    assert len(compare_runs(projection, 'projection')[0]) == 2
    assert len(compare_runs(spline, 'spline')[0]) == 2
    rejected([projection[0], projection[0]], 'projection')
    with tempfile.TemporaryDirectory(prefix='pic-analysis-') as tmp:
        folder = Path(tmp)
        prefix = folder/'copy'
        for path in Path(projection[0]).parent.glob(Path(projection[0]).name+'_*'):
            if path.is_file():
                shutil.copyfile(path, folder/('copy'+path.name[len(Path(projection[0]).name):]))
        config_path = Path(str(prefix)+'_config_echo.txt')
        original_config = config_path.read_text()
        config_path.write_text(original_config.replace('dt=0.10000000000000001', 'dt=0.2'))
        rejected([prefix, projection[1]], 'projection')
        config_path.write_text(original_config)
        diag_path = Path(str(prefix)+'_diagnostics.csv')
        original_diag = diag_path.read_text()
        diag_path.write_text('\n'.join(original_diag.splitlines()[:-1])+'\n')
        rejected([prefix, projection[1]], 'projection')
        diag_path.write_text(original_diag)
        # Failed runs with NaN moments must still write a strict JSON report.
        moments_path = Path(str(prefix)+'_velocity_moments.csv')
        with moments_path.open() as stream:
            reader=csv.DictReader(stream); columns=reader.fieldnames; rows=list(reader)
        rows[-1]['rms_vy']='nan'
        with moments_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        rejected([prefix, projection[1]], 'projection')
        result=subprocess.run([sys.executable,str(ROOT/'summarize_hpc_run.py'),str(prefix)],capture_output=True,text=True)
        assert result.returncode==1, result.stderr
        report=json.loads(Path(str(prefix)+'_verification_summary.json').read_text())
        assert report['screening_pass'] is False
        assert report['final_weighted_vperp_rms'] is None
        # An expected failed trial followed by a converged map is allowed only
        # under the explicit refinement policy. Use an unmodified good run.
        source=Path(spline[0])
        for path in source.parent.glob(source.name+'_*'):
            if path.is_file():shutil.copyfile(path,folder/('copy'+path.name[len(source.name):]))
        history=Path(str(prefix)+'_particle_inner_history.csv')
        lines=history.read_text().splitlines()
        fields=lines[1].split(',');fields[3:7]=['0','1','inf','nan']
        history.write_text('\n'.join([lines[0],','.join(fields),*lines[1:]])+'\n')
        result=subprocess.run([sys.executable,str(ROOT/'summarize_hpc_run.py'),str(prefix)],capture_output=True,text=True)
        assert result.returncode==0, result.stdout+result.stderr
        report=json.loads(Path(str(prefix)+'_verification_summary.json').read_text())
        assert report['failed_inner_evaluations']==1 and report['final_inner_evaluations_converged']
        config_path.write_text(config_path.read_text().replace('selective_particle_fallback=true','selective_particle_fallback=false'))
        result=subprocess.run([sys.executable,str(ROOT/'summarize_hpc_run.py'),str(prefix)],capture_output=True,text=True)
        assert result.returncode==1, result.stdout+result.stderr
    print('PASS: both paired analyses; same-setting/mismatch/incomplete/nonfinite rejection; strict failure JSON; refinement-aware screening')


if __name__ == '__main__':
    main()
