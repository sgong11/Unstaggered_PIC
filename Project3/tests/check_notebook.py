"""Execute the delivered notebook against reduced solver data and missing-data cases."""
import argparse
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Execute a fresh Jupyter kernel with explicitly selected saved data
# Inputs:
#   label : str, result filename; env : dict, per-kernel data selection
#   out : Path, validation output directory; source : notebook document
# Output:
#   executed notebook with no error outputs; exceptions fail validation
# Dependencies:
#   - nbclient.NotebookClient, nbformat, ipykernel, current Python environment
# ---------------------------------------------------------------------------
def execute(label, env, out, source):
    notebook = nbformat.reads(nbformat.writes(source), as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name='rel-validation',
                            resources={'metadata': {'path': str(ROOT)}})
    client.execute(env=env)
    assert not any(o.output_type == 'error' for c in notebook.cells if c.cell_type == 'code' for o in c.outputs)
    nbformat.write(notebook, out/f'notebook_{label}.ipynb')
    print('PASS notebook:', label, flush=True)


# ---------------------------------------------------------------------------
# Check empty-folder, TSI, Weibel, optional-file, and both paired workflows
# Inputs:
#   CLI manifest : Path, manifest.json produced by validate.py
# Output:
#   five executed notebooks and console status under the validation directory
# Dependencies:
#   - execute, pathlib, json, os, nbformat
# ---------------------------------------------------------------------------
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text()); out=args.manifest.resolve().parent
    kernel_dir=out/'jupyter/kernels/rel-validation';kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Validation','language':'python'}))
    os.environ['JUPYTER_PATH']=str(out/'jupyter')
    os.environ['JUPYTER_RUNTIME_DIR']=str(out/'jupyter/runtime')
    source=nbformat.read(ROOT/'plot_pic_saved_analysis.ipynb',as_version=4)
    nbformat.validate(source)
    env=dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(out/'matplotlib'), IPYTHONDIR=str(out/'ipython'), PIC_PREFIX='', PIC_DATA_DIR=str(ROOT), PIC_COMPARISONS='{}')
    execute('no_data',env,out,source)
    for name in ['tsi_main_dt01_T60','weibel_main_dt004_T100']:
        prefix=Path(manifest['runs'][name])
        execute(name,dict(env,PIC_PREFIX=prefix.name,PIC_DATA_DIR=str(prefix.parent)),out,source)
    optional=out/'optional_missing';optional.mkdir(exist_ok=True)
    name='tsi_main_dt01_T60';prefix=Path(manifest['runs'][name])
    for suffix in ['_config_echo.txt','_diagnostics.csv']:
        (optional/(name+suffix)).write_bytes(Path(str(prefix)+suffix).read_bytes())
    execute('optional_missing',dict(env,PIC_PREFIX=name,PIC_DATA_DIR=str(optional)),out,source)
    pairs={'projection':[manifest['runs'][f'tsi_projection_{v}_dt01_T60'] for v in ('off','on')],
           'spline':[manifest['runs'][f'tsi_spline_r{v}_dt01_T60'] for v in (1,2)]}
    execute('comparisons',dict(env,PIC_DATA_DIR=str(out),PIC_COMPARISONS=json.dumps(pairs)),out,source)


if __name__=='__main__':
    main()
