"""Read-only loading and matched comparisons of saved PIC output.

No function in this module runs the solver or supplies synthetic production data.
"""
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Read one solver configuration echo
#
# Inputs:
#   prefix : str or Path, output prefix including its directory
# Output:
#   dict[str, str] : recorded configuration (legacy inner aliases normalized)
# Dependencies:
#   - pathlib.Path
# ---------------------------------------------------------------------------
def read_config(prefix):
    path = Path(str(prefix) + '_config_echo.txt')
    cfg = dict((k.strip(), v.strip()) for line in path.read_text().splitlines()
               if '=' in line for k, v in [line.split('=', 1)])
    for suffix in ('rtol', 'atol', 'max_iter'):
        cfg.setdefault('particle_local_' + suffix, cfg.get('uniform_local_' + suffix, 'not recorded'))
    return cfg


# ---------------------------------------------------------------------------
# Load a saved diagnostic table and validate its schema
#
# Inputs:
#   prefix   : str or Path, output prefix including its directory
#   suffix   : str, filename part between prefix and .csv
#   required : bool, whether missing or empty data must raise an error
#   columns  : iterable[str], columns required when a table is present
# Output:
#   pandas.DataFrame : parsed table, or empty frame for unavailable optional data
# Dependencies:
#   - pathlib.Path, pandas as pd
# ---------------------------------------------------------------------------
def load_table(prefix, suffix, required=False, columns=()):
    path = Path(f'{prefix}_{suffix}.csv')
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'Required data missing: {path}')
        print('Not available:', path.name)
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
    if frame.empty:
        if required:
            raise ValueError(f'Required table is empty: {path}')
        return frame
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    return frame


# ---------------------------------------------------------------------------
# Convert numpy scalars and nonfinite values to strict JSON-compatible values
#
# Inputs:
#   value : nested dict/list/tuple, scalar, or numpy scalar
# Output:
#   object : JSON-compatible value; NaN and infinity become None
# Dependencies:
#   - numpy as np
# ---------------------------------------------------------------------------
def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Compare two completed runs that differ only in the intended experiment
#
# Inputs:
#   prefixes : sequence of exactly two str/Path output prefixes
#   kind     : 'projection' (off/on) or 'spline' (degrees 1/2)
# Output:
#   table, runs : summary DataFrame and list of loaded per-run tables/configs
#   Raises ValueError for mismatched controls, incomplete or nonfinite data.
# Dependencies:
#   - read_config, load_table, numpy as np, pandas as pd
# ---------------------------------------------------------------------------
def compare_runs(prefixes, kind):
    if kind not in ('projection', 'spline') or len(prefixes) != 2:
        raise ValueError('Supply exactly two prefixes and kind projection or spline')
    varied = 'nyquist_projection' if kind == 'projection' else 'spline_order'
    runs = []
    for prefix in map(Path, prefixes):
        cfg = read_config(prefix)
        required_cfg = ('test_case', 'dt', 'n_steps', 'nyquist_projection', 'spline_order',
                        'selective_particle_fallback', 'adaptive_dt', 'uniform_particle_subcycling')
        if any(k not in cfg for k in required_cfg):
            raise ValueError(f'{prefix}: incomplete configuration echo')
        if 'two_stream' not in cfg['test_case']:
            raise ValueError('The manuscript comparisons require TSI runs')
        if cfg['adaptive_dt'] not in ('false', '0') or cfg['uniform_particle_subcycling'] not in ('false', '0'):
            raise ValueError('Comparison requires fixed field steps and no uniform subcycling')
        columns = ('step', 'time', 'dt', 'total_energy', 'nonlinear_converged')
        diag = load_table(prefix, 'diagnostics', True, columns)
        n = int(cfg['n_steps']); dt = float(cfg['dt']); target = n * dt
        if (len(diag) != n + 1 or not np.array_equal(diag['step'], np.arange(n + 1))
                or not np.allclose(diag['time'], np.arange(n + 1) * dt, rtol=1e-10, atol=1e-10)
                or not np.allclose(diag['dt'].iloc[1:], dt, rtol=1e-10, atol=1e-10)
                or not np.isfinite(diag[list(columns)].iloc[1:]).all().all()
                or not (diag['nonlinear_converged'].iloc[1:] == 1).all()):
            raise ValueError(f'{prefix}: incomplete, nonfinite, or unconverged run')
        moments = load_table(prefix, 'velocity_moments', True, ('step', 'time', 'rms_vy', 'rms_vz'))
        if (len(moments) != len(diag) or not np.array_equal(moments['step'], diag['step'])
                or not np.allclose(moments['time'], diag['time'])
                or not np.isfinite(moments[['rms_vy', 'rms_vz']]).all().all()):
            raise ValueError(f'{prefix}: incomplete or nonfinite velocity moments')
        knots = load_table(prefix, 'field_knot_jumps', kind == 'spline',
                           ('time', 'E_shape_phi_jump', 'gradA_jump')) if kind == 'spline' else pd.DataFrame()
        runs.append(dict(prefix=str(prefix), config=cfg, diagnostics=diag, moments=moments, knots=knots))
    a, b = (r['config'] for r in runs)
    # The finite-grid theory rate changes with spline degree; all other echoed
    # controls must match, including output schedules and refinement policy.
    ignored = {varied, 'output_prefix'}
    if kind == 'spline':
        ignored.add('auto_theory_growth_rate')
    differences = [k for k in sorted(set(a) | set(b)) if k not in ignored and a.get(k) != b.get(k)]
    if differences:
        raise ValueError('Not a matched comparison; configuration differs: ' + ', '.join(differences))
    values = {c[varied].lower() for c in (a, b)}
    if kind == 'projection':
        values = {'true' if v in ('1', 'true') else 'false' if v in ('0', 'false') else v for v in values}
        if values != {'false', 'true'}:
            raise ValueError('Projection comparison needs one off run and one on run')
    elif values != {'1', '2'} or a['nyquist_projection'] not in ('1', 'true') or a['selective_particle_fallback'] not in ('1', 'true'):
        raise ValueError('Spline comparison needs r=1 and r=2, projection on and selective refinement enabled')
    rows = []
    for run in runs:
        diag, moments, knots, cfg = (run[k] for k in ('diagnostics', 'moments', 'knots', 'config'))
        e0 = float(diag['total_energy'].iloc[0])
        if not np.isfinite(e0) or e0 == 0:
            raise ValueError('Invalid initial energy')
        vperp = np.hypot(moments['rms_vy'], moments['rms_vz'])
        row = dict(prefix=run['prefix'], comparison=kind, setting=cfg[varied],
                   final_time=float(diag['time'].iloc[-1]), final_vperp_rms=float(vperp.iloc[-1]),
                   max_relative_energy_error=float(((diag['total_energy'] - e0) / abs(e0)).abs().max()))
        for col in ('E_shape_phi_jump', 'gradA_jump'):
            if col in knots:
                if not np.isfinite(knots[col]).all():
                    raise ValueError(f'{run["prefix"]}: nonfinite knot diagnostic {col}')
                row['max_' + col] = float(knots[col].abs().max())
        if 'selective_refined_particles' in diag:
            refined = diag.loc[diag['selective_refined_particles'] > 0]
            row['refined_field_steps'] = len(refined)
            row['first_refinement_time_n'] = float(refined['time'].iloc[0] - refined['dt'].iloc[0]) if len(refined) else None
            row['max_particle_substeps'] = int(diag['uniform_substeps_used'].max())
        rows.append(row)
    return pd.DataFrame(rows), runs
