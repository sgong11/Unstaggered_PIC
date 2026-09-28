"""Check run completeness and conservation diagnostics; keep physical validation separate."""
from pathlib import Path
import argparse
import csv
import json
import math

ap=argparse.ArgumentParser()
ap.add_argument('prefix')
ap.add_argument('--solver-exit',type=int,default=0)
ap.add_argument('--energy-limit',type=float,default=1e-9)
ap.add_argument('--constraint-limit',type=float,default=1e-9)
ap.add_argument('--chain-limit',type=float,default=1e-10)
ap.add_argument('--work-limit',type=float,default=1e-10,help='Absolute residual divided by max(abs(initial total energy),1)')
args=ap.parse_args()
p=Path(args.prefix)
issues=[]
# ---------------------------------------------------------------------------
# Resolve a diagnostic filename
#
# Inputs:
#   suffix : str, suffix appended to the configured prefix
# Output:
#   Path : output filename
# Dependencies:
#   - pathlib.Path; p : configured prefix
# ---------------------------------------------------------------------------
def file(suffix): return Path(str(p)+suffix)
# ---------------------------------------------------------------------------
# Read a CSV table and report missing diagnostics
#
# Inputs:
#   suffix : str, diagnostic filename suffix
# Output:
#   list[dict] : rows, or empty list when missing
# Dependencies:
#   - file; csv.DictReader; issues : shared diagnostic issue list
# ---------------------------------------------------------------------------
def read(suffix):
    path=file(suffix)
    if not path.exists():
        issues.append('Missing '+path.name); return []
    with path.open() as f:return list(csv.DictReader(f))
# ---------------------------------------------------------------------------
# Measure the maximum absolute finite diagnostic
#
# Inputs:
#   rows : list[dict], parsed CSV; key : str, numeric column
# Output:
#   float or None : absolute maximum, with invalid data added to issues
# Dependencies:
#   - math.isfinite; issues : shared diagnostic issue list
# ---------------------------------------------------------------------------
def maximum(rows,key):
    values=[abs(float(r[key])) for r in rows]
    if not values or any(not math.isfinite(x) for x in values):
        issues.append('Empty/nonfinite diagnostic: '+key);return None
    return max(values)

# ---------------------------------------------------------------------------
# Sanitize failed-run reports without turning nonfinite values into valid data
# Inputs:
#   value : nested dict/list/scalar diagnostic report
# Output:
#   object : JSON-safe report, nonfinite floats represented by None
# Dependencies:
#   - math.isfinite
# ---------------------------------------------------------------------------
def clean_json(value):
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

diag=read('_diagnostics.csv');step=read('_step_diagnostics.csv')
inner=read('_particle_inner_history.csv');moments=read('_velocity_moments.csv')
attempts=read('_adaptive_diagnostics.csv')
cfg={}
if file('_config_echo.txt').exists():
    cfg=dict(line.split('=',1) for line in file('_config_echo.txt').read_text().splitlines() if '=' in line)
else:issues.append('Missing configuration echo')
report={'prefix':str(p),'configuration':cfg,'solver_exit':args.solver_exit,'issues':issues,
        'limits':{'relative_energy':args.energy_limit,'gauss_and_gauge_rms':args.constraint_limit,'chain_rms':args.chain_limit,'normalized_work_and_energy_residuals':args.work_limit}}
if args.solver_exit:issues.append('Solver returned exit status '+str(args.solver_exit))
if diag:
    target=float(cfg.get('target_time','nan'));end=float(diag[-1]['time'])
    nsteps=int(cfg.get('n_steps',-1));dt=float(cfg.get('dt','nan'))
    complete=(math.isfinite(target) and abs(end-target)<1e-8*max(1,abs(target))
              and len(diag)==nsteps+1 and len(step)==nsteps
              and [int(r['step']) for r in diag]==list(range(nsteps+1)))
    report.update(final_time=end if math.isfinite(end) else None,target_time=target if math.isfinite(target) else None,accepted_steps=len(step),complete=complete)
    if not complete:issues.append('Incomplete time interval or missing/duplicate step records')
    # Only the final endpoint may be clipped by accumulated floating-point time error.
    for i,r in enumerate(diag[1:],1):
        tolerance=max(1e-12,1e-10*abs(dt))
        if i==nsteps and complete: tolerance=max(tolerance,32*nsteps*math.ulp(max(1.0,abs(target))))
        if abs(float(r['dt'])-dt)>tolerance:
            issues.append('Accepted dt differs from configured fixed timestep'); break
    energy=[float(r['total_energy']) for r in diag]
    if not energy or energy[0]==0 or any(not math.isfinite(e) for e in energy):
        issues.append('Invalid total energy');energy_scale=1.0
    else:
        energy_scale=max(abs(energy[0]),1.0)
        report['max_relative_energy_error']=max(abs((e-energy[0])/energy[0]) for e in energy)
        if report['max_relative_energy_error']>args.energy_limit:issues.append('Relative energy error exceeds configured screening limit')
    report['all_outer_converged']=all(float(r['nonlinear_converged'])==1 for r in diag[1:])
    if not report['all_outer_converged']:issues.append('An accepted outer solve is unconverged')
    keys=['gauss_rms','gauss_max','gauss_relative','gauge_rms','gauge_max','gauge_relative','nonlinear_iterations','nonlinear_rms_residual','nonlinear_max_particle_residual','local_particle_iterations_max']
    # Initial-state solver residuals are intentionally NaN: no solve occurred.
    report['max_abs_main']={k:maximum(diag[1:] if k.startswith(('nonlinear_','local_')) else diag,k) for k in keys}
    for k in ['gauss_rms','gauge_rms']:
        value=report['max_abs_main'][k]
        if value is not None and value>args.constraint_limit:issues.append(k+' exceeds configured screening limit')
    keys=['A_chain_abs_rms','A_chain_rel_rms','A_chain_max_norm','A_chain_work_defect','delta_total_energy','deposit_gather_work_residual','particle_energy_residual','field_energy_residual']
    report['max_abs_step']={k:maximum(step,k) for k in keys}
    chain=report['max_abs_step']['A_chain_abs_rms']
    if chain is not None and chain>args.chain_limit:issues.append('Orbit chain RMS exceeds configured screening limit')
    report['normalized_residuals']={k:report['max_abs_step'][k]/energy_scale for k in ['deposit_gather_work_residual','particle_energy_residual','field_energy_residual'] if report['max_abs_step'][k] is not None}
    for k,v in report['normalized_residuals'].items():
        if v>args.work_limit:issues.append(k+' exceeds normalized screening limit')
if inner:
    allow_refinement=cfg.get('selective_particle_fallback') in ('true','1')
    failed=[r for r in inner if float(r['particles_converged'])!=1 or float(r['failed_particles'])!=0]
    report['all_inner_converged']=not failed
    report['failed_inner_evaluations']=len(failed)
    # Failed trial maps are expected under convergence-triggered refinement.
    # Every accepted interval must still end in a successful inner evaluation.
    last_by_interval={}
    for r in inner:last_by_interval[(float(r['time_n']),float(r['dt']))]=r
    final_ok=all(float(r['particles_converged'])==1 and float(r['failed_particles'])==0
                 for r in last_by_interval.values())
    report['final_inner_evaluations_converged']=final_ok
    if not final_ok:issues.append('A final inner-map evaluation recorded failure')
    if failed and not allow_refinement:issues.append('An inner-map evaluation recorded failure without refinement enabled')
    successful=[r for r in inner if float(r['particles_converged'])==1 and float(r['failed_particles'])==0]
    report['max_inner_scaled_residual']=maximum(successful,'max_scaled_particle_residual')
    if report['max_inner_scaled_residual'] is not None and report['max_inner_scaled_residual']>1+1e-9:issues.append('Successful inner scaled residual exceeds 1')
    for r in diag[1:]:
        time_n=float(r['time'])-float(r['dt'])
        if not any(abs(t-time_n)<1e-9 and abs(h-float(r['dt']))<1e-12 for t,h in last_by_interval):
            issues.append('Missing inner history for an accepted interval');break
else:issues.append('No inner-map records')
if attempts:
    report['rejected_attempts']=sum(r['outcome']!='accepted' for r in attempts)
    if report['rejected_attempts']:issues.append('Non-accepted solver attempts recorded')
else:issues.append('No solver-attempt records')
if moments:
    if len(moments)!=len(diag):issues.append('Missing velocity-moment rows')
    with file('_transverse_rms.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','time','weighted_vperp_rms','mean_vy','mean_vz'])
        values=[]
        for r in moments:
            v=math.hypot(float(r['rms_vy']),float(r['rms_vz']));values.append(v)
            writer.writerow([r['step'],r['time'],v,r['mean_vy'],r['mean_vz']])
    if any(not math.isfinite(v) for v in values):issues.append('Nonfinite transverse RMS')
    report['initial_weighted_vperp_rms']=values[0];report['final_weighted_vperp_rms']=values[-1]
    report['max_weighted_vperp_rms']=max(values)
else:issues.append('No transverse-moment records')
report['screening_pass']=not issues
file('_verification_summary.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False)+'\n')
lines=['# HPC instability diagnostic summary','', 'PASS: completion and configured conservation screens.' if not issues else 'FAIL: see issues below.', '',
       'These screens do not establish physical accuracy or spatial convergence.', '',
       f"Final time: {report.get('final_time','unavailable')}; target: {report.get('target_time','unavailable')}; accepted steps: {report.get('accepted_steps',0)}.",
       f"Maximum relative energy error: {report.get('max_relative_energy_error','unavailable')}",
       f"Final weighted transverse RMS: {report.get('final_weighted_vperp_rms','unavailable')}", '', '| Diagnostic | Maximum absolute value |','|---|---:|']
for group in ['max_abs_main','max_abs_step']:
    for key,value in report.get(group,{}).items():lines.append(f'| {key} | {value} |')
lines+=['','## Solver checks','',f"All outer converged: {report.get('all_outer_converged','unavailable')}",f"All inner converged: {report.get('all_inner_converged','unavailable')}",f"Maximum inner scaled residual: {report.get('max_inner_scaled_residual','unavailable')}",f"Rejected attempts: {report.get('rejected_attempts','unavailable')}", '', '## Issues','']
lines+=issues or ['None.']
file('_verification_summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
raise SystemExit(0 if not issues else 1)
