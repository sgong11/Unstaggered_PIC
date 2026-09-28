"""Stream production particle snapshots into weighted/unweighted RMS diagnostics."""
from pathlib import Path
import argparse
import csv
import math
import sys

# ---------------------------------------------------------------------------
# Stream all-particle snapshots into transverse-velocity statistics
#
# Inputs:
#   CLI prefix : str, simulation output prefix including optional directory
#   CSV files  : <prefix>_particles_t*.csv with time, vy, vz, and q
# Output:
#   stdout : CSV sorted by actual saved time; weights are abs(q)
# Dependencies:
#   - pathlib, argparse, csv, math, sys (Python standard library)
# ---------------------------------------------------------------------------
parser=argparse.ArgumentParser()
parser.add_argument('prefix', help='Simulation output prefix, e.g. tsi3d_projected_T60')
args=parser.parse_args()
prefix=Path(args.prefix)
files=list(prefix.parent.glob(prefix.name+'_particles_t*.csv'))
if not files: parser.error('No snapshots match the prefix')
result=[]
for path in files:
    count=0; vv=wv=w=my=mz=0.0
    with path.open() as f:
        for row in csv.DictReader(f):
            t=float(row['time']);vy=float(row['vy']);vz=float(row['vz']);weight=abs(float(row['q']))
            if not all(math.isfinite(v) for v in (t,vy,vz,weight)):
                raise ValueError(f'Nonfinite particle data: {path}')
            r=vy*vy+vz*vz
            count+=1;vv+=r;wv+=weight*r;w+=weight;my+=weight*vy;mz+=weight*vz
    if not count or not w: raise ValueError(f'Empty or zero-weight snapshot: {path}')
    result.append([t,count,math.sqrt(vv/count),math.sqrt(wv/w),my/w,mz/w,str(path)])
writer=csv.writer(sys.stdout)
writer.writerow(['time','particles','vperp_rms','vperp_weighted_rms','weighted_mean_vy','weighted_mean_vz','snapshot'])
writer.writerows(sorted(result))
