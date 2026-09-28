#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Compile once and run a manuscript suite locally in separate output folders
# Inputs:
#   $1 : main, projection, spline, or all (default all)
#   CXX : compiler command (default c++); OPENMP : 0 or 1 (default 0)
#   OMP_NUM_THREADS : thread count when OpenMP is enabled (default 16)
# Output:
#   runs/local_<timestamp>_<pid>/<case>/ : inputs, source, logs, CSV, summaries
#   Exit status 1 if any solver or diagnostic check fails; zero otherwise
# Dependencies:
#   Bash, C++17 compiler, Python >=3.9, the two summary scripts
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CODE_DIR="$PWD"
suite="${1:-all}"
main=(tsi_main_dt01_T60 tsi_main_dt03_T60 weibel_main_dt004_T100 weibel_main_dt02_T100)
projection=(tsi_projection_off_dt01_T60 tsi_projection_on_dt01_T60)
spline=(tsi_spline_r1_dt01_T60 tsi_spline_r2_dt01_T60)
case "$suite" in
    main) cases=("${main[@]}");;
    projection) cases=("${projection[@]}");;
    spline) cases=("${spline[@]}");;
    all) cases=("${main[@]}" "${projection[@]}" "${spline[@]}");;
    *) echo 'Suite must be main, projection, spline, or all.' >&2; exit 2;;
esac
[[ "${OPENMP:-0}" =~ ^[01]$ ]] || { echo 'OPENMP must be 0 or 1.' >&2; exit 2; }
command -v python3 >/dev/null
compiler="${CXX:-c++}"
command -v "$compiler" >/dev/null
flags=(-std=c++17 -O2)
if [[ "${OPENMP:-0}" == 1 ]]; then flags+=(-fopenmp); fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OMP_DYNAMIC=false
batch="$CODE_DIR/runs/local_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$CODE_DIR/runs"
mkdir "$batch"
cp ec_pic_nyquist_projected.cpp summarize_hpc_run.py summarize_snapshots.py "$batch/"
"$compiler" --version > "$batch/compiler.txt"
"$compiler" "${flags[@]}" "$batch/ec_pic_nyquist_projected.cpp" -o "$batch/pic"
printf 'case\tsolver_exit\tsummary_exit\tsnapshot_exit\n' > "$batch/status.tsv"
failed=0
for name in "${cases[@]}"; do
    folder="$batch/$name"
    mkdir "$folder"
    cp "inputs/manuscript/$name.txt" "$folder/input.txt"
    echo "Running $name; outputs: $folder"
    (
        cd "$folder"
        set +e
        "$batch/pic" input.txt > simulation.log 2>&1
        solver_status=$?
        python3 "$batch/summarize_hpc_run.py" "$name" --solver-exit "$solver_status" > verification.log 2>&1
        summary_status=$?
        python3 "$batch/summarize_snapshots.py" "$name" > snapshot_transverse_rms.csv 2> snapshot_summary.log
        snapshot_status=$?
        printf '%s\t%s\t%s\t%s\n' "$name" "$solver_status" "$summary_status" "$snapshot_status" >> "$batch/status.tsv"
        (( solver_status == 0 && summary_status == 0 && snapshot_status == 0 ))
    ) || failed=1
    # Every case is independent; retain failures and continue the remaining cases.
done
echo "Finished. Check $batch/status.tsv and each case's verification.log."
exit "$failed"
