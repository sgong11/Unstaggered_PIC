#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit a named manuscript experiment suite to Slurm
# Inputs:
#   $1 : main, projection, spline, or all; remaining args : sbatch options
# Output:
#   independent jobs; submission record and persistent suite lock
# Dependencies:
#   bash, sbatch, sha256sum, run_case.sb, SHA256SUMS
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
suite="${1:-main}"
if (( $# )); then shift; fi
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
command -v sbatch >/dev/null
sha256sum --check SHA256SUMS
# One lock per case also prevents overlapping main/all submissions.
mkdir -p submissions
for name in "${cases[@]}"; do
    if [[ -e "submissions/${name}.lock" ]]; then
        echo "Already attempted: $name. Inspect submissions/ before retrying." >&2; exit 2
    fi
done
record="submissions/${suite}_$(date +%Y%m%d_%H%M%S)_$$.tsv"
printf 'case\tjob_id\n' > "$record"
for name in "${cases[@]}"; do
    mkdir "submissions/${name}.lock"
    job=$(sbatch --parsable "$@" --job-name="$name" --output="${name}-%j.out" \
        run_case.sb "inputs/manuscript/${name}.txt" "$name")
    printf '%s\t%s\n' "$name" "$job" | tee -a "$record"
done
