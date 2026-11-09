#!/bin/bash
#SBATCH --job-name=rt-dataprep
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# Slurm job: organize, convert and manifest RAW into WORK/{organized,converted}. Extra arguments after WORK are
# preprocess flags; with them the job also writes WORK/pre and its manifest. The log carries counts and reasons only.
#
#   sbatch -A <account> examples/slurm_pipeline.sbatch RAW WORK
#   sbatch -A <account> examples/slurm_pipeline.sbatch RAW WORK --initial-resample --rescale --resample --reshape --new-shape 320 320 256
#
# WORK must be outside RAW. RT_DATAPREP overrides the command (default: rt-dataprep on PATH, e.g. an activated venv).
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: sbatch -A <account> $0 RAW WORK [preprocess flags...]" >&2; exit 2; }
RAW=$1 WORK=$2
shift 2
RT=${RT_DATAPREP:-rt-dataprep}
stage() { echo "=== $1 $(date -u +%FT%TZ)"; }
stage organize; "$RT" organize "$RAW" "$WORK/organized" --workers auto
stage convert;  "$RT" convert "$WORK/organized" "$WORK/converted" --workers auto
stage manifest; "$RT" manifest "$WORK/converted"
if [ $# -gt 0 ]; then
    stage preprocess; "$RT" preprocess "$WORK/converted" "$WORK/pre" --workers auto "$@"
    stage manifest;   "$RT" manifest "$WORK/pre"
fi
stage done
sacct -j "${SLURM_JOB_ID:-}" -o JobID,Elapsed,MaxRSS,ReqMem,AllocCPUS,State || true
