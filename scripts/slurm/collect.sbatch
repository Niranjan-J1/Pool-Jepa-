#!/bin/bash
# Sharded data collection. One shard per job, so each job fits the 2 h limit.
# Usage:  sbatch scripts/slurm/collect.sbatch <mode> <shard> [episodes] [extra collect.py args...]
#   e.g.  sbatch scripts/slurm/collect.sbatch strike_outcomes 0 500
#         sbatch scripts/slurm/collect.sbatch mb_shots 3 5000 --split train
# Shard k uses --seed k and --prefix <mode>_s<k>, so shards never overwrite each other.
#SBATCH --job-name=pj-collect
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# EDIT ME: uncomment/fill if your WATcloud account needs them
##SBATCH --account=EDIT_ME
##SBATCH --partition=EDIT_ME
set -euo pipefail

MODE=${1:?mode}; SHARD=${2:?shard}; EPISODES=${3:-}; shift $(( $# < 3 ? $# : 3 ))
: "${POOLJEPA_DATA:?set POOLJEPA_DATA (e.g. /mnt/wato-drive2/<you>/pool-jepa)}"
source "${POOLJEPA_VENV:-$HOME/pool-jepa/.venv}/bin/activate"   # EDIT ME if your venv lives elsewhere
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "commit $(git rev-parse --short HEAD)  mode=$MODE shard=$SHARD data=$POOLJEPA_DATA"
if [[ "$MODE" == mb_* ]]; then
    # Multiball: collect.py names files <mode>_<split>_s<shard>; pass --split in the extra args.
    python data/collect.py --mode "$MODE" --shard "$SHARD" ${EPISODES:+--episodes "$EPISODES"} "$@"
else
    python data/collect.py --mode "$MODE" --seed "$SHARD" --prefix "${MODE}_s${SHARD}" \
        ${EPISODES:+--episodes "$EPISODES"} "$@"
fi
