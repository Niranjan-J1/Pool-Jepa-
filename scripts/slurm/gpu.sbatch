#!/bin/bash
# Generic GPU job for training or eval: runs whatever python command you pass.
# Usage:  sbatch scripts/slurm/gpu.sbatch experiments/train_encoder.py [args...]
#         sbatch scripts/slurm/gpu.sbatch planning/full_shot_test.py --trials 100 --suffix _bigcue
# The 2 h limit means long training must resume from its last checkpoint; scripts that
# support it get a --resume flag in the bucket that needs it. Resubmit the same command.
#SBATCH --job-name=pj-gpu
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
# EDIT ME: uncomment/fill if your WATcloud account needs them
##SBATCH --account=EDIT_ME
##SBATCH --partition=EDIT_ME
set -euo pipefail

: "${POOLJEPA_DATA:?set POOLJEPA_DATA (e.g. /mnt/wato-drive2/<you>/pool-jepa)}"
source "${POOLJEPA_VENV:-$HOME/pool-jepa/.venv}/bin/activate"   # EDIT ME if your venv lives elsewhere
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "commit $(git rev-parse --short HEAD)  cmd: python $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$@"
