#!/bin/bash
#SBATCH --time=29:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=mig
#SBATCH --gres=gpu:1

# These are set by submit_online.sh via --export:
#   ENV, CONFIG, OFFLINE_DIR, NUM_STEPS, EXTRA_ARGS, JOB_NAME

module load intel-mkl/2024.2

CODE_DIR="/home/de7281/thesis/shortcuts_from_demos"
SCRATCH_DIR="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos"

# RESUME_DIR is always set by submit_online.sh to the canonical output path
# (whether or not it exists yet). Falls back to jobid-named dir if unset.
if [ -n "${RESUME_DIR:-}" ]; then
    RUN_DIR="$RESUME_DIR"
else
    RUN_DIR="$SCRATCH_DIR/outputs/${JOB_NAME}_${SLURM_JOB_ID}"
fi
mkdir -p "$RUN_DIR"

exec > >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Activating conda environment: slap_env..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate slap_env

if ! python -c "import torch; import numpy; import gymnasium; import robomimic" 2>/dev/null; then
    echo "ERROR: Required dependencies not found!"
    exit 1
fi
echo "Dependencies verified."

cd "$CODE_DIR" || exit 1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Online pipeline: $JOB_NAME"
echo "  config:      $CONFIG"
echo "  offline_dir: $OFFLINE_DIR"
echo "  num_steps:   $NUM_STEPS"
echo "  extra args:  $EXTRA_ARGS"
echo "  output:      $RUN_DIR"
echo "========================================"
echo ""

python experiments/online_pipeline.py \
    --config-name "$CONFIG" \
    offline_dir="$OFFLINE_DIR" \
    online_system.num_steps="$NUM_STEPS" \
    hydra.run.dir="$RUN_DIR" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"
echo "Output location: $RUN_DIR"

exit $EXIT_CODE
