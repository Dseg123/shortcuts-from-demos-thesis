#!/bin/bash
#SBATCH --time=71:59:00  # Maximum runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=mig
#SBATCH --gres=gpu:1

# Load modules
module load intel-mkl/2024.2

# Set up paths
CODE_DIR="/home/de7281/thesis/shortcuts_from_demos"
CONDA_ENV="slap_env"
SCRATCH_DIR="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos"

# Create output directories
mkdir -p "$SCRATCH_DIR/slurm_logs"
RUN_DIR="$SCRATCH_DIR/outputs/online_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Tee stdout/stderr into the run directory as well
exec > >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Verify dependencies
if ! python -c "import torch; import numpy; import gymnasium; import robomimic" 2>/dev/null; then
    echo "ERROR: Required dependencies not found!"
    exit 1
fi
echo "Dependencies verified."

cd "$CODE_DIR" || exit 1
export HYDRA_FULL_ERROR=1
# MIG partitions set CUDA_VISIBLE_DEVICES to a UUID string, which breaks
# robosuite's EGL context (egl_context.py parses it as an integer).
# Override to "0" so EGL finds a valid device; CUDA still sees the MIG instance.
export CUDA_VISIBLE_DEVICES=0

echo "Starting online pipeline — output dir: $RUN_DIR"
echo ""

# Requires offline_dir pointing to a completed pipeline.py output directory.
# Example:
#   sbatch run_online.slurm offline_dir=/scratch/gpfs/.../outputs/pipeline_job6100692
#   sbatch run_online.slurm offline_dir=... online_system.residualize=true
# python experiments/online_pipeline.py \
#     --config-name simple \
#     offline_dir="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs/pipeline_job6146963" \
#     online_system.num_steps=0 \
#     hydra.run.dir="$RUN_DIR" \
#     "$@"

# python experiments/online_pipeline.py \
#     --config-name lift_ph_lowdim \
#     offline_dir="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs/lift_embeddings" \
#     online_system.num_steps=0 \
#     hydra.run.dir="$RUN_DIR" \
#     "$@"

# python experiments/online_pipeline.py \
#     --config-name can_ph_lowdim \
#     offline_dir="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs/can_embeddings" \
#     online_system.num_steps=0 \
#     hydra.run.dir="$RUN_DIR" \
#     "$@"

# python experiments/online_pipeline.py \
#     --config-name square_ph_lowdim \
#     offline_dir="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs/square_embeddings" \
#     online_system.num_steps=0 \
#     hydra.run.dir="$RUN_DIR" \
#     "$@"

# python experiments/online_pipeline.py \
#     --config-name tool_hang_ph_lowdim \
#     offline_dir="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs/tool_hang_embeddings" \
#     online_system.num_steps=0 \
#     hydra.run.dir="$RUN_DIR" \
#     "$@"

python experiments/online_pipeline.py \
    --config-name transport_ph_lowdim \
    offline_dir="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos/outputs/transport_embeddings" \
    online_system.num_steps=0 \
    resume="transport_online_4000steps_6550951" \
    hydra.run.dir="$RUN_DIR" \
    "$@"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Experiment completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================"
echo "Output location: $RUN_DIR"

exit $EXIT_CODE
