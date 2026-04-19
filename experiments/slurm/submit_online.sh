#!/bin/bash
# Submit an online training job for a specific environment.
#
# Usage:
#   ./experiments/slurm/submit_online.sh <env> <num_steps> [--residual] [extra hydra overrides...]
#   ./experiments/slurm/submit_online.sh --resume <existing_dir> [extra hydra overrides...]
#
# Examples:
#   ./experiments/slurm/submit_online.sh lift 0
#   ./experiments/slurm/submit_online.sh can 5000
#   ./experiments/slurm/submit_online.sh can 5000 --residual
#   ./experiments/slurm/submit_online.sh square 10000 online_system.sampling_method=uniform
#
#   # Re-evaluate an existing run (loads online_system.pt, skips training):
#   ./experiments/slurm/submit_online.sh --resume /scratch/.../outputs/can_online_5000steps
#
# Output directory naming:
#   {env}_online_{num_steps}steps          (default)
#   {env}_online_{num_steps}steps_residual (with --residual)
# Re-running with the same args overwrites the same directory.

set -euo pipefail

SCRATCH_DIR="/scratch/gpfs/TSILVER/de7281/shortcuts_from_demos"

# ── Resume mode ──────────────────────────────────────────────────────────
if [ "${1:-}" = "--resume" ]; then
    if [ $# -lt 2 ]; then
        echo "Usage: $0 --resume <existing_output_dir> [extra overrides...]"
        exit 1
    fi
    RESUME_DIR="$2"
    shift 2
    EXTRA_ARGS="resume=true $*"

    if [ ! -d "$RESUME_DIR" ]; then
        echo "Error: resume dir not found: $RESUME_DIR"
        exit 1
    fi

    # Read config name from the saved online_config.yaml
    CONFIG=$(python3 -c "
import yaml, sys
with open('$RESUME_DIR/online_config.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('_config_name_', 'unknown'))
" 2>/dev/null || echo "unknown")

    # Infer env name from the directory name
    dir_name=$(basename "$RESUME_DIR")
    ENV=$(echo "$dir_name" | sed 's/_online_.*//')

    # Look up config from env name
    case "$ENV" in
        lift)       CONFIG="lift_ph_lowdim" ;;
        can)        CONFIG="can_ph_lowdim" ;;
        square)     CONFIG="square_ph_lowdim" ;;
        tool_hang)  CONFIG="tool_hang_ph_lowdim" ;;
        transport)  CONFIG="transport_ph_lowdim" ;;
        *)          echo "Could not infer env from dir name: $dir_name"; exit 1 ;;
    esac

    # Read offline_dir from the saved config
    OFFLINE_DIR=$(python3 -c "
import yaml
with open('$RESUME_DIR/online_config.yaml') as f:
    print(yaml.safe_load(f)['offline_dir'])
")

    JOB_NAME="${ENV}_eval_resume"

    sbatch \
        --job-name="$JOB_NAME" \
        --export=ALL,ENV="$ENV",CONFIG="$CONFIG",OFFLINE_DIR="$OFFLINE_DIR",NUM_STEPS=0,EXTRA_ARGS="$EXTRA_ARGS",JOB_NAME="$JOB_NAME",RESUME_DIR="$RESUME_DIR" \
        experiments/slurm/online_worker.slurm

    echo "Submitted resume: $JOB_NAME  (dir=$RESUME_DIR)"
    exit 0
fi

# ── Normal mode ──────────────────────────────────────────────────────────
if [ $# -lt 2 ]; then
    echo "Usage: $0 <env> <num_steps> [extra overrides...]"
    echo "       $0 --resume <existing_dir> [extra overrides...]"
    echo "  env: lift | can | square | tool_hang | transport"
    echo "  num_steps: number of online training steps (0 = eval only)"
    exit 1
fi

ENV="$1"
NUM_STEPS="$2"
shift 2

# Parse --residual flag from remaining args.
RESIDUAL=false
REMAINING_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--residual" ]; then
        RESIDUAL=true
    else
        REMAINING_ARGS+=("$arg")
    fi
done

if [ "$RESIDUAL" = true ]; then
    EXTRA_ARGS="online_system.residualize=true ${REMAINING_ARGS[*]:-}"
    SUFFIX="_true_residual"
else
    EXTRA_ARGS="${REMAINING_ARGS[*]:-}"
    SUFFIX=""
fi

# Map env name to config and offline dir
case "$ENV" in
    lift)       CONFIG="lift_ph_lowdim";      OFFLINE_DIR="$SCRATCH_DIR/outputs/lift_embeddings" ;;
    can)        CONFIG="can_ph_lowdim";        OFFLINE_DIR="$SCRATCH_DIR/outputs/can_embeddings" ;;
    square)     CONFIG="square_ph_lowdim";     OFFLINE_DIR="$SCRATCH_DIR/outputs/square_embeddings" ;;
    tool_hang)  CONFIG="tool_hang_ph_lowdim";  OFFLINE_DIR="$SCRATCH_DIR/outputs/tool_hang_embeddings" ;;
    transport)  CONFIG="transport_ph_lowdim";  OFFLINE_DIR="$SCRATCH_DIR/outputs/transport_embeddings" ;;
    *)
        echo "Unknown env: $ENV"
        echo "Choose from: lift, can, square, tool_hang, transport"
        exit 1
        ;;
esac

# Verify offline dir exists
if [ ! -d "$OFFLINE_DIR" ]; then
    echo "Error: offline dir not found: $OFFLINE_DIR"
    exit 1
fi

JOB_NAME="${ENV}_online_${NUM_STEPS}steps${SUFFIX}"
RUN_DIR="$SCRATCH_DIR/outputs/${JOB_NAME}"

sbatch \
    --job-name="$JOB_NAME" \
    --export=ALL,ENV="$ENV",CONFIG="$CONFIG",OFFLINE_DIR="$OFFLINE_DIR",NUM_STEPS="$NUM_STEPS",EXTRA_ARGS="$EXTRA_ARGS",JOB_NAME="$JOB_NAME",RESUME_DIR="$RUN_DIR" \
    experiments/slurm/online_worker.slurm

echo "Submitted: $JOB_NAME  (config=$CONFIG, offline=$OFFLINE_DIR, steps=$NUM_STEPS, residual=$RESIDUAL)"
echo "  output: $RUN_DIR"
