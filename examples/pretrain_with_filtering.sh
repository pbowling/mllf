#!/bin/bash
#SBATCH --job-name=mllf_pretrain
#SBATCH --output=pretrain_status.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH --export=ALL
#SBATCH --time=15:00:00  # 15 hours for large pretraining datasets

# SLURM script for running policy pretraining with statistical filtering
#
# This script runs pretraining on all collected simulation data with
# automatic outlier filtering to exclude runs with abnormal coefficient values.
#
# Usage:
#   sbatch pretrain_with_filtering.sh
#
# Or with custom parameters:
#   sbatch --export=ALL,OUTLIER_THRESHOLD=2.5,NO_FILTER=false pretrain_with_filtering.sh

set -e  # Exit on error

echo "========================================="
echo "MLLF Policy Pretraining with Filtering"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Initialize conda for bash shell    
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the mllf environment
conda activate mllf

cd $SLURM_SUBMIT_DIR

echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Configuration
CONFIG_FILE="${CONFIG_FILE:-workflow_pretrain.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-models/pretraining}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-3.0}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-0}"
NO_FILTER="${NO_FILTER:-false}"
USE_BEST_ONLY="${USE_BEST_ONLY:-false}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Outlier threshold: ${OUTLIER_THRESHOLD}σ"
echo "  Reward threshold: >= $REWARD_THRESHOLD"
echo "  Filtering enabled: $([ "$NO_FILTER" = "false" ] && echo "yes" || echo "no")"
echo "  Use best only: $([ "$USE_BEST_ONLY" = "true" ] && echo "yes" || echo "no")"
echo ""

# Find pretraining directory
PRETRAIN_DIR=""
if [ -d "pretraining" ]; then
    PRETRAIN_DIR="pretraining"
elif [ -d "../pretraining" ]; then
    PRETRAIN_DIR="../pretraining"
else
    echo "Error: pretraining/ directory not found"
    exit 1
fi

echo "========================================="
echo "Scanning Pretraining Data"
echo "========================================="
echo ""

# Auto-detect all subdirectories in pretraining/
# Handle two structures:
# 1. Normal: pretraining/system/run_*/ (e.g., 14benz_solv/run1/)
# 2. Combos: pretraining/system/comb_*/run_*/ (e.g., 14benz_pair_combos/comb_0063.../run_046/)
total_systems=0
total_runs_available=0
pretrain_dirs=""

for dataset_dir in $PRETRAIN_DIR/*/; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        
        # Check if this has combo subdirectories (like 14benz_pair_combos)
        has_combos=false
        if [ -d "$dataset_dir" ]; then
            # Look for directories starting with "comb_"
            combo_dirs=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d -name "comb_*" 2>/dev/null | wc -l)
            if [ $combo_dirs -gt 0 ]; then
                has_combos=true
            fi
        fi
        
        if [ "$has_combos" = true ]; then
            # Handle combo structure: add each comb_* directory separately
            echo "  - $dataset_name (combo structure):"
            for comb_dir in "$dataset_dir"comb_*/; do
                if [ -d "$comb_dir" ]; then
                    comb_name=$(basename "$comb_dir")
                    run_count=$(find "$comb_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
                    if [ $run_count -gt 0 ]; then
                        total_systems=$((total_systems + 1))
                        total_runs_available=$((total_runs_available + run_count))
                        echo "      $comb_name: $run_count runs"
                        pretrain_dirs="$pretrain_dirs --pretraining-dir $comb_dir"
                    fi
                fi
            done
        else
            # Normal structure: add the system directory directly
            count=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            if [ $count -gt 0 ]; then
                total_systems=$((total_systems + 1))
                total_runs_available=$((total_runs_available + count))
                echo "  - $dataset_name: $count runs"
                pretrain_dirs="$pretrain_dirs --pretraining-dir $dataset_dir"
            fi
        fi
    fi
done

echo ""
echo "Total: $total_systems systems, $total_runs_available runs available"
echo ""

if [ -z "$pretrain_dirs" ]; then
    echo "Error: No pretraining directories found"
    exit 1
fi

echo "========================================="
echo "Starting Pretraining"
echo "========================================="
echo ""

# Build command with optional flags
CMD="python -u -m mllf.cb.pretrain_policy \
    $pretrain_dirs \
    --output-dir $OUTPUT_DIR \
    --config $CONFIG_FILE \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE"

# Add optional filtering parameters
if [ "$NO_FILTER" = "true" ]; then
    CMD="$CMD --no-filter-outliers"
    echo "Statistical filtering: DISABLED"
else
    CMD="$CMD --outlier-std-threshold $OUTLIER_THRESHOLD"
    echo "Statistical filtering: ENABLED (±${OUTLIER_THRESHOLD}σ threshold)"
fi

# Add reward filtering
if [ -n "$REWARD_THRESHOLD" ]; then
    CMD="$CMD --min-reward-threshold $REWARD_THRESHOLD"
    echo "Reward filtering: ENABLED (>= $REWARD_THRESHOLD)"
else
    echo "Reward filtering: DISABLED"
fi

# Add best-only flag if requested
if [ "$USE_BEST_ONLY" = "true" ]; then
    CMD="$CMD --use-best-only"
    echo "Training mode: Best run per system only"
else
    echo "Training mode: All valid runs"
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "Starting pretraining at $(date)..."
echo ""

# Run pretraining with unbuffered output for real-time logging
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Pretraining Complete"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pretraining successful!"
    echo ""
    echo "Output saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check pretrain_status.out for filtering statistics"
    echo "2. Update your workflow config to use pretrained model:"
    echo "   pretrain:"
    echo "     model_path: $OUTPUT_DIR/best_policy.pt"
    echo "3. Run training workflow"
else
    echo "✗ Pretraining failed with exit code $EXIT_CODE"
    echo "Check pretrain_status.out for error details"
fi

exit $EXIT_CODE
