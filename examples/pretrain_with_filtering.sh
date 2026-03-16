#!/bin/bash
#SBATCH --job-name=mllf_pretrain
#SBATCH --output=pretrain_status.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu
#SBATCH --export=ALL
#SBATCH --time=15:00:00  # 15 hours for large pretraining datasets

# SLURM script for running CB policy pretraining on collected MSLD simulation data.
#
# Uses the full DeepSet → sum-pool → RGCN pipeline by default:
#   - Loads the pretrained DeepSet encoder from pretraining/deepset_pretraining_output/
#   - Replaces standard atom-type node features with 64-dim AEV-based embeddings
#   - Trains the RGCN+EdgePolicy on behavior-cloning MSE loss
#
# Automatic outlier filtering excludes runs with abnormal coefficient values.
#
# Usage:
#   sbatch pretrain_with_filtering.sh
#
# With custom parameters:
#   sbatch --export=ALL,OUTLIER_THRESHOLD=2.5,EPOCHS=100 pretrain_with_filtering.sh
#
# To use stratified negative sampling (keeps all positive + 25% of each neg bucket by default):
#   sbatch --export=ALL,STRATIFIED_FRACTION=0.25 pretrain_with_filtering.sh
#
# To disable stratified sampling and fall back to reward threshold filtering:
#   sbatch --export=ALL,STRATIFIED_FRACTION=0,REWARD_THRESHOLD=0 pretrain_with_filtering.sh
#
# To disable DeepSet (standard atom-type features):
#   sbatch --export=ALL,DEEPSET_ENCODER=none pretrain_with_filtering.sh
#
# To use a different encoder checkpoint:
#   sbatch --export=ALL,DEEPSET_ENCODER=/path/to/best_encoder.pt pretrain_with_filtering.sh
#
# Default excluded datasets (EXCLUDE_DATASETS):
#   14benz_pair_combos  - combo structure handled separately; exclude from normal scan
#   luis_cdk2_*         - all 4 groups have zero positive-reward runs (max <= -7), genuinely poor sampling
#   luis_ptp1b_{protein,solvent}_group1 - max -14, no positive runs, large datasets (220 runs each)
#   p38_protein_groupA/B/C - max -1.66 to -21.6, zero positive runs across all protein groups
#   mup1_solvent_group2 - max -2.86, zero positive runs
#   luis_p38_protein_group2 - max -8.01, zero positive runs
#   (Systems that only hit the completeness gate, max=-0.01, are left in as their
#    coefficients were partially effective and the data may still be informative.)

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

# Locate workflow_pretrain.yaml regardless of submission directory
# (works whether sbatch is run from mllf/ or examples/)
if [ -z "$CONFIG_FILE" ]; then
    if [ -f "workflow_pretrain.yaml" ]; then
        CONFIG_FILE="workflow_pretrain.yaml"
    elif [ -f "examples/workflow_pretrain.yaml" ]; then
        CONFIG_FILE="examples/workflow_pretrain.yaml"
    else
        echo "Error: workflow_pretrain.yaml not found. Set CONFIG_FILE explicitly."
        exit 1
    fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-models/pretraining}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-3.0}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-0}"
NO_FILTER="${NO_FILTER:-true}"
USE_BEST_ONLY="${USE_BEST_ONLY:-false}"
STRATIFIED_FRACTION="${STRATIFIED_FRACTION:-0.25}"
DEEPSET_ENCODER="${DEEPSET_ENCODER:-pretraining/deepset_pretraining_output/trained_models/best_encoder.pt}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-14benz_pair_combos luis_cdk2_protein_group1 luis_cdk2_protein_group2 luis_cdk2_solvent_group1 luis_cdk2_solvent_group2 luis_ptp1b_protein_group1 luis_ptp1b_solvent_group1 p38_protein_groupA p38_protein_groupB p38_protein_groupC mup1_solvent_group2 luis_p38_protein_group2}"
PATIENCE="${PATIENCE:-10}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Outlier threshold: ${OUTLIER_THRESHOLD}σ"
echo "  Reward threshold: >= $REWARD_THRESHOLD"
echo "  Filtering enabled: $([ "$NO_FILTER" = "false" ] && echo "yes" || echo "no")"
echo "  Use best only: $([ "$USE_BEST_ONLY" = "true" ] && echo "yes" || echo "no")"
if [ -n "$STRATIFIED_FRACTION" ] && [ "$STRATIFIED_FRACTION" != "0" ]; then
    echo "  Stratified negative sampling: ${STRATIFIED_FRACTION} (fraction per bucket)"
else
    echo "  Stratified negative sampling: DISABLED"
fi
if [ -n "$DEEPSET_ENCODER" ] && [ "$DEEPSET_ENCODER" != "none" ] && [ -f "$DEEPSET_ENCODER" ]; then
    echo "  DeepSet encoder: $DEEPSET_ENCODER"
else
    echo "  DeepSet encoder: DISABLED (standard atom-type node features)"
fi
if [ -n "$EXCLUDE_DATASETS" ]; then
    echo "  Excluded datasets: $EXCLUDE_DATASETS"
fi
echo "  Early stopping patience: $PATIENCE epochs"
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

        # Skip the DeepSet pretraining output directory — it holds encoder weights,
        # not MSLD simulation runs, so it is not valid input for CB pretraining.
        if [ "$dataset_name" = "deepset_pretraining_output" ]; then
            continue
        fi

        # Skip any datasets listed in EXCLUDE_DATASETS (space-separated names)
        skip_dataset=false
        for excl in $EXCLUDE_DATASETS; do
            if [ "$dataset_name" = "$excl" ]; then
                skip_dataset=true
                break
            fi
        done
        if [ "$skip_dataset" = "true" ]; then
            echo "  - $dataset_name: EXCLUDED"
            continue
        fi
        
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
if [ -n "$STRATIFIED_FRACTION" ] && [ "$STRATIFIED_FRACTION" != "0" ]; then
    CMD="$CMD --stratified-negative-fraction $STRATIFIED_FRACTION"
    echo "Stratified negative sampling: ENABLED (${STRATIFIED_FRACTION} per bucket)"
elif [ -n "$REWARD_THRESHOLD" ]; then
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

# Add DeepSet encoder if available
if [ -n "$DEEPSET_ENCODER" ] && [ "$DEEPSET_ENCODER" != "none" ] && [ -f "$DEEPSET_ENCODER" ]; then
    CMD="$CMD --deepset-encoder $DEEPSET_ENCODER"
    echo "DeepSet encoder: ENABLED ($DEEPSET_ENCODER)"
else
    echo "DeepSet encoder: DISABLED"
    if [ -n "$DEEPSET_ENCODER" ] && [ "$DEEPSET_ENCODER" != "none" ]; then
        echo "  Warning: DEEPSET_ENCODER path not found: $DEEPSET_ENCODER"
    fi
fi

# Add early stopping patience
CMD="$CMD --patience $PATIENCE"
echo "Early stopping patience: $PATIENCE epochs"

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
    echo "1. Check pretrain_status.out for filtering statistics and per-run losses"
    echo "2. Update your workflow config to use the pretrained policy:"
    echo "   pretrain:"
    echo "     model_path: $OUTPUT_DIR/best_policy.pt"
    echo "3. If DeepSet was enabled, also set the encoder path in your workflow config:"
    echo "   deepset:"
    echo "     encoder_path: $DEEPSET_ENCODER"
    echo "4. Run the training workflow"
else
    echo "✗ Pretraining failed with exit code $EXIT_CODE"
    echo "Check pretrain_status.out for error details"
fi

exit $EXIT_CODE
