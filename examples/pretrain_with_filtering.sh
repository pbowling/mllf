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
# FEATURES:
#
# • Behavior Cloning: Trains UnimolPolicy to predict bias coefficients from
#   successful simulation runs using supervised learning (MSE loss)
#
# • Pairwise Population Weighting (NEW): Automatically weights edges by:
#   - Population balance: min(pop_i, pop_j) / max(pop_i, pop_j)
#   - DDG reliability: skips edges with NaN/inf DDG or zero populations
#   - Antisymmetric DDG: uses reverse pair with sign flip if forward missing
#   Result: high-confidence edges drive training, low-confidence edges skipped
#
# • Per-Pair AWR (Advantage-Weighted Regression) (NEW): Switches from whole-run 
#   scalar weighting to per-edge, per-dimension weighting. Well-resolved pairs
#   (high DDG quality, good population balance, high FRACTION PHYSICAL) within a
#   successful run drive more gradient than poorly-resolved pairs in that run.
#   Enabled by default; disable with --no-per-pair-awr or USE_PER_PAIR_AWR=false.
#
# • Automatic outlier filtering: Excludes runs with abnormal coefficient values.
#
# Usage:
#   sbatch pretrain_with_filtering.sh
#
# With custom parameters:
#   sbatch --export=ALL,OUTLIER_THRESHOLD=2.5,EPOCHS=100 pretrain_with_filtering.sh
#
# To use stratified negative sampling (quadratic ramp: 0% for worst bucket to 55% for best):
#   sbatch --export=ALL,STRATIFIED_FRACTION=0.55 pretrain_with_filtering.sh
#
# To enable reward-weighted loss (high-reward runs get proportionally more gradient signal):
#   sbatch --export=ALL,REWARD_WEIGHTED=true pretrain_with_filtering.sh
#
# To disable per-pair AWR (fall back to whole-run scalar weighting):
#   sbatch --export=ALL,USE_PER_PAIR_AWR=false pretrain_with_filtering.sh
#
# To tune AWR temperature (higher → more uniform, lower → sharper emphasis; default 1.0):
#   sbatch --export=ALL,AWR_TEMPERATURE=0.5 pretrain_with_filtering.sh
#
# To also produce a NeuralLinear + Thompson Sampling checkpoint (best_policy_bayesian.pt /
# final_policy_bayesian.pt) alongside the normal deterministic one, for use with
# training/workflow_instructions_neurallinear_ts.yaml (bandit.algorithm: neurallinear_ts):
#   sbatch --export=ALL,BAYESIAN_HEADS=true pretrain_with_filtering.sh
#   (optionally tune BAYESIAN_PRIOR_PRECISION, default 1.0 — must match the
#    bandit.prior_precision used in the online-training config)
#
# To disable stratified sampling and fall back to reward threshold filtering:
#   sbatch --export=ALL,STRATIFIED_FRACTION=0,REWARD_THRESHOLD=0 pretrain_with_filtering.sh
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
echo "FEATURES ENABLED:"
echo "  ✓ Behavior Cloning: MSE loss on simulation targets"
echo "  ✓ Pairwise AWR Weighting: Automatic edge confidence scoring"
echo "  ✓ Reverse Pair Handling: DDG antisymmetry (ΔG_ij = -ΔG_ji)"
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
STRATIFIED_FRACTION="${STRATIFIED_FRACTION:-0.55}"
REWARD_WEIGHTED="${REWARD_WEIGHTED:-false}"
#INCLUDE_REVERSE_PAIRS="${INCLUDE_REVERSE_PAIRS:-false}"
#EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-14benz_pair_combos luis_cdk2_protein_group1 luis_cdk2_protein_group2 luis_cdk2_solvent_group1 luis_cdk2_solvent_group2 luis_ptp1b_protein_group1 luis_ptp1b_solvent_group1 p38_protein_groupA p38_protein_groupB p38_protein_groupC mup1_solvent_group2 luis_p38_protein_group2}"
PATIENCE="${PATIENCE:-5}"
USE_PER_PAIR_AWR="${USE_PER_PAIR_AWR:-true}"
AWR_TEMPERATURE="${AWR_TEMPERATURE:-1.0}"
BAYESIAN_HEADS="${BAYESIAN_HEADS:-false}"
BAYESIAN_PRIOR_PRECISION="${BAYESIAN_PRIOR_PRECISION:-1.0}"

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
    echo "  Stratified negative sampling: ${STRATIFIED_FRACTION} (max fraction, quadratic ramp)"
else
    echo "  Stratified negative sampling: DISABLED"
fi
echo "  Reward-weighted loss: $([ "$REWARD_WEIGHTED" = "true" ] && echo "ENABLED" || echo "disabled")"
#echo "  Reverse pair training: $([ "$INCLUDE_REVERSE_PAIRS" = "true" ] && echo "ENABLED" || echo "disabled")"
if [ -n "$EXCLUDE_DATASETS" ]; then
    echo "  Excluded datasets: $EXCLUDE_DATASETS"
fi
echo "  Early stopping patience: $PATIENCE epochs"
echo "  Per-pair AWR: $([ "$USE_PER_PAIR_AWR" = "true" ] && echo "ENABLED" || echo "disabled")"
echo "  AWR temperature: $AWR_TEMPERATURE"
echo "  NeuralLinear+TS Bayesian-head checkpoint: $([ "$BAYESIAN_HEADS" = "true" ] && echo "ENABLED (prior_precision=$BAYESIAN_PRIOR_PRECISION)" || echo "disabled")"

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

# Auto-detect all subdirectories in pretraining/ recursively.
# Handle three scenarios:
# 1. Top-level combos: pretraining/system_combos/ with comb_*/ subdirectories
# 2. Nested combos: pretraining/system_name/comb_*/ at any depth
# 3. Regular runs: pretraining/system_name/run_*/ (no combos)
#
# Strategy: For each system-level directory, recursively find all comb_* subdirectories
# (at any depth) and add each. If a system has no combos, add the system directory itself.
total_systems=0
total_runs_available=0
pretrain_dirs=""

for dataset_dir in "$PRETRAIN_DIR"/*/; do
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi
    
    dataset_name=$(basename "$dataset_dir")

    # Skip special output directories
    if [ "$dataset_name" = "deepset_pretraining_output" ] \
    || [ "$dataset_name" = "atombondgnn_training_output" ] \
    || [ "$dataset_name" = "1_analysis_scripts" ]; then
        continue
    fi

    # Skip excluded datasets
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
    
    # Find all comb_* directories recursively under this system
    comb_dirs=$(find "$dataset_dir" -type d -name "comb_*" 2>/dev/null | sort)
    
    if [ -n "$comb_dirs" ]; then
        # This system has nested or direct combo directories
        echo "  - $dataset_name (combo structure):"
        system_has_combos=false
        while IFS= read -r comb_dir; do
            if [ -z "$comb_dir" ]; then
                continue
            fi
            comb_name=$(basename "$comb_dir")
            run_count=$(find "$comb_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            if [ $run_count -gt 0 ]; then
                system_has_combos=true
                total_systems=$((total_systems + 1))
                total_runs_available=$((total_runs_available + run_count))
                echo "      $comb_name: $run_count runs"
                pretrain_dirs="$pretrain_dirs --pretraining-dir $comb_dir"
            fi
        done <<< "$comb_dirs"
        
        # Check if there are also direct run directories at the system level
        # Look for directories that contain graph_info.json (indicating they are runs)
        direct_run_dirs=$(find "$dataset_dir" -maxdepth 1 -type d -exec test -f "{}/graph_info.json" \; -print 2>/dev/null | wc -l)
        if [ $direct_run_dirs -gt 0 ]; then
            total_systems=$((total_systems + 1))
            total_runs_available=$((total_runs_available + direct_run_dirs))
            echo "      [direct runs]: $direct_run_dirs runs"
            pretrain_dirs="$pretrain_dirs --pretraining-dir $dataset_dir"
        fi
    else
        # This system has no combo directories, check for direct runs
        run_count=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ $run_count -gt 0 ]; then
            total_systems=$((total_systems + 1))
            total_runs_available=$((total_runs_available + run_count))
            echo "  - $dataset_name: $run_count runs"
            pretrain_dirs="$pretrain_dirs --pretraining-dir $dataset_dir"
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
if [ -n "$MIN_TRANSITIONS" ] && [ "$MIN_TRANSITIONS" != "0" ]; then
    CMD="$CMD --min-transitions $MIN_TRANSITIONS"
    echo "Transition filtering: ENABLED (>= $MIN_TRANSITIONS per site)"
elif [ -n "$STRATIFIED_FRACTION" ] && [ "$STRATIFIED_FRACTION" != "0" ]; then
    CMD="$CMD --stratified-negative-fraction $STRATIFIED_FRACTION"
    echo "Stratified negative sampling: ENABLED (max ${STRATIFIED_FRACTION}, quadratic ramp)"
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

# Add early stopping patience
CMD="$CMD --patience $PATIENCE"
echo "Early stopping patience: $PATIENCE epochs"

# Add reward-weighted loss if requested
if [ "$REWARD_WEIGHTED" = "true" ]; then
    CMD="$CMD --reward-weighted"
    echo "Reward-weighted loss: ENABLED"
fi

# Add per-pair AWR control
if [ "$USE_PER_PAIR_AWR" = "false" ]; then
    CMD="$CMD --no-per-pair-awr"
    echo "Per-pair AWR: DISABLED (falling back to whole-run scalar weighting)"
else
    echo "Per-pair AWR: ENABLED (well-resolved pairs drive more gradient)"
fi

# Add AWR temperature if not default
if [ "$AWR_TEMPERATURE" != "1.0" ]; then
    CMD="$CMD --awr-temperature $AWR_TEMPERATURE"
fi

# Add NeuralLinear + Thompson Sampling Bayesian-head checkpoint generation
if [ "$BAYESIAN_HEADS" = "true" ]; then
    CMD="$CMD --bayesian-heads --bayesian-prior-precision $BAYESIAN_PRIOR_PRECISION"
    echo "NeuralLinear+TS: ENABLED (will also save best_policy_bayesian.pt / final_policy_bayesian.pt)"
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
    echo "Pairwise Weighting Summary:"
    echo "  • High-confidence edges (balanced populations, valid DDG): full gradient"
    echo "  • Low-confidence edges (imbalanced populations): reduced gradient"
    echo "  • Zero-confidence edges (missing pop/DDG): skipped entirely"
    echo ""
    echo "Output saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check pretrain_status.out for filtering statistics and per-run losses"
    echo "2. Review pairwise weighting report (run statistics)"
    echo "3. Update your workflow config to use the pretrained policy:"
    echo "   pretrain:"
    echo "     model_path: $OUTPUT_DIR/best_policy.pt"
    if [ "$BAYESIAN_HEADS" = "true" ]; then
        echo ""
        echo "   ...or, for NeuralLinear + Thompson Sampling online training"
        echo "   (training/workflow_instructions_neurallinear_ts.yaml), point at the"
        echo "   Bayesian-head sibling checkpoint instead:"
        echo "     pretrain:"
        echo "       model_path: $OUTPUT_DIR/best_policy_bayesian.pt"
        echo "     training:"
        echo "       policy:"
        echo "         use_bayesian_heads: true"
        echo "     bandit:"
        echo "       algorithm: neurallinear_ts"
    fi
    echo "4. Run the training workflow"
else
    echo "✗ Pretraining failed with exit code $EXIT_CODE"
    echo "Check pretrain_status.out for error details"
fi

exit $EXIT_CODE
