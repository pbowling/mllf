#!/bin/bash
# Complete workflow: Pretraining → CB Training
#
# This script demonstrates the full pipeline:
# 1. Pretrain policy on collected simulation data
# 2. Update your workflow config to use the pretrained policy
# 3. Run CB training with pretrained initialization
#
# Prerequisites:
# - Collected pretraining data in pretraining/* subdirectories
# - workflow_pretrain.yaml with matching model architecture
# - mllf conda environment activated
#
# IMPORTANT: Now uses ALL valid runs for pretraining (not just best)
# - Includes all runs where terminated_normally=True
# - All combo runs are included
# - No reward-based filtering
# - Benefits: More training data, captures coefficient diversity
# - Monitor MSE loss to determine if filtering is needed

set -e  # Exit on error

# Ensure we're using the mllf environment
if [[ "$CONDA_DEFAULT_ENV" != "mllf" ]]; then
    echo "Error: Please activate the mllf conda environment first:"
    echo "  conda activate mllf"
    exit 1
fi

echo "========================================="
echo "Step 1: Count Available Pretraining Data"
echo "========================================="
echo ""
echo "Scanning all available pretraining datasets..."
echo ""

# Find pretraining directory (could be in current dir or parent dir)
PRETRAIN_DIR=""
if [ -d "pretraining" ]; then
    PRETRAIN_DIR="pretraining"
elif [ -d "../pretraining" ]; then
    PRETRAIN_DIR="../pretraining"
else
    echo "Error: pretraining/ directory not found"
    echo "Expected location: pretraining/ or ../pretraining/"
    echo ""
    echo "Please create pretraining/ directory with subdirectories for each system:"
    echo "  pretraining/"
    echo "    ├── 14benz_solv/    # System 1 data"
    echo "    ├── indole_solv/    # System 2 data"
    echo "    └── ..."
    exit 1
fi

echo "Found pretraining directory: $PRETRAIN_DIR"
echo ""

# Auto-detect all subdirectories in pretraining/
# Note: Using ALL valid runs (terminated_normally=True) from all systems
# This includes all combo runs from nested structures like 14benz_pair_combos
total_systems=0
total_runs_available=0
for dataset_dir in $PRETRAIN_DIR/*/; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        # Count entries (could be run* directories or other directories like comb_*)
        count=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            total_systems=$((total_systems + 1))
            total_runs_available=$((total_runs_available + count))
            echo "  - $dataset_name: $count entries"
        fi
    fi
done

echo ""
echo "Total: $total_systems systems, $total_runs_available runs available"
echo ""

echo "========================================="
echo "Step 2: Collecting All Systems"
echo "========================================="
echo ""
echo "Including ALL runs from all systems for pretraining..."
echo "Note: Will use all valid runs (terminated normally) from each system"
echo ""

# Collect all system directories (no filtering)
pretrain_dirs=""
for dataset_dir in $PRETRAIN_DIR/*/; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        echo "  ✓ $dataset_name"
        pretrain_dirs="$pretrain_dirs --pretraining-dir $dataset_dir"
    fi
done

echo ""
echo "Will use all $total_runs_available runs from $total_systems systems for pretraining"
echo ""

echo "========================================="
echo "Step 3: Policy Pretraining"
echo "========================================="
echo ""

# Find models directory (could be in current dir or parent dir)
MODEL_FILE=""
if [ -f "models/pretrained_combined/best_policy.pt" ]; then
    MODEL_FILE="models/pretrained_combined/best_policy.pt"
    MODEL_DIR="models/pretrained_combined"
elif [ -f "../models/pretrained_combined/best_policy.pt" ]; then
    MODEL_FILE="../models/pretrained_combined/best_policy.pt"
    MODEL_DIR="../models/pretrained_combined"
fi

# Check if pretrained model already exists
if [ -n "$MODEL_FILE" ]; then
    echo "Pretrained model found at $MODEL_FILE"
    echo ""
    read -p "Do you want to retrain from scratch? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping pretraining. Using existing model."
        echo ""
        echo "========================================="
        echo "Step 3: Next Steps"
        echo "========================================="
        echo ""
        echo "Using existing pretrained policy! To use it:"
        echo ""
        echo "1. Edit your workflow config (e.g., examples/workflow_sample.yaml)"
        echo "   Update the pretrain section:"
        echo ""
        echo "   pretrain:"
        echo "     model_path: $MODEL_FILE"
        echo ""
        echo "2. Make sure you have generated combinations ready"
        echo "   (or configure create_combos in your workflow config)"
        echo ""
        echo "3. Run training:"
        echo "   python examples/run_workflow.py examples/workflow_sample.yaml"
        echo ""
        exit 0
    fi
    echo ""
    echo "Removing existing model and retraining..."
    rm -rf "$MODEL_DIR"
    echo ""
fi

echo "Pretraining policy via Behavior Cloning (supervised learning)..."
echo "This learns to predict bias coefficients from successful runs"
echo ""
echo "Approach: Uses ALL valid runs (terminated_normally=True) from all systems"
echo "Training with MSE loss for 50 epochs"
echo ""

if [ -z "$pretrain_dirs" ]; then
    echo "Error: No pretraining directories found"
    echo ""
    echo "Check your pretraining data structure."
    exit 1
fi

echo "Using pretraining directories:"
echo "$pretrain_dirs" | tr ' ' '\n' | grep "pretraining" | sed 's/--pretraining-dir/  -/'
echo ""

# Find config file (could be in current dir or examples/ dir)
CONFIG_FILE=""
if [ -f "workflow_pretrain.yaml" ]; then
    CONFIG_FILE="workflow_pretrain.yaml"
elif [ -f "examples/workflow_pretrain.yaml" ]; then
    CONFIG_FILE="examples/workflow_pretrain.yaml"
else
    echo "Error: workflow_pretrain.yaml not found"
    echo "Expected location: workflow_pretrain.yaml or examples/workflow_pretrain.yaml"
    exit 1
fi

# Pretrain on all datasets (behavior cloning with MSE loss)
# Note: By default pretrain_policy uses ALL runs that terminated normally
# Previously it filtered to best run per system - that behavior was removed
python -m mllf.cb.pretrain_policy \
    $pretrain_dirs \
    --output-dir models/pretraining \
    --config $CONFIG_FILE \
    --epochs 50 \
    --learning-rate 0.001

echo ""
echo "Pretraining complete! Saved to models/pretraining/"
echo ""

echo "========================================="
echo "Step 3: Next Steps"
echo "========================================="
echo ""
echo "Pretraining complete on $total_runs_available runs from $total_systems systems! To use the pretrained policy:"
echo ""
echo "1. Edit your workflow config (e.g., examples/workflow_sample.yaml)"
echo "   Update the pretrain section:"
echo ""
echo "   pretrain:"
echo "     model_path: models/pretraining/best_policy.pt"
echo ""
echo "2. Make sure you have generated combinations ready"
echo "   (or configure create_combos in your workflow config)"
echo ""
echo "3. Run training:"
echo "   python examples/run_workflow.py examples/workflow_sample.yaml"
echo ""
echo "The policy will start from pretrained weights (trained on $total_runs_available runs from $total_systems systems)"
echo "and fine-tune on new combinations."
echo ""
