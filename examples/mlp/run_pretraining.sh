#!/bin/bash
# Pairwise MLP Policy Pretraining Script
#
# This script demonstrates pretraining the pairwise MLP policy on collected simulation data.
# The pairwise MLP predicts bias coefficients directly from substituent features without
# requiring graph construction.
#
# Prerequisites:
# - Collected pretraining data in ../../pretraining/* subdirectories
# - workflow_pretrain.yaml with pairwise MLP architecture
# - mllf conda environment activated
#
# Differences from Graph-Based Policy:
# - No graph construction: Uses substituent features directly
# - Pairwise predictions: Predicts for each directed pair
# - Simpler architecture: MLP on difference features (feat_i - feat_j)
# - Faster training: No RGCN message passing, smaller input dimension

set -e  # Exit on error

# Ensure we're using the mllf environment
if [[ "$CONDA_DEFAULT_ENV" != "mllf" ]]; then
    echo "Error: Please activate the mllf conda environment first:"
    echo "  conda activate mllf"
    exit 1
fi

echo "========================================="
echo "Pairwise MLP Policy Pretraining"
echo "========================================="
echo ""

# Find pretraining directory (look in parent directories)
PRETRAIN_DIR=""
if [ -d "pretraining" ]; then
    PRETRAIN_DIR="pretraining"
elif [ -d "../pretraining" ]; then
    PRETRAIN_DIR="../pretraining"
elif [ -d "../../pretraining" ]; then
    PRETRAIN_DIR="../../pretraining"
else
    echo "Error: pretraining/ directory not found"
    echo "Expected location: pretraining/ or ../pretraining/ or ../../pretraining/"
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

# Count available data
echo "Scanning available pretraining datasets..."
echo ""

total_systems=0
total_runs=0
for dataset_dir in $PRETRAIN_DIR/*/; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        count=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            total_systems=$((total_systems + 1))
            total_runs=$((total_runs + count))
            echo "  - $dataset_name: $count runs"
        fi
    fi
done

echo ""
echo "Total: $total_systems systems, $total_runs runs available"
echo ""

# Filter systems by reward quality
echo "========================================="
echo "Filtering Systems by Reward Quality"
echo "========================================="
echo ""

python3 << 'FILTER_EOF'
import sys
import json
from pathlib import Path

# Import reward calculation
from mllf.cb.pretrain_policy import compute_reward_from_sim_results

# Find pretraining directory
for candidate in [Path("pretraining"), Path("../pretraining"), Path("../../pretraining")]:
    if candidate.exists():
        pretrain_dir = candidate
        break
else:
    print("Error: Could not find pretraining directory")
    sys.exit(1)

good_systems = []
bad_systems = []

for dataset_dir in sorted(pretrain_dir.glob("*/")):
    if not dataset_dir.is_dir():
        continue
    
    dataset_name = dataset_dir.name
    
    # Special case: 14benz_pair_combos has nested comb_*/run_* structure
    # These are already the best runs from training, so we include all of them
    if dataset_name == "14benz_pair_combos":
        for combo_dir in sorted(dataset_dir.glob("comb_*")):
            if not combo_dir.is_dir():
                continue
            
            combo_name = f"{dataset_name}/{combo_dir.name}"
            max_reward = -float('inf')
            best_run = None
            
            for run_dir in combo_dir.glob("run_*"):
                metadata_file = run_dir / "metadata.json"
                sim_file = run_dir / "simulation_results.json"
                
                if not metadata_file.exists() or not sim_file.exists():
                    continue
                
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    if not metadata.get("terminated_normally", False):
                        continue
                    
                    with open(sim_file) as f:
                        sim_results = json.load(f)
                    
                    num_sites = metadata.get("num_sites", 1)
                    num_subs = metadata.get("num_substituents", 2)
                    # For pair combos, typically single site with 2 substituents
                    nsubs_per_site = [num_subs]
                    
                    reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site)
                    
                    if reward > max_reward:
                        max_reward = reward
                        best_run = run_dir.name
                
                except Exception as e:
                    continue
            
            if max_reward > 0:
                good_systems.append((combo_name, max_reward, best_run))
                print(f"  ✓ {combo_name}: best reward = {max_reward:.2f} (run {best_run})")
            elif max_reward > -float('inf'):
                bad_systems.append((combo_name, max_reward, best_run))
                print(f"  ✗ {combo_name}: best reward = {max_reward:.2f} (run {best_run}) - EXCLUDED")
        continue
    
    # Standard case: run directories directly under dataset directory
    run_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        continue
    
    max_reward = -float('inf')
    best_run = None
    
    for run_dir in run_dirs:
        metadata_file = run_dir / "metadata.json"
        sim_file = run_dir / "simulation_results.json"
        
        if not metadata_file.exists() or not sim_file.exists():
            continue
        
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            if not metadata.get("terminated_normally", False):
                continue
            
            with open(sim_file) as f:
                sim_results = json.load(f)
            
            num_sites = metadata.get("num_sites", 1)
            num_subs = metadata.get("num_substituents", 1)
            nsubs_per_site = [num_subs] if num_sites == 1 else [num_subs // 2, num_subs - num_subs // 2]
            
            reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site)
            
            if reward > max_reward:
                max_reward = reward
                best_run = run_dir.name
        
        except Exception as e:
            continue
    
    if max_reward > 0:
        good_systems.append((dataset_name, max_reward, best_run))
        print(f"  ✓ {dataset_name}: best reward = {max_reward:.2f} (run {best_run})")
    else:
        bad_systems.append((dataset_name, max_reward, best_run))
        print(f"  ✗ {dataset_name}: best reward = {max_reward:.2f} (run {best_run}) - EXCLUDED")

print()
print(f"Summary: {len(good_systems)} systems with positive rewards, {len(bad_systems)} systems excluded")

# Write good systems to temp file
with open("/tmp/pretrain_good_systems.txt", "w") as f:
    for system_name, _, _ in good_systems:
        f.write(f"{system_name}\n")

sys.exit(0 if good_systems else 1)
FILTER_EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: No systems with positive rewards found!"
    exit 1
fi

# Build pretraining directory arguments
pretrain_dirs=""
while IFS= read -r system_name; do
    pretrain_dirs="$pretrain_dirs --pretraining-dir $PRETRAIN_DIR/$system_name"
done < /tmp/pretrain_good_systems.txt

echo ""
echo "========================================="
echo "Running Pretraining"
echo "========================================="
echo ""

# Check for existing model
MODEL_DIR="models/pretrained_pairwise"
if [ -d "$MODEL_DIR" ]; then
    echo "Existing pretrained model found at $MODEL_DIR"
    read -p "Do you want to retrain from scratch? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping pretraining. Using existing model."
        exit 0
    fi
    echo "Removing existing model and retraining..."
    rm -rf "$MODEL_DIR"
    echo ""
fi

echo "Training pairwise MLP policy via Behavior Cloning..."
echo "This learns to predict bias coefficients from successful runs"
echo ""
echo "Architecture:"
echo "  - Input: Difference features (feat_i - feat_j) = 178 dims"
echo "  - Features: Count-based encoding (CGenFF vocab: 161 types, 14 elements)"
echo "  - Shared trunk: [256, 128] with ReLU and dropout"
echo "  - Bias-type embeddings: 16-dim per bias type"
echo "  - Separate heads: 3-layer networks per bias type"
echo "  - Directionality: i→j preserved by difference (opposite of j→i)"
echo ""
echo "Training with MSE loss for 50 epochs"
echo ""

# Run pretraining
python -m mllf.cb.pretrain_pairwise_policy \
    $pretrain_dirs \
    --output-dir "$MODEL_DIR" \
    --config workflow_pretrain.yaml \
    --epochs 50 \
    --learning-rate 0.001

echo ""
echo "========================================="
echo "Pretraining Complete!"
echo "========================================="
echo ""
echo "Model saved to: $MODEL_DIR/best_policy.pt"
echo ""
echo "To use the pretrained policy in training:"
echo "  1. Edit workflow_train.yaml"
echo "  2. Set pretrain.model_path: $MODEL_DIR/best_policy.pt"
echo "  3. Run: python run_training.py workflow_train.yaml"
echo ""
