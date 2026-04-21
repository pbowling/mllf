#!/bin/bash
#SBATCH --job-name=pretrain_with14benz
#SBATCH --output=pretrain_with14benz_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p cpu
#SBATCH --export=ALL
#SBATCH --time=15:00:00

# Pretraining run 3: same low-quality exclusions as the previous pretraining run,
# but NOW including the 14benz_pair_combos and 14benz_triplet_combos datasets
# (14benz_pair_combos was excluded before; 14benz_triplet_combos is new).
# 14benz_solv and 14benz_vac were always included, so they remain included here.

export OUTPUT_DIR="models/pretraining_representative_system"
export USE_BEST_ONLY=true
export REWARD_THRESHOLD=0
export STRATIFIED_FRACTION=0.0
export Q_EPOCHS=20
export Q_LR=1e-3
export Q_STRATIFIED_FRACTION=0.55
export LEARNING_RATE=0.0001
export PATIENCE=20

export EXCLUDE_DATASETS="\
luis_cdk2_protein_group1 luis_cdk2_protein_group2 luis_cdk2_solvent_group1 luis_cdk2_solvent_group2 \
luis_ptp1b_protein_group1 luis_ptp1b_solvent_group1 \
p38_protein_groupA p38_protein_groupB p38_protein_groupC \
mup1_solvent_group2 luis_p38_protein_group1 luis_p38_protein_group2"

echo "=== Pretraining run: previous baseline + 14benz pair/triplet combos ==="
echo "Output dir: $OUTPUT_DIR"
echo "Excluded: $EXCLUDE_DATASETS"
echo ""

# ── Pretraining ───────────────────────────────────────────────────────────────
source "$SLURM_SUBMIT_DIR/pretrain_with_filtering.sh"
