#!/bin/bash
#SBATCH --job-name=train_with14benz
#SBATCH --output=status_with14benz_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH --export=ALL
#SBATCH --time=120:00:00

# Training run using pretrained policy from pretraining with 14benz pair/triplet combos.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mllf

cd $SLURM_SUBMIT_DIR

python -u run_workflow_deepset.py workflow_14benz_with14benz.yaml
