#!/bin/bash
#SBATCH --job-name=pairwise_mlp_training
#SBATCH --output=training_status.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH --export=ALL
#SBATCH --time=120:00:00

# Initialize conda for bash shell
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the mllf environment
conda activate mllf

cd $SLURM_SUBMIT_DIR

# Run pairwise MLP workflow in unbuffered mode
python -u run_pairwise_workflow.py workflow_14benz_mlp.yaml
