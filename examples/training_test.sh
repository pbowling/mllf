#!/bin/bash
#SBATCH --job-name=mllf_training
#SBATCH --output=status.out
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
# Run Python in unbuffered mode (-u) to see output in real-time
python -u run_workflow.py workflow_sample.yaml