#!/bin/bash
#SBATCH --job-name=RL_14benz_solv_5.5
#SBATCH --output=RL_14benz_solv_5.5.%j
#SBATCH --ntasks=1 --tasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH -p gpu2080 --gres=gpu:1 
#SBATCH --export=ALL
#SBATCH --time=01:00:00

python msld_flat.py
