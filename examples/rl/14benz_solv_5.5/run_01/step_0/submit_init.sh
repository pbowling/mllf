#!/bin/bash
#SBATCH --job-name=init_14benz
#SBATCH --output=init_14benz.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu2080 --gres=gpu:1 
#SBATCH --export=ALL
#SBATCH --time=01:00:00

#module load charmm
#export CHARMMEXEC='/home/dave/bin/charmm_6123706'
#module load anaconda
#conda activate  /home/dave/envs/charmm_6123706
module load charmm/charmm/c51a1
echo $CHARMMEXEC

cd /home/pbowling/mllf/examples/rl/14benz_solv_5.5
python3 msld_flat.py --vars-file run_01/step_0/variables0.py --out-dir run_01/step_0 > run_01/step_0/output.out
