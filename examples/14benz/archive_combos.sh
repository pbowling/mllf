#!/bin/bash
#SBATCH --job-name=archive_combos
#SBATCH --output=archive_combos_%j.out
#SBATCH --error=archive_combos_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu

# Archive the generated_combos directory
cd /home/pbowling/mllf/examples/14benz

echo "Starting archive at $(date)"
echo "Current directory: $(pwd)"
echo "Archiving generated_combos directory..."

tar -czvf pairwise_test1.tar.gz generated_combos/

if [ $? -eq 0 ]; then
    echo "Archive created successfully: pairwise_test1.tar.gz"
    ls -lh pairwise_test1.tar.gz
else
    echo "Error creating archive"
    exit 1
fi

echo "Completed at $(date)"
