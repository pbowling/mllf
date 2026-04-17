#!/bin/bash
#SBATCH --job-name=tar_triples
#SBATCH --output=tar_triples_%j.out
#SBATCH --ntasks=1
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -euo pipefail

BASE="/home/pbowling/mllf/examples/14benz"
DEST="${BASE}/1_april_2026_14benz_triples.tar.gz"

echo "Starting tar at $(date)"
echo "Source : ${BASE}/generated_combos{,_no14benz,_with14benz}"
echo "Output : $DEST"

tar --use-compress-program="pigz -p ${SLURM_CPUS_PER_TASK}" \
    -cvf "$DEST" \
    -C "$BASE" \
    generated_combos \
    generated_combos_no14benz \
    generated_combos_with14benz

echo "Finished at $(date)"
echo "Archive size: $(du -sh "$DEST" | cut -f1)"
