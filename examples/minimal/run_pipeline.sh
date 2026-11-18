#!/bin/bash
#SBATCH --job-name=mllf_minimal
#SBATCH --output=mllf_minimal_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH --time=06:00:00
#SBATCH --export=ALL
#
# The whole pipeline on one system (1,4-benzene in water), small enough to check
# an installation end to end:
#
#   1. pretrain     up to 10 epochs of behavior cloning on 39 collected 14benz
#                   runs; each bias head keeps its best epoch on held-out runs
#   2. replay       the regression replay set, from the runs the model saw
#   3. scout        one scouting run of the system's group with the new model
#   4. train        one online epoch (at least one combo, even if scouting passes)
#   5. cost         simulated time of the online run
#
# Submit from the repository root:
#   sbatch examples/minimal/run_pipeline.sh
# Settings (environment variables):
#   CHARMM_TOPPAR     directory of the CHARMM36 topology and parameter files
#                     (top_all36_prot.rtf, par_all36_cgenff.prm,
#                     toppar_water_ions.str, ...), read by every simulation
#                     (examples/minimal/14benz/msld_flat.py); required
#   PRETRAINING_ROOT  collected runs, holding 14benz_solv/ (default: the 39 runs
#                     shipped in examples/minimal/pretraining/)
#   MLLF              the mllf command (default: mllf)
#   CONDA_ENV         conda environment to activate (default: mllf)
# Outputs go to examples/minimal/output/; a rerun skips the steps whose output
# exists, so delete that directory to start over.
set -euo pipefail

PRETRAINING_ROOT="${PRETRAINING_ROOT:-examples/minimal/pretraining}"
MLLF="${MLLF:-mllf}"
if [ -n "${CONDA_ENV-mllf}" ] && [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-mllf}"
fi
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
for f in top_all36_prot.rtf par_all36m_prot.prm top_all36_cgenff.rtf par_all36_cgenff.prm \
         toppar_water_ions.str; do
    if [ ! -f "${CHARMM_TOPPAR:-}/$f" ]; then
        echo "Set CHARMM_TOPPAR to the CHARMM36 toppar directory (missing ${CHARMM_TOPPAR:-<unset>}/$f)"
        exit 1
    fi
done
export CHARMM_TOPPAR     # the simulation jobs inherit it (sbatch exports the environment)
EX=examples/minimal
OUT=$EX/output
mkdir -p "$OUT"
echo "Started $(date); outputs in $OUT"

# 1. Pretraining. The manifest lists the collected runs to read.
if [ ! -f "$OUT/pretrain/best_policy_bayesian.pt" ]; then
    python - "$PRETRAINING_ROOT" "$EX/pretraining_combos.txt" "$OUT/manifest.json" <<'EOF'
import json, sys
from pathlib import Path
root, names, out = Path(sys.argv[1]) / '14benz_solv', sys.argv[2], sys.argv[3]
dirs = [str(root / n.strip()) for n in open(names) if n.strip() and not n.startswith('#')]
missing = [d for d in dirs if not Path(d).is_dir()]
if missing:
    sys.exit(f"{len(missing)} pretraining directories missing, e.g. {missing[0]}")
Path(out).write_text(json.dumps({'directories': dirs}, indent=1))
print(f"manifest: {len(dirs)} directories")
EOF
    $MLLF pretrain --manifest-file "$OUT/manifest.json" --output-dir "$OUT/pretrain" \
        --config "$EX/pretrain.yaml" --epochs 10 --bayesian-heads --device cpu
fi

# 2. Regression replay set (bandit.regression.replay_cache).
if [ ! -f "$OUT/replay.pt" ]; then
    $MLLF replay "$EX/workflow.yaml"
fi

# 3. Scouting: submit the group's run, wait for it, summarize (stage key scouting:).
if [ ! -f "$OUT/scouting/scout_summary.json" ]; then
    $MLLF scout "$EX/workflow.yaml" --out "$OUT/scouting" --run-steps 13000 39000 \
        --wait --poll 30
fi

# 4. One online epoch, then 5. its simulated time.
$MLLF train "$EX/workflow.yaml"
$MLLF cost "$OUT/training"
echo "Finished $(date)"
