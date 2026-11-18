#!/bin/bash
#SBATCH --job-name=mllf_pretrain
#SBATCH --output=mllf_pretrain_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#
# Pretraining on every collected dataset under a pretraining/ directory, with
# the settings of the current models: the run filters, per-pair AWR with the
# online end-state success gate, Bayesian heads for online training
# (bandit.algorithm: hindsight), and each bias head's best epoch chosen on
# held-out combos. docs/cb_pretraining.rst explains each setting;
# examples/minimal/ is the smaller, single-system example.
#
# Submit from the repository root (or from examples/):
#   sbatch examples/pretrain_with_filtering.sh
#   sbatch --export=ALL,OUTPUT_DIR=models/my_run,EXCLUDE_DATASETS="sysA sysB" \
#       examples/pretrain_with_filtering.sh
#
# Settings (environment variables; defaults in parentheses):
#   PRETRAINING_DIR      collected datasets, one directory each (pretraining)
#   EXCLUDE_DATASETS     space-separated dataset names to leave out (none)
#   CONFIG_FILE          pretraining config, reward and unimol settings
#                        (examples/workflow_pretrain.yaml)
#   OUTPUT_DIR           where the model and its log go (models/pretraining)
#   EPOCHS               most epochs (100); PATIENCE: stop once no head has
#                        improved for this many epochs (5)
#   LEARNING_RATE        cosine annealed to 1/100 of it (0.0001)
#   VAL_FRACTION         combo directories held out to choose each head's best
#                        epoch (0.05)
#   STRATIFIED_FRACTION  keep every positive-reward run and up to this share of
#                        the best negative-reward bucket (0.90)
#   MAX_RUNS_PER_COMBO   runs kept per combo after that filter (15)
#   AWR_TEMPERATURE      beta in the advantage weight exp(A / beta) (0.5)
#   EXTRA_ARGS           any other mllf pretrain flags, e.g. "--no-data-hygiene"
#                        or "--emphasize-systems luis_tyk2:5" (mllf pretrain -h)
#   CONDA_ENV            conda environment to activate (mllf)
# A requeued job (e.g. after a node failure) resumes from its latest
# checkpoint in OUTPUT_DIR; delete that directory to start over.
set -euo pipefail

if [ -n "${CONDA_ENV-mllf}" ] && [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-mllf}"
fi
cd "${SLURM_SUBMIT_DIR:-.}"
[ -d examples ] || cd ..          # submitted from examples/

PRETRAINING_DIR="${PRETRAINING_DIR:-pretraining}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-}"
CONFIG_FILE="${CONFIG_FILE:-examples/workflow_pretrain.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-models/pretraining}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
STRATIFIED_FRACTION="${STRATIFIED_FRACTION:-0.90}"
MAX_RUNS_PER_COMBO="${MAX_RUNS_PER_COMBO:-15}"
AWR_TEMPERATURE="${AWR_TEMPERATURE:-0.5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for path in "$PRETRAINING_DIR" "$CONFIG_FILE"; do
    [ -e "$path" ] || { echo "Not found: $path (see the settings at the top of this script)"; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
echo "Started $(date) in $(pwd); output in $OUTPUT_DIR"

# The manifest: every combo directory (comb_*, pairNN_site*, groupNN_site*)
# with runs, a dataset's own runs, and each dataset without combos. Pretraining
# loads each run once, even where a dataset and its combos are both listed.
MANIFEST="$OUTPUT_DIR/pretraining_manifest_scan.json"
python - "$PRETRAINING_DIR" "$MANIFEST" $EXCLUDE_DATASETS <<'EOF'
import json, re, sys
from pathlib import Path
root, out, exclude = Path(sys.argv[1]), sys.argv[2], set(sys.argv[3:])
skip = exclude | {'embeddings_cache'}
combo = re.compile(r'^(comb_.*|pair[0-9]+.*_site.*|group[0-9]+.*_site.*)$')
dirs, runs = [], 0
for ds in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in skip):
    combos = sorted(d for d in ds.iterdir() if d.is_dir() and combo.match(d.name))
    own = [r for r in ds.iterdir() if r.is_dir() and (r / 'graph_info.json').exists()]
    for c in combos:
        n = sum(1 for r in c.iterdir() if r.is_dir() and r.name != 'prep')
        if n:
            dirs.append(str(c)); runs += n
    if own or not combos:
        n = len(own) if combos else sum(1 for r in ds.iterdir()
                                        if r.is_dir() and r.name != 'prep')
        if n:
            dirs.append(str(ds)); runs += n
Path(out).write_text(json.dumps({'directories': dirs}, indent=1))
print(f"manifest: {len(dirs)} directories, {runs} run directories"
      + (f"; excluded {sorted(exclude)}" if exclude else ''))
EOF

RESUME=()
LAST=$(ls "$OUTPUT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LAST" ]; then
    echo "Resuming from $LAST"
    RESUME=(--resume-checkpoint "$LAST")
fi

# The settings and the per-head loss table of each epoch go to the console and
# to $OUTPUT_DIR/pretrain.log; --analyze writes $OUTPUT_DIR/analysis at the end.
# shellcheck disable=SC2086  # EXTRA_ARGS is split into flags on purpose
mllf pretrain \
    --manifest-file "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --learning-rate "$LEARNING_RATE" \
    --val-fraction "$VAL_FRACTION" \
    --no-filter-outliers \
    --stratified-negative-fraction "$STRATIFIED_FRACTION" \
    --max-runs-per-combo "$MAX_RUNS_PER_COMBO" \
    --awr-temperature "$AWR_TEMPERATURE" \
    --bayesian-heads \
    --analyze \
    "${RESUME[@]}" $EXTRA_ARGS

echo "Finished $(date). For online training set, in the workflow config:"
echo "  pretrain:"
echo "    model_path: $OUTPUT_DIR/best_policy_bayesian.pt"
echo "and keep its unimol settings equal to the model's (pretrain_metadata.json)."
