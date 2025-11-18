# Training setup for examples/cb/14benz_solv_5.5

This document describes the example workflow and the helper utilities present
in `setup_training.py` for preparing a training session.

Overview
--------
- The example contains a combos file `combos_14benz_solv_5.5.txt` listing all
  combination tokens produced by the generator.
- `setup_training.py` provides functions to split the combos into train/val/test
  manifests, scaffold a small sample of per-combo directories with required
  template files, and produce a `variables.py` file from a toy predicted graph.

Files created by the scaffold
----------------------------
When you scaffold a combo (using `scaffold_combo` or `scaffold_sample`) the
following are created inside `examples/cb/14benz_solv_5.5/scaffolded/<combo_name>`:

- `msld_flat.py` (copied from the example root if present)
- `run.sh` (copied)
- `prep/` (copied directory)
- `site#_sub#.rtf` (copied RTF fragment files used for this combo)
- `variables.py` (generated from a toy predicted graph)
- `scaffold_metadata.json` (metadata describing tokens and files created)

How the variables.py is created
------------------------------
1. A toy Graph is created with one node per substituent in the combo.
2. Each edge is assigned deterministic random coefficients (linear/quadratic/skew/end).
3. A temporary `.inp` bias file is written via `write_bias_inp_from_graph`.
4. `write_variables_py_from_inp` converts the `.inp` to a `variables.py` that
   contains a YAML `bias_string` with `b` (flattened lams) and full `c/x/s` matrices.

Notes and next steps
--------------------
- The scaffold uses a toy predictor. Replace the prediction step with your
  model inference pipeline to produce real edge coefficients.
- You can easily extend `setup_training.py` to:
  - read parsed RTF results and populate a Graph via `Graph.from_rtf_results`;
  - generate larger numbers of combo dirs; or
  - write additional job scripts / slurm templates per combo.
- Training harness: create a `train.py` that loads `variables.py` and runs
  the chosen training loop (checkpointing with `torch.save` or similar).
- Archival: after checkpoint rotation, use `tar`/`zip` to compress older
  checkpoints and store a small JSON manifest with epoch/score/file info.

Quick start
-----------
From the repository root:

```bash
python examples/cb/14benz_solv_5.5/setup_training.py
```

This will scaffold 5 example combo directories in
`examples/cb/14benz_solv_5.5/scaffolded/` and print their paths.

Recommended follow-ups
---------------------
- Add a training script that accepts a manifest file and runs over the train
  set, saving checkpoints and a small metadata file per checkpoint (epoch,
  validation score, best flag). Keep a rotation policy (e.g., keep N best)
  and compress older checkpoints to a tarball with the metadata included.
- Add unit tests for `setup_training.py` to ensure manifests and scaffolds are
  created as expected.
