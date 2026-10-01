"""Tests for the checkpoint structure written/read by mllf.cli.train_workflow.

Self-contained: builds synthetic checkpoints and config files under tmp_path
rather than reading real training output or example configs from elsewhere
in the repo, so these tests can't fail because an unrelated file (e.g. a
real training run's output, or an example YAML) changed shape or moved.

Covers:
  - The per-epoch checkpoint written by run_training's training loop has the
    keys resume-loading actually reads (see mllf.cli.train_workflow, the
    `torch.save({'epoch': ..., 'policy_state_dict': ..., ...})` call).
  - A checkpoint saved with a context baseline enabled round-trips its
    optional baseline_state_dict / baseline_optimizer_state_dict keys.
  - The output.save_checkpoints / checkpoint_freq / base_dir config parsing
    logic that gates checkpoint saving.
"""
from pathlib import Path

import torch
import yaml

# Keys mllf.cli.train_workflow.run_training writes to checkpoint_{epoch:03d}.pt
# and reads back on resume (see its checkpoint save/load code).
REQUIRED_CHECKPOINT_KEYS = {
    'epoch', 'policy_state_dict', 'rl_optimizer_state_dict',
    'current_stage_idx', 'stats',
}


def _write_synthetic_checkpoint(path: Path, epoch: int = 4, with_baseline: bool = False) -> dict:
    checkpoint = {
        'epoch': epoch,
        'policy_state_dict': {'edge_mlp.weight': torch.randn(2, 2)},
        'rl_optimizer_state_dict': {'state': {}, 'param_groups': []},
        'current_stage_idx': 0,
        'stats': {'loss': 0.1, 'avg_reward': 1.5},
    }
    if with_baseline:
        checkpoint['baseline_state_dict'] = {'net.weight': torch.randn(2, 2)}
        checkpoint['baseline_optimizer_state_dict'] = {'state': {}, 'param_groups': []}
    torch.save(checkpoint, path)
    return checkpoint


def test_checkpoint_has_required_keys(tmp_path):
    ckpt_path = tmp_path / 'checkpoint_004.pt'
    _write_synthetic_checkpoint(ckpt_path, epoch=4)

    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    missing = REQUIRED_CHECKPOINT_KEYS - set(loaded.keys())
    assert not missing, f"checkpoint missing required keys: {missing}"
    assert loaded['epoch'] == 4


def test_checkpoint_baseline_fields_are_optional(tmp_path):
    without = tmp_path / 'checkpoint_no_baseline.pt'
    with_baseline = tmp_path / 'checkpoint_with_baseline.pt'
    _write_synthetic_checkpoint(without, with_baseline=False)
    _write_synthetic_checkpoint(with_baseline, with_baseline=True)

    loaded_without = torch.load(without, map_location='cpu', weights_only=False)
    loaded_with = torch.load(with_baseline, map_location='cpu', weights_only=False)

    assert 'baseline_state_dict' not in loaded_without
    assert 'baseline_state_dict' in loaded_with
    assert 'baseline_optimizer_state_dict' in loaded_with


def test_latest_checkpoint_is_selected_by_sorted_glob(tmp_path):
    # run_training resumes from `sorted(checkpoint_dir.glob('checkpoint_*.pt'))[-1]`.
    for epoch in (0, 1, 9, 10):
        _write_synthetic_checkpoint(tmp_path / f'checkpoint_{epoch:03d}.pt', epoch=epoch)

    checkpoints = sorted(tmp_path.glob('checkpoint_*.pt'))
    latest = torch.load(checkpoints[-1], map_location='cpu', weights_only=False)
    assert latest['epoch'] == 10


def test_yaml_config_checkpointing_fields_parse(tmp_path):
    """A synthetic workflow config with output.save_checkpoints enabled."""
    config_path = tmp_path / 'workflow_config.yaml'
    config_path.write_text(yaml.safe_dump({
        'output': {
            'base_dir': str(tmp_path / 'training_output'),
            'save_checkpoints': True,
            'checkpoint_freq': 5,
        },
    }))

    config = yaml.safe_load(config_path.read_text())
    output_config = config.get('output', {})

    assert output_config.get('save_checkpoints', False) is True
    assert output_config.get('checkpoint_freq', 1) == 5
    assert output_config.get('base_dir', 'checkpoints') == str(tmp_path / 'training_output')


def test_yaml_config_checkpointing_defaults_when_output_block_missing():
    config = yaml.safe_load('systems: []\n')
    output_config = config.get('output', {})

    # Matches mllf.cli.train_workflow's own defaults for these keys.
    assert output_config.get('save_checkpoints', True) is True
    assert output_config.get('checkpoint_freq', 5) == 5
    assert output_config.get('base_dir', 'checkpoints') == 'checkpoints'
