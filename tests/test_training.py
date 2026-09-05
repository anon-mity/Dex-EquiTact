from pathlib import Path

import torch

from dex_equitact.synthetic import create_synthetic_dataset
from dex_equitact.training import train, load_policy
from test_policy import tiny_config


def test_checkpoint_roundtrip_and_exact_cpu_resume(tmp_path):
    data = tmp_path / 'data'
    create_synthetic_dataset(data, frames=42, episodes=3)
    c = tiny_config()
    full = train(data, tmp_path / 'full', c, steps=2, batch_size=1, seed=21)
    part = train(data, tmp_path / 'part', c, steps=1, batch_size=1, seed=21)
    resumed = train(data, tmp_path / 'part', c, steps=2, batch_size=1, seed=21, resume=part)
    a = torch.load(full, weights_only=True, map_location='cpu')
    b = torch.load(resumed, weights_only=True, map_location='cpu')
    assert a['step'] == b['step'] == 2
    assert a['split'] == b['split']
    assert a['normalizer']['train_episode_ids'] == a['split']['train']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    model, normalizer, checkpoint = load_policy(full)
    assert not model.training and not model.force_ema.training
    assert checkpoint['step'] == 2
    assert normalizer is not None
    assert Path(full).is_file()
