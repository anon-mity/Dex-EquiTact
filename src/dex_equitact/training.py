"""Policy training, held-out evaluation, and complete resumable checkpoints."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .config import PolicyConfig
from .data import EpisodeDataset, Normalizer, read_manifest, split_episode_ids
from .policy import DexEquiTactPolicy


MANUSCRIPT_SHA256 = 'c49b2571b4e6110295ceb093aca40c28b61fb5104896645cdb0104b4e52cdaa5'


def dataset_fingerprint(root):
    root = Path(root)
    manifest = read_manifest(root)
    digest = hashlib.sha256()
    for filename in ['manifest.json'] + sorted(ep['file'] for ep in manifest['episodes']):
        digest.update(filename.encode())
        with (root / filename).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _device_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _numpy_rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return (name, keys.tolist(), pos, has_gauss, cached)


def _restore_rng(checkpoint):
    torch.set_rng_state(checkpoint['rng']['torch'].cpu())
    random.setstate(checkpoint['rng']['python'])
    name, keys, pos, has_gauss, cached = checkpoint['rng']['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    if torch.cuda.is_available() and checkpoint['rng']['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['rng']['cuda']])


def _save(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_policy(checkpoint_path, device='cpu'):
    """Load this repository's format only, with PyTorch's restricted tensor loader."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if state.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    policy = DexEquiTactPolicy(PolicyConfig(**state['config'])).to(device)
    policy.load_state_dict(state['model'], strict=True)
    policy.eval()
    return policy, Normalizer.from_state_dict(state['normalizer']), state


@torch.no_grad()
def evaluate(policy, dataset, *, batch_size=4, seed=0, max_batches=None):
    """Evaluate action-denoising and future-force latent losses on held-out data."""
    if len(dataset) == 0:
        raise ValueError('Validation dataset has no complete windows')
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    totals, count = {}, 0
    # Validation must not change the subsequent training diffusion RNG stream.
    cuda_devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for number, start in enumerate(range(0, len(dataset), batch_size)):
                if max_batches is not None and number >= max_batches:
                    break
                examples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                loss = policy.loss(_device_batch(default_collate(examples), device))
                for key, value in loss.items():
                    totals[key] = totals.get(key, 0.0) + value.item() * len(examples)
                count += len(examples)
    finally:
        policy.train(was_training)
    if count == 0:
        raise ValueError('No validation batches requested')
    return {key: value / count for key, value in totals.items()}


def train(data_root, output_dir, config, *, steps=1000, batch_size=8, learning_rate=1e-4,
          weight_decay=1e-6, seed=0, val_fraction=0.2, slow_stride=16, device='cpu',
          resume=None, save_every=100, validation_batches=4):
    """Train to total `steps`; resuming preserves sample selection and noise RNG.

    Sampling is uniform over full windows, with replacement. Each run trains
    one embodiment policy and records its optimizer and sampling configuration.
    """
    if steps < 1 or batch_size < 1 or save_every < 1 or learning_rate <= 0:
        raise ValueError('steps, batch_size, save_every and learning_rate must be positive')
    if weight_decay < 0 or validation_batches < 1:
        raise ValueError('Invalid weight decay or validation batch count')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f'Output exists; use resume or a new directory: {output}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    manifest = read_manifest(data_root)
    metadata = manifest['metadata']
    for key, expected in [('action_dim', config.action_dim), ('proprio_dim', config.proprio_dim),
                          ('num_cameras', config.num_views)]:
        if metadata[key] != expected:
            raise ValueError(f'Dataset {key}={metadata[key]} disagrees with policy {expected}')
    fingerprint = dataset_fingerprint(data_root)
    settings = dict(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
                    seed=seed, val_fraction=val_fraction, slow_stride=slow_stride)
    previous = None
    if resume is not None:
        previous = torch.load(resume, weights_only=True, map_location=device)
        if previous.get('format_version') != 1:
            raise ValueError('Unsupported resume format')
        if previous['config'] != config.to_dict() or previous['settings'] != settings:
            raise ValueError('Resume config/training settings differ from checkpoint')
        if previous['dataset_fingerprint'] != fingerprint:
            raise ValueError('Dataset contents changed since checkpoint')
        if previous['step'] >= steps:
            raise ValueError('Total steps must exceed checkpoint step')
        split = previous['split']
        normalizer = Normalizer.from_state_dict(previous['normalizer'])
    else:
        train_ids, val_ids = split_episode_ids(data_root, val_fraction, seed)
        split = {'train': train_ids, 'val': val_ids}
        if not split['val']:
            raise ValueError('Training requires a separate validation episode')
        normalizer = Normalizer.fit(data_root, split['train'])
    train_data = EpisodeDataset(data_root, split['train'], normalizer, slow_stride=slow_stride)
    val_data = EpisodeDataset(data_root, split['val'], normalizer, slow_stride=slow_stride)
    if not len(train_data) or not len(val_data):
        raise ValueError('Dataset lacks complete train/validation 16+8 windows and slow history')
    policy = DexEquiTactPolicy(config).to(device).train()
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    sampling_rng = torch.Generator().manual_seed(seed + 1009)
    start = 0
    if previous:
        policy.load_state_dict(previous['model'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        sampling_rng.set_state(previous['sampling_rng'].cpu())
        _restore_rng(previous)
        start = previous['step']
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(manuscript_sha256=MANUSCRIPT_SHA256, dataset_fingerprint=fingerprint,
                      dataset_metadata=metadata, torch_version=str(torch.__version__),
                      numpy_version=np.__version__, device=str(device))
    (output / 'run.json').write_text(json.dumps(dict(config=config.to_dict(), settings=settings,
                                      split=split, provenance=provenance), indent=2) + '\n')
    last_path = output / 'last.pt'
    for step in range(start + 1, steps + 1):
        indices = torch.randint(len(train_data), (batch_size,), generator=sampling_rng).tolist()
        batch = _device_batch(default_collate([train_data[i] for i in indices]), device)
        optimizer.zero_grad(set_to_none=True)
        losses = policy.loss(batch)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        policy.update_ema()
        metrics = {'step': step, **{key: value.item() for key, value in losses.items()}}
        if step % save_every == 0 or step == steps:
            metrics['validation'] = evaluate(policy, val_data, batch_size=batch_size,
                                              seed=seed, max_batches=validation_batches)
            checkpoint = dict(format_version=1, step=step, config=config.to_dict(), settings=settings,
                model=policy.state_dict(), optimizer=optimizer.state_dict(),
                normalizer=normalizer.state_dict(), split=split, dataset_fingerprint=fingerprint,
                sampling_rng=sampling_rng.get_state(), provenance=provenance,
                rng={'torch': torch.get_rng_state(), 'python': random.getstate(),
                     'numpy': _numpy_rng_state(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
            _save(last_path, checkpoint)
        with (output / 'metrics.jsonl').open('a') as log:
            log.write(json.dumps(metrics) + '\n')
    return last_path
