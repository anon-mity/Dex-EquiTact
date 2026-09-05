"""Explicitly synthetic fixtures for exercising software, never robot results."""
import json
from pathlib import Path

import numpy as np


def create_synthetic_dataset(root, *, action_dim=26, frames=48, episodes=3, seed=0):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite dataset directory: {root}')
    if action_dim not in (26, 28) or frames < 40 or episodes < 2:
        raise ValueError('Synthetic fixture requires action_dim 26/28, >=40 frames, >=2 episodes')
    (root / 'episodes').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = {'schema_version': 1, 'synthetic': True, 'metadata': {
        'finger_order': ['thumb', 'index', 'middle', 'ring', 'little'],
        'position_frame': 'wrist', 'position_unit': 'm', 'force_frame': 'sensor_local',
        'force_unit': 'N', 'force_axes': 'common_calibrated',
        'action_convention': 'arm_delta_hand_target', 'action_dim': action_dim,
        'proprio_dim': action_dim, 'num_cameras': 2, 'timestamp_unit': 's',
        'arm_translation_frame': 'base', 'arm_rotation_representation': 'rotation_vector',
        'arm_rotation_composition': 'left_multiply', 'hand_command_unit': 'rad',
        'position_source': 'synthetic', 'force_calibration_id': 'synthetic'}, 'episodes': []}
    for i in range(episodes):
        times = np.arange(frames, dtype=np.float64) * 0.02
        slow_index = np.arange(frames) // 16
        slow_images = rng.integers(0, 256, (int(slow_index[-1]) + 1, 2, 3, 16, 16), dtype=np.uint8)
        arrays = {
            'images': slow_images[slow_index],
            'proprio': rng.normal(0, 0.2, (frames, action_dim)).astype('float32'),
            'positions': rng.normal(0, 0.05, (frames, 5, 3)).astype('float32'),
            'forces': rng.normal(0, 0.3, (frames, 5, 3)).astype('float32'),
            'actions': rng.normal(0, 0.01, (frames, action_dim)).astype('float32'),
            'timestamps': times,
            'image_timestamps': np.repeat(times[slow_index * 16, None], 2, axis=1)}
        episode_id = f'synthetic_{i:03d}'
        filename = f'episodes/{episode_id}.npz'
        np.savez(root / filename, **arrays)
        manifest['episodes'].append({'id': episode_id, 'file': filename})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return root
