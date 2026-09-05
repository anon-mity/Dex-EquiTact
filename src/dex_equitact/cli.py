"""Train and evaluate Dex-EquiTact policies using recorded demonstrations."""
import argparse
import json

import torch

from .config import PolicyConfig
from .data import EpisodeDataset, read_manifest
from .training import dataset_fingerprint, evaluate, load_policy, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1, help='CPU torch threads (default: 1)')
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate-data', help='Strict recorded-schema and complete-window check')
    validate.add_argument('--data', required=True)
    validate.add_argument('--slow-stride', type=int, default=16)
    fit = sub.add_parser('train', help='Train a separate policy for one embodiment')
    fit.add_argument('--data', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--config', required=True)
    fit.add_argument('--steps', type=int, default=1000, help='Total optimizer steps, including resumed steps')
    fit.add_argument('--batch-size', type=int, default=8)
    fit.add_argument('--learning-rate', type=float, default=1e-4)
    fit.add_argument('--weight-decay', type=float, default=1e-6)
    fit.add_argument('--seed', type=int, default=0)
    fit.add_argument('--val-fraction', type=float, default=0.2)
    fit.add_argument('--slow-stride', type=int, default=16)
    fit.add_argument('--device', default='cpu')
    fit.add_argument('--resume')
    fit.add_argument('--save-every', type=int, default=100)
    fit.add_argument('--validation-batches', type=int, default=4)
    ev = sub.add_parser('evaluate', help='Evaluate checkpoint on its recorded holdout split')
    ev.add_argument('--data', required=True)
    ev.add_argument('--checkpoint', required=True)
    ev.add_argument('--batch-size', type=int, default=4)
    ev.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'validate-data':
        manifest = read_manifest(args.data)
        ds = EpisodeDataset(args.data, [entry['id'] for entry in manifest['episodes']],
                            slow_stride=args.slow_stride)
        result = {'episodes': len(manifest['episodes']), 'windows': len(ds),
                  'action_dim': manifest['metadata']['action_dim'],
                  'sample_shapes': {key: list(value.shape) for key, value in ds[0].items()},
                  'validation_scope': 'Episode schema, numerical values, metadata, timestamps, and complete windows.'}
    elif args.command == 'train':
        config = PolicyConfig.from_yaml(args.config)
        path = train(args.data, args.output, config, steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed,
            val_fraction=args.val_fraction, slow_stride=args.slow_stride, device=args.device,
            resume=args.resume, save_every=args.save_every, validation_batches=args.validation_batches)
        result = {'checkpoint': str(path)}
    elif args.command == 'evaluate':
        policy, normalizer, state = load_policy(args.checkpoint, args.device)
        if dataset_fingerprint(args.data) != state['dataset_fingerprint']:
            raise ValueError('Evaluation dataset differs from the checkpoint dataset')
        ds = EpisodeDataset(args.data, state['split']['val'], normalizer,
                            slow_stride=state['settings']['slow_stride'])
        result = {'heldout_losses': evaluate(policy, ds, batch_size=args.batch_size),
                  'windows': len(ds)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
