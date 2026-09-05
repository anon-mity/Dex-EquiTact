import copy

import pytest
import torch

from dex_equitact.config import PolicyConfig
from dex_equitact.policy import DexEquiTactPolicy
from dex_equitact.streaming import ReactiveController


def tiny_config(action_dim=26):
    return PolicyConfig(action_dim=action_dim, proprio_dim=action_dim,
                        vector_channels=8, vector_layers=1, vector_heads=2,
                        model_dim=32, action_layers=1, action_heads=2,
                        train_diffusion_steps=12, inference_steps=3)


def make_batch(c, batch_size=1):
    return dict(images=torch.rand(batch_size, 2, c.num_views, 3, 16, 16),
                proprio=torch.randn(batch_size, 2, c.proprio_dim),
                positions=torch.randn(batch_size, 16, 5, 3),
                forces=torch.randn(batch_size, 16, 5, 3),
                future_forces=torch.randn(batch_size, 8, 5, 3),
                actions=torch.randn(batch_size, 16, c.action_dim))


@pytest.mark.parametrize('action_dim', [26, 28])
def test_real_losses_update_online_modules_and_ema(action_dim):
    torch.manual_seed(5)
    c = tiny_config(action_dim)
    policy = DexEquiTactPolicy(c).train()
    teacher_before = copy.deepcopy(policy.force_ema.state_dict())
    assert not policy.force_ema.training
    result = policy.loss(make_batch(c))
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    for module in [policy.tactile.position_encoder, policy.tactile.force_encoder,
                   policy.film, policy.predictor, policy.denoiser, policy.slow_encoder]:
        assert sum(p.grad.abs().sum().item() for p in module.parameters()
                   if p.grad is not None) > 0
    assert all(p.grad is None and not p.requires_grad for p in policy.force_ema.parameters())
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-3)
    optimizer.step()
    policy.update_ema(0.5)
    online = policy.tactile.force_encoder.state_dict()
    for k, target in policy.force_ema.state_dict().items():
        if target.is_floating_point():
            torch.testing.assert_close(target, 0.5 * teacher_before[k] + 0.5 * online[k])


def test_future_targets_are_next_one_through_eight_and_causal():
    c = tiny_config()
    policy = DexEquiTactPolicy(c).eval()
    b = make_batch(c)
    with torch.no_grad():
        all_latents = policy.force_ema(torch.cat([b['forces'], b['future_forces']], dim=1))
        targets = policy.future_targets(b['forces'], b['future_forces'])
    assert targets.shape == (1, 16, 8, 5, 8, 3)
    torch.testing.assert_close(targets[:, 0, 0], all_latents[:, 1])
    torch.testing.assert_close(targets[:, 15, 7], all_latents[:, 23])
    assert not targets.requires_grad
    k, eps = torch.ones(1, dtype=torch.long), torch.randn_like(b['actions'])
    loss_a = policy.loss(b, timesteps=k, noise=eps)['action_loss']
    b['future_forces'] *= 100
    loss_b = policy.loss(b, timesteps=k, noise=eps)['action_loss']
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)


def test_streaming_matches_batch_prefix_and_only_returns_latest():
    torch.manual_seed(13)
    c = tiny_config()
    policy = DexEquiTactPolicy(c).eval()
    b = make_batch(c)
    noise = torch.randn_like(b['actions'])
    controller = ReactiveController(policy)
    controller.start_cycle(b['images'], b['proprio'], initial_noise=noise, timestamp=0.0)
    with torch.no_grad():
        context = policy.slow_encoder(b['images'], b['proprio'])
        field = policy.film(policy.tactile(b['positions'], b['forces']), policy.film.gains(context))
        full = policy.diffusion.sample(policy.denoiser, noise, context, field, c.inference_steps)
    for j in range(4):
        newest = controller.step(b['positions'][:, j], b['forces'][:, j], timestamp=j + 0.1)
        assert newest.shape == (1, 26)
        torch.testing.assert_close(newest, full[:, j], rtol=2e-4, atol=2e-4)
    with pytest.raises(ValueError, match='timestamp'):
        controller.step(b['positions'][:, 4], b['forces'][:, 4], timestamp=3.1)


def test_bad_action_dim_and_incomplete_future_supervision_rejected():
    with pytest.raises(ValueError):
        PolicyConfig(action_dim=27)
    c = tiny_config()
    p = DexEquiTactPolicy(c)
    b = make_batch(c)
    b['future_forces'] = b['future_forces'][:, :7]
    with pytest.raises(ValueError, match='future_forces'):
        p.loss(b)


def test_first_tactile_can_share_visual_timestamp_but_cannot_repeat_across_cycles():
    c = tiny_config()
    p = DexEquiTactPolicy(c).eval()
    b = make_batch(c)
    ctrl = ReactiveController(p)
    ctrl.start_cycle(b['images'], b['proprio'], timestamp=0.0)
    ctrl.step(b['positions'][:, 0], b['forces'][:, 0], timestamp=0.0)
    ctrl.start_cycle(b['images'], b['proprio'], timestamp=0.0)
    with pytest.raises(ValueError, match='timestamp'):
        ctrl.step(b['positions'][:, 0], b['forces'][:, 0], timestamp=0.0)
