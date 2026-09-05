"""Independent lifecycle checks using the production diffusion noise schedule.

These are software contract checks on tiny randomly initialized models, not
trained-policy performance measurements. In contrast to short-schedule unit
tests, the controller tests retain the production 1000/10 diffusion defaults.
"""

import pytest
import torch
from torch import nn

from dex_equitact.config import PolicyConfig
from dex_equitact.policy import DexEquiTactPolicy
from dex_equitact.streaming import ReactiveController


def _small_model_config(action_dim=26):
    return PolicyConfig(
        action_dim=action_dim,
        proprio_dim=action_dim,
        vector_channels=8,
        vector_layers=1,
        vector_heads=2,
        model_dim=32,
        action_layers=2,
        action_heads=2,
    )


class _ForbiddenTrainingHead(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("A training-only head was called during reactive inference")


@pytest.mark.parametrize("action_dim", [26, 28])
def test_production_schedule_full_cycle_preserves_prefix_and_inference_boundaries(action_dim):
    torch.manual_seed(231)
    config = _small_model_config(action_dim)
    assert config.train_diffusion_steps == 1000
    assert config.inference_steps == 10
    policy = DexEquiTactPolicy(config).eval()
    images = torch.rand(1, 2, config.num_views, 3, 16, 16)
    proprio = torch.randn(1, 2, action_dim)
    positions = torch.randn(1, 16, 5, 3)
    forces = torch.randn(1, 16, 5, 3)
    initial_noise = torch.randn(1, 16, action_dim)
    original_noise = initial_noise.clone()
    controller = ReactiveController(policy)
    controller.start_cycle(images, proprio, timestamp=0.0, initial_noise=initial_noise)
    original_context = controller.context.clone()
    original_gains = controller.gains.clone()

    with torch.no_grad():
        field = policy.film(policy.tactile(positions, forces), controller.gains)
        full = policy.diffusion.sample(
            policy.denoiser, initial_noise, controller.context, field,
            inference_steps=config.inference_steps,
        )

    # Raising from either module makes accidental use by online control visible.
    policy.predictor = _ForbiddenTrainingHead()
    policy.force_ema = _ForbiddenTrainingHead()
    returned_actions = []
    for step in range(16):
        newest = controller.step(positions[:, step], forces[:, step], timestamp=float(step))
        assert newest.shape == (1, action_dim)
        assert torch.isfinite(newest).all()
        assert not newest.requires_grad
        returned_actions.append(newest)
        # FP32 DDIM amplifies tiny prefix-dependent GEMM roundoff near terminal
        # alpha_bar=4.04e-5. An independent 26D run measured 1.22e-4 maximum
        # absolute drift; 2e-4 absolute/relative tolerances allow this numerical
        # accumulation while still detecting causal or noise-cache changes.
        torch.testing.assert_close(newest, full[:, step], atol=2e-4, rtol=2e-4)
        assert len(controller.forces) == len(returned_actions) == step + 1
        torch.testing.assert_close(controller.initial_noise, original_noise, rtol=0, atol=0)
        torch.testing.assert_close(initial_noise, original_noise, rtol=0, atol=0)
        torch.testing.assert_close(controller.context, original_context, rtol=0, atol=0)
        torch.testing.assert_close(controller.gains, original_gains, rtol=0, atol=0)

    old_positions = torch.stack(controller.positions, dim=1).clone()
    old_forces = torch.stack(controller.forces, dim=1).clone()
    old_timestamp = controller.last_timestamp
    with pytest.raises(RuntimeError, match="exhausted"):
        controller.step(positions[:, 0], forces[:, 0], timestamp=17.0)
    assert len(controller.positions) == len(controller.forces) == 16
    assert controller.last_timestamp == old_timestamp
    torch.testing.assert_close(torch.stack(controller.positions, dim=1), old_positions, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(controller.forces, dim=1), old_forces, rtol=0, atol=0)
    torch.testing.assert_close(controller.initial_noise, original_noise, rtol=0, atol=0)
    torch.testing.assert_close(controller.context, original_context, rtol=0, atol=0)
    torch.testing.assert_close(controller.gains, original_gains, rtol=0, atol=0)


class _ReadTimeCode(nn.Module):
    """Return input codes as vector carriers, without implementing target gather."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, forces):
        return forces.unsqueeze(-2).expand(-1, -1, -1, self.channels, -1)


def test_every_origin_and_horizon_targets_exactly_the_next_one_through_eight():
    policy = DexEquiTactPolicy(_small_model_config()).eval()
    policy.force_ema = _ReadTimeCode(policy.config.vector_channels)
    time_codes = torch.arange(24.0).reshape(1, 24, 1, 1).expand(1, 24, 5, 3)
    time_codes = time_codes.clone().requires_grad_()
    targets = policy.future_targets(time_codes[:, :16], time_codes[:, 16:])
    assert targets.shape == (1, 16, 8, 5, policy.config.vector_channels, 3)
    assert not targets.requires_grad
    # Direct specification of all 128 zero-based origin/horizon pairs.
    expected = torch.tensor([[origin + delta for delta in range(1, 9)] for origin in range(16)])
    expected = expected.float()[None, :, :, None, None, None].expand_as(targets)
    torch.testing.assert_close(targets, expected, rtol=0, atol=0)


def test_real_ema_teacher_cannot_read_force_after_its_target_time():
    torch.manual_seed(632)
    policy = DexEquiTactPolicy(_small_model_config()).eval()
    forces = torch.randn(1, 24, 5, 3)
    changed_forces = forces.clone()
    changed_forces[:, 7:] = 100 * changed_forces[:, 7:] + 13
    with torch.no_grad():
        before = policy.force_ema(forces)
        after = policy.force_ema(changed_forces)
        prefix_only = policy.force_ema(forces[:, :7])
    torch.testing.assert_close(after[:, :7], before[:, :7], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(prefix_only, before[:, :7], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(after[:, 7:], before[:, 7:])
