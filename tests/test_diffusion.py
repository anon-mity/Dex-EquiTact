"""Behavioral checks for causal epsilon prediction and deterministic DDIM."""

from importlib import import_module
import math

import pytest
import torch
from torch import nn


def _diffusion():
    try:
        return import_module("dex_equitact.models.diffusion")
    except ModuleNotFoundError:
        pytest.fail("Cannot import dex_equitact.models.diffusion.")


def _example(action_dim=26, length=5):
    module = _diffusion()
    torch.manual_seed(314)
    model = module.ActionDenoiser(
        action_dim, channels=2, model_dim=24, layers=2, heads=4,
        max_steps=16, context_dim=12,
    )
    actions = torch.randn(2, length, action_dim)
    times = torch.tensor([2, 7], dtype=torch.long)
    slow = torch.randn(2, 3, 12)
    tactile = torch.randn(2, length, 5, 2, 2, 3)
    return model, actions, times, slow, tactile


def test_forward_noising_matches_three_step_hand_computation():
    schedule = _diffusion().DiffusionSchedule(
        train_steps=3, beta_start=0.1, beta_end=0.3,
    )
    actions = torch.tensor([[[2.0, -1.0]], [[-3.0, 4.0]]])
    noise = torch.tensor([[[0.5, 0.2]], [[-0.4, 0.7]]])
    times = torch.tensor([0, 2])
    actual = schedule.add_noise(actions, noise, times)
    expected = torch.stack([
        math.sqrt(0.9) * actions[0] + math.sqrt(0.1) * noise[0],
        math.sqrt(0.504) * actions[1] + math.sqrt(0.496) * noise[1],
    ])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(schedule.alpha_bar, torch.tensor([0.9, 0.72, 0.504]))
    assert not list(schedule.parameters())
    assert {"betas", "alpha_bar"}.issubset(schedule.state_dict())


class _KnownNoise(nn.Module):
    """Analytical noise oracle for checking the sampler's numerical update."""

    def __init__(self, fixed=None):
        super().__init__()
        self.fixed = fixed
        self.seen_times = []

    def forward(self, actions, timesteps, slow_tokens, tactile_field):
        self.seen_times.append(timesteps.clone())
        if self.fixed is not None:
            return torch.full_like(actions, self.fixed)
        return timesteps[:, None, None].to(actions) / 10.0 + torch.zeros_like(actions)


def test_ddim_two_step_update_matches_hand_computation_and_clean_terminal():
    schedule = _diffusion().DiffusionSchedule(3, beta_start=0.1, beta_end=0.3)
    oracle = _KnownNoise()
    initial = torch.tensor([[[1.5, -0.7]]])
    original = initial.clone()
    actual = schedule.sample(
        oracle, initial, torch.zeros(1, 1, 4), torch.zeros(1, 1, 5, 2, 2, 3),
        inference_steps=2,
    )
    # k=2 uses epsilon=.2 and alpha_bar=.504; next k=0 has alpha_bar=.9.
    # k=0 then predicts zero epsilon and returns its clean x0 at alpha_prev=1.
    expected = (initial - math.sqrt(0.496) * 0.2) / math.sqrt(0.504)
    expected = expected + math.sqrt(0.1 / 0.9) * 0.2
    torch.testing.assert_close(actual, expected)
    assert [int(t.item()) for t in oracle.seen_times] == [2, 0]
    torch.testing.assert_close(initial, original, rtol=0, atol=0)


def test_single_ddim_step_starts_at_final_training_noise_level():
    schedule = _diffusion().DiffusionSchedule(3, beta_start=0.1, beta_end=0.3)
    oracle = _KnownNoise(fixed=0.25)
    initial = torch.tensor([[[1.2]]])
    actual = schedule.sample(
        oracle, initial, torch.zeros(1, 1, 4), torch.zeros(1, 1, 5, 2, 2, 3),
        inference_steps=1,
    )
    expected = (initial - math.sqrt(0.496) * 0.25) / math.sqrt(0.504)
    torch.testing.assert_close(actual, expected)
    assert [int(t.item()) for t in oracle.seen_times] == [2]


@pytest.mark.parametrize("action_dim", [26, 28])
@pytest.mark.parametrize("training", [False, True])
def test_denoiser_prefix_matches_full_sequence(action_dim, training):
    model, actions, times, slow, tactile = _example(action_dim)
    model.train(training)
    with torch.no_grad():
        full = model(actions, times, slow, tactile)
        assert full.shape == actions.shape
        for length in (1, 3, 4):
            prefix = model(actions[:, :length], times, slow, tactile[:, :length])
            torch.testing.assert_close(prefix, full[:, :length], atol=2e-6, rtol=2e-5)


def test_future_noisy_actions_and_tactile_cannot_change_past_noise_predictions():
    model, actions, times, slow, tactile = _example()
    model.eval()
    with torch.no_grad():
        original = model(actions, times, slow, tactile)
        future_actions = actions.clone()
        future_actions[:, 3:] += 50 * torch.randn_like(future_actions[:, 3:])
        changed_actions = model(future_actions, times, slow, tactile)
        future_tactile = tactile.clone()
        future_tactile[:, 3:] += 50 * torch.randn_like(future_tactile[:, 3:])
        changed_touch = model(actions, times, slow, future_tactile)
    torch.testing.assert_close(changed_actions[:, :3], original[:, :3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(changed_touch[:, :3], original[:, :3], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(changed_actions[:, 3:], original[:, 3:])
    assert not torch.allclose(changed_touch[:, 3:], original[:, 3:])


def test_all_five_current_finger_tokens_are_visible_without_affecting_past():
    model, actions, times, slow, tactile = _example()
    model.eval()
    with torch.no_grad():
        original = model(actions, times, slow, tactile)
        for finger in range(5):
            changed = tactile.clone()
            changed[:, 2, finger] += 10 * torch.randn_like(changed[:, 2, finger])
            output = model(actions, times, slow, changed)
            torch.testing.assert_close(output[:, :2], original[:, :2], atol=1e-6, rtol=1e-5)
            assert not torch.allclose(output[:, 2], original[:, 2]), finger


def test_denoiser_uses_slow_context_and_backpropagates_to_shared_vector_field():
    model, actions, times, slow, tactile = _example()
    tactile.requires_grad_()
    slow.requires_grad_()
    original = model(actions, times, slow, tactile)
    changed_slow = model(actions, times, -slow, tactile)
    assert not torch.allclose(original, changed_slow)
    original.square().mean().backward()
    assert tactile.grad is not None and tactile.grad.abs().sum() > 0
    assert slow.grad is not None and slow.grad.abs().sum() > 0
    assert torch.isfinite(tactile.grad).all()


@pytest.mark.parametrize("action_dim", [26, 28])
def test_actual_denoiser_ddim_is_deterministic_and_prefix_consistent(action_dim):
    model, initial, _, slow, tactile = _example(action_dim, length=5)
    model.eval()
    schedule = _diffusion().DiffusionSchedule(train_steps=12)
    full = schedule.sample(model, initial, slow, tactile, inference_steps=4)
    repeated = schedule.sample(model, initial, slow, tactile, inference_steps=4)
    torch.testing.assert_close(repeated, full, rtol=0, atol=0)
    assert not full.requires_grad
    for length in (1, 2, 4):
        prefix = schedule.sample(
            model, initial[:, :length], slow, tactile[:, :length], inference_steps=4,
        )
        torch.testing.assert_close(prefix, full[:, :length], atol=2e-6, rtol=2e-5)


def test_invalid_diffusion_steps_and_misaligned_conditioning_are_rejected():
    module = _diffusion()
    schedule = module.DiffusionSchedule(train_steps=3)
    model, actions, times, slow, tactile = _example()
    with pytest.raises(ValueError, match="timesteps"):
        schedule.add_noise(actions, torch.zeros_like(actions), torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="inference_steps"):
        schedule.sample(model, actions, slow, tactile, inference_steps=4)
    with pytest.raises(ValueError, match="tactile"):
        model(actions, times, slow, tactile[:, :-1])
    with pytest.raises(ValueError, match="max_steps"):
        model(
            actions[:, :1].expand(-1, 17, -1), times, slow,
            tactile[:, :1].expand(-1, 17, -1, -1, -1, -1),
        )
