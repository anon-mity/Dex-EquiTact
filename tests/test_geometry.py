"""Geometric and temporal contracts, independent of training/data plumbing."""

import importlib

import pytest
import torch


def geometry_classes():
    try:
        vn = importlib.import_module("dex_equitact.models.vn")
        tactile = importlib.import_module("dex_equitact.models.tactile")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Cannot import the geometry modules: {exc}")
    return vn.CausalVNEncoder, tactile.TypedTactileEncoder, tactile.ScalarFiLM, tactile.FutureForcePredictor


def rotation(seed):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q[:, -1] *= torch.linalg.det(q)
    return q


def samples():
    generator = torch.Generator().manual_seed(112)
    return tuple(torch.randn(2, 4, 5, 3, generator=generator, dtype=torch.float64) for _ in range(2))


def assert_close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)


def test_vector_encoder_rotates_output_and_keeps_all_fingers():
    CausalVNEncoder, _, _, _ = geometry_classes()
    torch.manual_seed(3)
    encoder = CausalVNEncoder(8, 2, 2).double().eval()
    positions, _ = samples()
    transformed = encoder(positions @ rotation(1).T)
    output = encoder(positions)
    assert output.shape == (2, 4, 5, 8, 3)
    assert_close(transformed, output @ rotation(1).T)
    assert output.std(dim=2).max() > 1e-3


def test_typed_encoder_obeys_independent_rotations():
    _, TypedTactileEncoder, _, _ = geometry_classes()
    torch.manual_seed(4)
    encoder = TypedTactileEncoder(8, 2, 2).double().eval()
    positions, forces = samples()
    output = encoder(positions, forces)
    assert output.shape == (2, 4, 5, 2, 8, 3)
    changed = encoder(positions @ rotation(4).T, forces @ rotation(8).T)
    assert_close(changed[..., 0, :, :], output[..., 0, :, :] @ rotation(4).T)
    assert_close(changed[..., 1, :, :], output[..., 1, :, :] @ rotation(8).T)
    assert encoder.position_encoder is not encoder.force_encoder


def test_future_input_cannot_change_earlier_tactile_carriers():
    _, TypedTactileEncoder, _, _ = geometry_classes()
    torch.manual_seed(5)
    encoder = TypedTactileEncoder(8, 2, 2).double().eval()
    positions, forces = samples()
    output = encoder(positions, forces)
    changed_positions, changed_forces = positions.clone(), forces.clone()
    changed_positions[:, 2:] = 30 * changed_positions[:, 2:] + 12
    changed_forces[:, 2:] = -50 * changed_forces[:, 2:] - 8
    changed = encoder(changed_positions, changed_forces)
    assert_close(changed[:, :2], output[:, :2])
    assert not torch.allclose(changed[:, 2:], output[:, 2:])


def test_truncating_prefix_preserves_each_available_output():
    _, TypedTactileEncoder, _, _ = geometry_classes()
    encoder = TypedTactileEncoder(8, 2, 2).double().eval()
    positions, forces = samples()
    output = encoder(positions, forces)
    for length in (1, 2, 3):
        assert_close(encoder(positions[:, :length], forces[:, :length]), output[:, :length])


def test_fixed_finger_identity_is_not_an_unordered_set():
    CausalVNEncoder, _, _, _ = geometry_classes()
    torch.manual_seed(14)
    encoder = CausalVNEncoder(8, 2, 2).double().eval()
    positions, _ = samples()
    permutation = [4, 1, 2, 3, 0]
    original = encoder(positions)
    swapped = encoder(positions[:, :, permutation])
    assert not torch.allclose(swapped, original)
    assert not torch.allclose(swapped, original[:, :, permutation], rtol=1e-5, atol=1e-6)


def test_film_uses_shared_per_finger_gains_without_coordinate_offsets():
    _, _, ScalarFiLM, _ = geometry_classes()
    film = ScalarFiLM(12, 8).double()
    tokens = torch.randn(2, 7, 12, dtype=torch.float64)
    field = torch.randn(2, 4, 5, 2, 8, 3, dtype=torch.float64)
    gains = film.gains(tokens)
    assert gains.shape == (2, 5, 8)
    assert_close(film(field, gains), field * (1 + gains[:, None, :, None, :, None]))
    assert_close(film(torch.zeros_like(field), gains), torch.zeros_like(field))


def test_film_respects_each_vector_types_rotation():
    _, _, ScalarFiLM, _ = geometry_classes()
    film = ScalarFiLM(12, 8).double()
    field = torch.randn(2, 4, 5, 2, 8, 3, dtype=torch.float64)
    gains = film.gains(torch.randn(2, 7, 12, dtype=torch.float64))
    rotated = field.clone()
    rotated[..., 0, :, :] = field[..., 0, :, :] @ rotation(10).T
    rotated[..., 1, :, :] = field[..., 1, :, :] @ rotation(11).T
    expected = film(field, gains)
    actual = film(rotated, gains)
    assert_close(actual[..., 0, :, :], expected[..., 0, :, :] @ rotation(10).T)
    assert_close(actual[..., 1, :, :], expected[..., 1, :, :] @ rotation(11).T)


def test_future_force_predictor_is_position_invariant_and_force_equivariant():
    _, _, _, FutureForcePredictor = geometry_classes()
    predictor = FutureForcePredictor(8, horizon=3).double()
    field = torch.randn(2, 4, 5, 2, 8, 3, dtype=torch.float64)
    output = predictor(field)
    assert output.shape == (2, 4, 3, 5, 8, 3)
    rotated = field.clone()
    rotated[..., 0, :, :] = field[..., 0, :, :] @ rotation(20).T
    rotated[..., 1, :, :] = field[..., 1, :, :] @ rotation(21).T
    assert_close(predictor(rotated), output @ rotation(21).T)
    position_only = field.clone()
    position_only[..., 0, :, :] = field[..., 0, :, :] @ rotation(20).T
    assert_close(predictor(position_only), output)


def test_predictor_gradients_reach_both_typed_streams_and_film():
    _, TypedTactileEncoder, ScalarFiLM, FutureForcePredictor = geometry_classes()
    encoder = TypedTactileEncoder(8, 1, 2).double()
    film = ScalarFiLM(12, 8).double()
    predictor = FutureForcePredictor(8, horizon=3).double()
    positions, forces = (x.requires_grad_() for x in samples())
    context = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    field = film(encoder(positions, forces), film.gains(context))
    loss = predictor(field).square().mean()
    loss.backward()
    for source in (positions, forces, context):
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert source.grad.abs().sum() > 0
    for module in (encoder.position_encoder, encoder.force_encoder, film, predictor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_zero_contact_has_finite_forward_and_backward():
    _, TypedTactileEncoder, ScalarFiLM, FutureForcePredictor = geometry_classes()
    encoder = TypedTactileEncoder(8, 1, 2).double()
    film = ScalarFiLM(12, 8).double()
    predictor = FutureForcePredictor(8).double()
    positions = torch.zeros(1, 2, 5, 3, dtype=torch.float64, requires_grad=True)
    forces = torch.zeros_like(positions, requires_grad=True)
    field = encoder(positions, forces)
    output = predictor(film(field, film.gains(torch.zeros(1, 2, 12, dtype=torch.float64))))
    assert torch.count_nonzero(output) == 0
    output.square().sum().backward()
    for source in (positions, forces):
        assert torch.isfinite(source.grad).all()


def test_encoder_rejects_invalid_finger_count_or_long_context():
    CausalVNEncoder, _, _, _ = geometry_classes()
    encoder = CausalVNEncoder(8, 1, 2, max_steps=3)
    with pytest.raises(ValueError, match="5"):
        encoder(torch.randn(1, 2, 4, 3))
    with pytest.raises(ValueError, match="max_steps"):
        encoder(torch.randn(1, 4, 5, 3))
