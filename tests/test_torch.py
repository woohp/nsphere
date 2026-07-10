import math

import pytest
import torch

import nsphere.torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_known_2d_values(dtype: torch.dtype) -> None:
    cartesian = torch.tensor([1, 2], dtype=dtype)

    spherical = nsphere.torch.to_spherical(cartesian)

    torch.testing.assert_close(spherical, torch.tensor([math.sqrt(5), math.atan2(2, 1)], dtype=dtype))
    torch.testing.assert_close(nsphere.torch.to_cartesian(spherical), cartesian)
    assert spherical.dtype == cartesian.dtype
    assert spherical.device == cartesian.device


def test_known_3d_values() -> None:
    cartesian = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    spherical = nsphere.torch.to_spherical(cartesian)

    expected = torch.tensor([math.sqrt(14), math.acos(1 / math.sqrt(14)), math.atan2(3, 2)], dtype=torch.float64)
    torch.testing.assert_close(spherical, expected)
    torch.testing.assert_close(nsphere.torch.to_cartesian(spherical), cartesian)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_high_dimensional_roundtrip(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(100)
    original = torch.randn((8, 7, 16), dtype=dtype, generator=generator)

    spherical = nsphere.torch.to_spherical(original)
    actual = nsphere.torch.to_cartesian(spherical)

    assert spherical.dtype == original.dtype
    assert actual.dtype == original.dtype
    torch.testing.assert_close(actual, original, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(torch.float32, 1e30), (torch.float32, 1e-30), (torch.float64, 1e300), (torch.float64, 1e-300)],
)
def test_extreme_finite_magnitudes(dtype: torch.dtype, scale: float) -> None:
    cartesian = torch.tensor([1, 2, 3], dtype=dtype) * scale

    spherical = nsphere.torch.to_spherical(cartesian)

    assert torch.isfinite(spherical).all()
    assert spherical[0] != 0
    torch.testing.assert_close(spherical[0] / scale, torch.tensor(math.sqrt(14), dtype=dtype))
    expected_angles = torch.tensor([math.acos(1 / math.sqrt(14)), math.atan2(3, 2)], dtype=dtype)
    torch.testing.assert_close(spherical[1:], expected_angles)


def test_zero_vector_has_canonical_representation() -> None:
    torch.testing.assert_close(nsphere.torch.to_spherical(torch.zeros(4)), torch.zeros(4))


@pytest.mark.parametrize("shape", [(), (0,), (1,)])
def test_rejects_fewer_than_two_coordinates(shape: tuple[int, ...]) -> None:
    coordinates = torch.empty(shape)

    with pytest.raises(ValueError, match="at least two"):
        nsphere.torch.to_spherical(coordinates)

    with pytest.raises(ValueError, match="at least two"):
        nsphere.torch.to_cartesian(coordinates)


def test_to_cartesian_autograd() -> None:
    spherical = torch.tensor([2.0, 1.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(nsphere.torch.to_cartesian, (spherical,))


def test_to_spherical_autograd() -> None:
    cartesian = torch.tensor([2.0, 1.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(nsphere.torch.to_spherical, (cartesian,))
