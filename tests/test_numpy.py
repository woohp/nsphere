import math

import numpy as np
import pytest

import nsphere


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_known_2d_values(dtype: type[np.floating]) -> None:
    cartesian = np.array([1, 2], dtype=dtype)

    spherical = nsphere.to_spherical(cartesian)

    np.testing.assert_allclose(spherical, [math.sqrt(5), math.atan2(2, 1)], rtol=1e-6)
    np.testing.assert_allclose(nsphere.to_cartesian(spherical), cartesian, rtol=1e-6)
    assert spherical.dtype == cartesian.dtype


def test_known_3d_values() -> None:
    spherical = nsphere.to_spherical(np.array([1.0, 2.0, 3.0]))

    expected = [math.sqrt(14), math.acos(1 / math.sqrt(14)), math.atan2(3, 2)]
    np.testing.assert_allclose(spherical, expected)
    np.testing.assert_allclose(nsphere.to_cartesian(spherical), [1, 2, 3])


@pytest.mark.parametrize(
    ("point", "angle"),
    [
        ([1.0, 0.0], 0),
        ([0.0, 1.0], math.pi / 2),
        ([-1.0, 0.0], math.pi),
        ([0.0, -1.0], 3 * math.pi / 2),
    ],
)
def test_last_angle_covers_all_quadrants(point: list[float], angle: float) -> None:
    assert nsphere.to_spherical(np.array(point))[-1] == pytest.approx(angle)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_high_dimensional_roundtrip(dtype: type[np.floating]) -> None:
    rng = np.random.default_rng(100)
    original = rng.normal(size=(8, 7, 16)).astype(dtype)

    spherical = nsphere.to_spherical(original)
    actual = nsphere.to_cartesian(spherical)

    assert spherical.dtype == original.dtype
    assert actual.dtype == original.dtype
    np.testing.assert_allclose(actual, original, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(np.float32, 1e30), (np.float32, 1e-30), (np.float64, 1e300), (np.float64, 1e-300)],
)
def test_extreme_finite_magnitudes(dtype: type[np.floating], scale: float) -> None:
    cartesian = np.array([1, 2, 3], dtype=dtype) * dtype(scale)

    spherical = nsphere.to_spherical(cartesian)

    assert np.isfinite(spherical).all()
    assert spherical[0] != 0
    np.testing.assert_allclose(spherical[0] / dtype(scale), math.sqrt(14), rtol=1e-6)
    np.testing.assert_allclose(spherical[1:], [math.acos(1 / math.sqrt(14)), math.atan2(3, 2)], rtol=1e-6)


def test_zero_vector_has_canonical_representation() -> None:
    np.testing.assert_array_equal(nsphere.to_spherical(np.zeros(4)), np.zeros(4))


@pytest.mark.parametrize("coordinates", [np.array(1.0), np.array([]), np.array([1.0])])
def test_rejects_fewer_than_two_coordinates(coordinates: np.ndarray) -> None:
    with pytest.raises(ValueError, match="at least two"):
        nsphere.to_spherical(coordinates)

    with pytest.raises(ValueError, match="at least two"):
        nsphere.to_cartesian(coordinates)
