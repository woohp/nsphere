import math

import numpy as np
import torch
import torch.nn.functional as F

import nsphere
import nsphere.torch


def test_2d_numpy() -> None:
    """A super simple test in 2d coordinates where the exact results are simple"""
    original = np.array([1, 2], dtype=np.float32)
    nspherical = nsphere.to_spherical(original)
    cartesian = nsphere.to_cartesian(nspherical)

    np.testing.assert_array_almost_equal(nspherical, np.array([math.sqrt(5), math.atan2(2, 1)]))
    np.testing.assert_array_almost_equal(cartesian, original)


def test_roundtrip_numpy_float32() -> None:
    rng = torch.Generator()
    rng.manual_seed(100)

    # points in cartesian coordinates
    norms = torch.empty(100, 100, 1).uniform_(0.5, 3, generator=rng)
    original = (F.normalize(torch.randn(100, 100, 64, generator=rng), dim=-1) * norms).numpy()

    nspherical = nsphere.to_spherical(original)
    cartesian = nsphere.to_cartesian(nspherical)

    assert nspherical.dtype == original.dtype
    assert cartesian.dtype == original.dtype
    np.testing.assert_array_almost_equal(nspherical[..., :1], norms)
    np.testing.assert_array_almost_equal(cartesian, original, decimal=3)


def test_roundtrip_numpy_float64() -> None:
    rng = torch.Generator()
    rng.manual_seed(100)

    # points in cartesian coordinates
    norms = torch.empty(100, 100, 1).uniform_(0.5, 3, generator=rng)
    original = (F.normalize(torch.randn(100, 100, 64, generator=rng), dim=-1) * norms).numpy()

    nspherical = nsphere.to_spherical(original)
    cartesian = nsphere.to_cartesian(nspherical)

    assert nspherical.dtype == original.dtype
    assert cartesian.dtype == original.dtype
    np.testing.assert_array_almost_equal(cartesian, original, decimal=3)


def test_roundtrip_torch_float32() -> None:
    rng = torch.Generator()
    rng.manual_seed(100)

    norms = torch.empty(100, 100, 1).uniform_(0.5, 3, generator=rng)
    original = F.normalize(torch.randn(100, 100, 64), dim=-1) * norms

    nspherical = nsphere.torch.to_spherical(original)
    cartesian = nsphere.torch.to_cartesian(nspherical)

    assert nspherical.dtype == original.dtype
    assert cartesian.dtype == original.dtype
    np.testing.assert_array_almost_equal(cartesian, original, decimal=3)


def test_roundtrip_torch_float64() -> None:
    rng = torch.Generator()
    rng.manual_seed(100)

    norms = torch.empty(100, 100, 1).uniform_(0.5, 3, generator=rng)
    original = F.normalize(torch.randn(100, 100, 64), dim=-1) * norms

    nspherical = nsphere.torch.to_spherical(original)
    cartesian = nsphere.torch.to_cartesian(nspherical)

    assert nspherical.dtype == original.dtype
    assert cartesian.dtype == original.dtype
    np.testing.assert_array_almost_equal(cartesian, original, decimal=3)


def test_torch_autograd() -> None:
    """Tests that the gradient is computed correctly"""

    # We want to minimize the angular distance between 2 vectors,
    # and if the target vector is itself, then the gradient should be 0
    x = torch.nn.Parameter(torch.tensor([2.0, 1.0, 3.0, 4.0, 5.0]))  # in spherical coordinates
    cartesian = nsphere.torch.to_cartesian(x)
    target = cartesian.detach()
    angular_distance = 1 - F.normalize(cartesian, dim=-1) @ F.normalize(target, dim=-1)
    angular_distance.backward()
    assert x.grad is not None
    np.testing.assert_array_almost_equal(x.grad, 0)
