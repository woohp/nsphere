"""Conversions between Cartesian and hyperspherical coordinates using NumPy."""

import math
from typing import Any

import numpy as np
import numpy.typing as npt


def to_cartesian[FloatingT: np.floating[Any]](
    spherical: npt.NDArray[FloatingT],
) -> npt.NDArray[FloatingT]:
    """Convert hyperspherical coordinates to Cartesian coordinates.

    The final axis contains the radius followed by ``n - 1`` angles.
    """
    _validate_coordinates(spherical)
    radius = spherical[..., :1]
    angular = spherical[..., 1:]
    padding = [(0, 0)] * (spherical.ndim - 1)

    sin = np.pad(np.sin(angular), [*padding, (1, 0)], constant_values=1)
    cos = np.pad(np.cos(angular), [*padding, (0, 1)], constant_values=1)
    return radius * np.cumprod(sin, axis=-1) * cos


def to_spherical[FloatingT: np.floating[Any]](
    cartesian: npt.NDArray[FloatingT],
) -> npt.NDArray[FloatingT]:
    """Convert Cartesian coordinates to hyperspherical coordinates.

    Angles except the last are in ``[0, pi]``. The last is in ``[0, 2*pi)``.
    The zero vector is represented by all zeros.
    """
    _validate_coordinates(cartesian)
    tail_norms = np.hypot.accumulate(cartesian[..., ::-1], axis=-1)[..., ::-1]
    radius = tail_norms[..., :1]
    middle_angles = np.arctan2(tail_norms[..., 1:-1], cartesian[..., :-2])
    last_angle = np.arctan2(cartesian[..., -1:], cartesian[..., -2:-1])
    last_angle = np.where(last_angle < 0, last_angle + 2 * math.pi, last_angle)
    return np.concatenate((radius, middle_angles, last_angle), axis=-1)


def _validate_coordinates(coordinates: npt.NDArray[Any]) -> None:
    if coordinates.ndim == 0 or coordinates.shape[-1] < 2:
        raise ValueError("the final dimension must contain at least two coordinates")
