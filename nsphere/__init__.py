from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=npt.NBitBase)


def to_cartesian(spherical: npt.NDArray[np.floating[T]]) -> npt.NDArray[np.floating[T]]:
    """Converts an array containing hyperspherical coordinates to cartesian coordinates"""
    radius = spherical[..., :1]
    angular = spherical[..., 1:]
    ndim = spherical.ndim

    sin = np.pad(np.sin(angular), [(0, 0)] * (ndim - 1) + [(1, 0)], constant_values=1)
    cos = np.pad(np.cos(angular), [(0, 0)] * (ndim - 1) + [(0, 1)], constant_values=1)
    return radius * np.cumprod(sin, axis=-1) * cos


def to_spherical(cartesian: npt.NDArray[np.floating[T]], eps: float = 1e-6) -> npt.NDArray[np.floating[T]]:
    """Converts an array containing cartesian coordinates to hyperspherical coordinates"""
    squares = np.square(cartesian)
    sum_squares = squares.sum(axis=-1, keepdims=True)
    r = np.sqrt(sum_squares)
    cumsum_squares = sum_squares - np.cumsum(
        np.pad(squares[..., :-2], [(0, 0)] * (cartesian.ndim - 1) + [(1, 0)], constant_values=0),
        axis=-1,
    )

    mids, last = np.split(
        np.arccos(cartesian[..., :-1] / np.sqrt(cumsum_squares + eps)), [cartesian.shape[-1] - 2], axis=-1
    )
    last = np.where(cartesian[..., -1:] >= 0, last, 2 * np.pi - last)
    return np.concatenate([r, mids, last], axis=-1)
