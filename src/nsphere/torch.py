"""Conversions between Cartesian and hyperspherical coordinates using PyTorch."""

import math

import torch
import torch.nn.functional as F


def to_cartesian(spherical: torch.Tensor) -> torch.Tensor:
    """Convert hyperspherical coordinates to Cartesian coordinates."""
    _validate_coordinates(spherical)
    radius = spherical[..., :1]
    angular = spherical[..., 1:]

    sin = F.pad(torch.sin(angular), [1, 0], value=1)
    cos = F.pad(torch.cos(angular), [0, 1], value=1)
    return radius * torch.cumprod(sin, dim=-1) * cos


def to_spherical(cartesian: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian coordinates to hyperspherical coordinates.

    The zero vector is represented by all zeros.
    """
    _validate_coordinates(cartesian)
    squares = torch.square(cartesian)
    radius = torch.sqrt(squares.sum(dim=-1, keepdim=True))

    tail_norms = torch.sqrt(torch.cumsum(squares.flip(-1), dim=-1).flip(-1))
    middle_angles = torch.atan2(tail_norms[..., 1:-1], cartesian[..., :-2])
    last_angle = torch.atan2(cartesian[..., -1:], cartesian[..., -2:-1])
    last_angle = torch.where(last_angle < 0, last_angle + 2 * math.pi, last_angle)
    return torch.cat((radius, middle_angles, last_angle), dim=-1)


def _validate_coordinates(coordinates: torch.Tensor) -> None:
    if coordinates.ndim == 0 or coordinates.shape[-1] < 2:
        raise ValueError("the final dimension must contain at least two coordinates")
