import torch
import torch.nn.functional as F


def to_cartesian(spherical: torch.Tensor) -> torch.Tensor:
    """Converts a tensor containing hyperspherical coordinates to cartesian coordinates"""
    radius = spherical[..., :1]
    angular = spherical[..., 1:]

    sin = F.pad(torch.sin(angular), [1, 0], value=1)
    cos = F.pad(torch.cos(angular), [0, 1], value=1)
    return radius * torch.cumprod(sin, dim=-1) * cos


def to_spherical(cartesian: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Converts a tensor containing cartesian coordinates to hyperspherical coordinates"""
    squares = torch.square(cartesian)
    sum_squares = squares.sum(dim=-1, keepdim=True)
    r = torch.sqrt(sum_squares)
    cumsum_squares = sum_squares - F.pad(squares[..., :-2], [1, 0], value=0).cumsum(dim=-1)

    mids, last = torch.split(
        torch.acos(cartesian[..., :-1] * torch.rsqrt(cumsum_squares + eps)), cartesian.shape[-1] - 2, dim=-1
    )
    last = torch.where(cartesian[..., -1:] >= 0, last, 2 * torch.pi - last)
    return torch.cat([r, mids, last], dim=-1)
