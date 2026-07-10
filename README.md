# nsphere

Convert between Cartesian and hyperspherical coordinates with NumPy or PyTorch.

## Usage

```python
import numpy as np
import nsphere

cartesian = np.array([1.0, 2.0, 3.0, 4.0])
spherical = nsphere.to_spherical(cartesian)
restored = nsphere.to_cartesian(spherical)
```

```python
import torch
import nsphere.torch

cartesian = torch.tensor([1.0, 2.0, 3.0, 4.0])
spherical = nsphere.torch.to_spherical(cartesian)
restored = nsphere.torch.to_cartesian(spherical)
```

The functions operate on the final axis and preserve leading batch dimensions. Cartesian input is ordered
`(x1, ..., xn)`. Spherical input is ordered `(radius, phi1, ..., phi(n-1))`, following the conventional
[N-sphere definition](https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates). The first angles are in
`[0, pi]` and the final angle is in `[0, 2*pi)`. At least two coordinates are required. Because angles at the
origin are undefined, `to_spherical` uses all zeros as the canonical representation of the zero vector.

Inputs should use floating-point NumPy dtypes or PyTorch tensors. PyTorch conversions remain differentiable away
from the usual spherical-coordinate singularities.

## Development

```console
uv sync --dev
make lint
make typecheck
make test
make build
```
