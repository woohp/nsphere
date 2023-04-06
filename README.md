# nsphere

Conversion between cartesian and hyperspherical coordinates. Implementations are provided for numpy and pytorch.

See [this wikipedia page](https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates) for more information.

## Usage

numpy:

```python
import nsphere
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0])
spherical = nsphere.to_spherical(x)
```

pytorch:

```python
import nsphere.torch
import torch
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
spherical = nsphere.torch.to_spherical(x)
```
