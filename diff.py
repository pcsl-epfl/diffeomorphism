#pylint: disable=no-member, invalid-name, line-too-long
"""
Computes diffeomorphism of 2D images in pytorch
"""
import functools
import math

import torch


@functools.lru_cache()
def scalar_field_modes(n, m):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    k = torch.arange(1, m + 1, dtype=torch.float64)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5) / r
    s = torch.sin(math.pi * x[:, None] * k[None, :])
    return e, s


def scalar_field(n, m):
    """
    random scalar field of size nxn made of the first m modes
    """
    e, s = scalar_field_modes(n, m)
    c = torch.randn(m, m, dtype=torch.float64) * e
    return torch.einsum('ij,xi,yj->yx', c, s, s)


def deform(image, T, cut):
    """
    1. Sample a displacement field xi: R2 -> R2, using tempertature `T` and cutoff `cut`
    2. Apply xi to `img`

    :param img Tensor: square image(s) [..., y, x]
    :param T float: temperature
    :param cut int: high frequency cutoff
    """
    n = image.shape[-1]
    assert image.shape[-2] == n
    
    # Sample xi = (dx, dy)
    u = scalar_field(n, cut)  # [n,n]
    v = scalar_field(n, cut)  # [n,n]
    dx = T**0.5 * u
    dy = T**0.5 * v
    
    # Apply xi
    return remap(image, dx, dy)


def remap(a, dx, dy):
    """
    :param a: Tensor of shape [..., y, x]
    :param dx: Tensor of shape [y, x]
    :param dy: Tensor of shape [y, x]
    """
    n, m = a.shape[-2:]
    
    assert dx.shape == (n, m) and dy.shape == (n, m)
    y, x = torch.meshgrid(torch.arange(n).double(), torch.arange(m).double())
    x = (x + dx).clamp(0, m-1)
    y = (y + dy).clamp(0, n-1)

    xf = x.floor().long()
    yf = y.floor().long()
    xc = x.ceil().long()
    yc = y.ceil().long()

    def re(x, y):
        i = m * y + x
        i = i.flatten()
        return torch.index_select(a.reshape(-1, n * m), 1, i).reshape(a.shape)

    xv = x - xf
    yv = y - yf

    return (1-yv)*(1-xv)*re(xf, yf) + (1-yv)*xv*re(xc, yf) + yv*(1-xv)*re(xf, yc) + yv*xv*re(xc, yc)
