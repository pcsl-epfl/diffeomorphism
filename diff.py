#pylint: disable=no-member, invalid-name, line-too-long
"""
Computes diffeomorphism of 2D images in pytorch
"""
import functools
import math

import torch


@functools.lru_cache()
def _scalar_field(n, m):
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    k = torch.arange(0, m, dtype=torch.float64)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m) / r
    e[0, :] = 0
    e[:, 0] = 0
    s = torch.sin(math.pi * x.reshape(n, 1) * k.reshape(1, m))
    return e, s


def scalar_field(n, cut):
    """
    random scalar field of size nxn made of the first m modes

    m = n * cut
    """
    m = round(n * cut)
    e, s = _scalar_field(n, m)
    c = torch.randn(m, m, dtype=torch.float64) * e
    return torch.einsum('ij,xi,yj->yx', c, s, s)


def deform(img, T, cut):
    """
    deform an image

    :param img Tensor: square image [batch, y, x]
    :param T float: temperature
    :param cut float: high frequency cutoff between 0 and 1
    """
    _, n, m = img.shape
    assert n == m

    u = scalar_field(n, cut)  # [n,n]
    v = scalar_field(n, cut)  # [n,n]
    dx = T**0.5 * u
    dy = T**0.5 * v
    img = remap(img, dx, dy)
    return img


def remap(a, dx, dy):
    """
    :param a: Tensor of shape [batch, y, x]
    :param dx: Tensor of shape [y, x]
    :param dy: Tensor of shape [y, x]
    """
    _, n, m = a.shape
    assert dx.shape == (n, m) and dy.shape == (n, m)
    y, x = torch.meshgrid(torch.arange(n).double(), torch.arange(m).double())
    x = (x + dx).clamp(0, m-1)
    y = (y + dy).clamp(0, n-1)

    xf = x.floor().long()
    yf = y.floor().long()
    xc = x.ceil().long()
    yc = y.ceil().long()

    def re(a, x, y):
        _, n, m = a.shape
        i = m * y + x
        i = i.flatten()
        return torch.index_select(a.reshape(-1, n * m), 1, i).reshape(-1, n, m)

    xv = x - xf
    yv = y - yf

    return (1-yv)*(1-xv)*re(a, xf, yf) + (1-yv)*xv*re(a, xc, yf) + yv*(1-xv)*re(a, xf, yc) + yv*xv*re(a, xc, yc)
