#pylint: disable=no-member, invalid-name, line-too-long
"""
Computes diffeomorphism of 2D images in pytorch
"""
import functools
import math

import torch

@functools.lru_cache()
def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = torch.linspace(0, 1, n, dtype=dtype, device=device)
    k = torch.arange(1, m + 1, dtype=dtype, device=device)
    i, j = torch.meshgrid(k, k, indexing='ij')
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5) / r
    s = torch.sin(math.pi * x[:, None] * k[None, :])
    return e, s


def scalar_field(n, m, B, device='cpu'):
    """
    random scalar field of size nxn made of the first m modes
    """
    e, s = scalar_field_modes(n, m, dtype=torch.get_default_dtype(), device=device)
    c = torch.randn(B, m, m, device=device) * e
    return torch.einsum('bij,xi,yj->byx', c, s, s)


def deform(image, T, cut, interp='linear', seed=None):
    """
    1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
    2. Apply tau to `image`
    :param img Tensor: square image(s) [B, :, y, x]
    :param T float: temperature
    :param cut int: high frequency cutoff
    """
    n = image.shape[-1]
    assert image.shape[-2] == n, 'Image(s) should be square.'
    
    device = image.device.type
    B = image.shape[0]
    
    if seed is not None:
        torch.manual_seed(seed)

    # Sample dx, dy
    # u, v are defined in [0, 1]^2
    # dx, dx are defined in [0, n]^2
    u = scalar_field(n, cut, B, device)  # [B,n,n]
    v = scalar_field(n, cut, B, device)  # [B,n,n]
    dx = T**0.5 * u * n
    dy = T**0.5 * v * n
    # Apply tau
    return remap(image, dx, dy, interp)


def remap(a, dx, dy, interp):
    """
    :param a: Tensor of shape [B, :, y, x]
    :param dx: Tensor of shape [B, y, x]
    :param dy: Tensor of shape [B, y, x]
    """
    n, m = a.shape[-2:]
    B = a.shape[0]

    dtype = dx.dtype
    device = dx.device.type
    
    y, x = torch.meshgrid(torch.arange(n, dtype=dtype, device=device), torch.arange(m, dtype=dtype, device=device), indexing='ij')

    xn = (x + dx).clamp(0, m-1)
    yn = (y + dy).clamp(0, n-1)

    if interp == 'linear':
        xf = xn.floor().long()
        yf = yn.floor().long()
        xc = xn.ceil().long()
        yc = yn.ceil().long()

        xv = (xn - xf).unsqueeze(1)
        yv = (yn - yf).unsqueeze(1)

        return (1-yv)*(1-xv)*a[torch.arange(B)[:, None, None], ..., yf, xf].permute(0, 3, 1, 2) + (1-yv)*xv*a[torch.arange(B)[:, None, None], ..., yf, xc].permute(0, 3, 1, 2) + yv*(1-xv)*a[torch.arange(B)[:, None, None], ..., yc, xf].permute(0, 3, 1, 2) + yv*xv*a[torch.arange(B)[:, None, None], ..., yc, xc].permute(0, 3, 1, 2)


def temperature_range(n, cut):
    """
    Define the range of allowed temperature
    for given image size and cut.
    """
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    T1 = 1 / (math.pi * n ** 2 * log)
    T2 = 4 / (math.pi**3 * cut ** 2 * log)
    return T1, T2


def typical_displacement(T, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return n * (math.pi * T * log) ** .5 / 2
