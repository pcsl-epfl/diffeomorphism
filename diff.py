import PIL
import PIL.Image
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import math
from tqdm.auto import tqdm
#import cv2
from torchvision import transforms
import functools

torch.set_default_dtype(torch.float64)


def triangle(x1, x2, y1, y2=None, slope=None, text="{}"):
    import math
    if y2 is None: y2 = y1 * (x2 / x1)**slope
    slope = math.log(y2 / y1) / math.log(x2 / x1)
    plt.plot([x1, x2, x2, x1], [y1, y2, y1, y1], 'k-')
    plt.annotate(text.format(slope), ((x1*x2**2)**(1/3), (y1**2*y2)**(1/3)), horizontalalignment='center', verticalalignment='center')


def load(path):
    img = PIL.Image.open(path)
    img = np.array(img)
    img = torch.from_numpy(img).double() / 255
    if img.dim() == 2:
        img = torch.stack([img] * 3, dim=-1)
    return img


def square(img):
    w, h, _ = img.shape
    assert _ == 3
    n = min(w, h)
    return img[w//2-n//2 : w//2-n//2 + n, h//2-n//2 : h//2-n//2 + n]


@functools.lru_cache()
def _scalar_field(n, m):
    x = torch.linspace(0, 1, n)
    k = torch.arange(0, m, dtype=x.dtype)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m) / r
    e[0, :] = 0
    e[:, 0] = 0
    s = torch.sin(math.pi * x.reshape(n, 1) * k.reshape(1, m))
    return e, s


def scalar_field(n, cut):
    m = round(n * cut)
    e, s = _scalar_field(n, m)
    c = torch.randn(m, m) * e
    return torch.einsum('ij,xi,yj->yx', c, s, s)


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


def deform(img, T, cut):
    n, m, _ = img.shape
    assert n == m

    u = scalar_field(n, cut)
    v = scalar_field(n, cut)
    dx = T**0.5 * u
    dy = T**0.5 * v
    img = img.transpose(0, 2)
    img = remap(img, dx, dy)
    img = img.transpose(0, 2)
    return img


def transpose(img):
    _ = {img.shape[-1], img.shape[-3]}
    assert 3 in _ and len(_) == 2, img.shape
    if img.shape[-3] == 3:
        return torch.einsum('...cyx->...yxc', img)
    else:
        return torch.einsum('...yxc->...cyx', img)


def mean(xs):
    return sum(xs) / len(xs)
