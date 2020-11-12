#pylint: disable=no-member, invalid-name
"""
basic operations on images
"""
import numpy as np
import PIL
import torch


def load(path):
    """
    from file to torch.Tensor [y, x, rgb]
    """
    img = PIL.Image.open(path)
    img = np.array(img)
    img = torch.from_numpy(img).double() / 255
    if img.dim() == 2:
        img = torch.stack([img] * 3, dim=-1)
    return img


def square(img):
    """
    make a square image [y, x, rgb]
    """
    w, h, _ = img.shape
    assert _ == 3
    n = min(w, h)
    return img[w//2-n//2 : w//2-n//2 + n, h//2-n//2 : h//2-n//2 + n]


def rgb_transpose(img):
    """
    from [..., y, x, rgb] to [..., y, x, rgb]
    """
    _ = {img.shape[-1], img.shape[-3]}
    assert 3 in _ and len(_) == 2, img.shape
    if img.shape[-3] == 3:
        return torch.einsum('...cyx->...yxc', img)
    else:
        return torch.einsum('...yxc->...cyx', img)
