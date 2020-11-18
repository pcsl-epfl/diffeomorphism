#pylint: disable=no-member, invalid-name, line-too-long
"""
unrelated functions
"""
import functools
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def triangle(x1, x2, y1, y2=None, slope=None, text="{}"):
    """
    draw a triangle
    """
    if y2 is None:
        y2 = y1 * (x2 / x1)**slope
    slope = math.log(y2 / y1) / math.log(x2 / x1)
    plt.plot([x1, x2, x2, x1], [y1, y2, y1, y1], 'k-')
    plt.annotate(text.format(slope), ((x1*x2**2)**(1/3), (y1**2*y2)**(1/3)), horizontalalignment='center', verticalalignment='center')


def mean(xs):
    """
    mean
    """
    return sum(xs) / len(xs)


def arbitrary_batch(dim):
    """
    allow a function to have an arbitrary batch shape
    """
    def decorator(f):
        @functools.wraps(f)
        def g(x, *args, **kwargs):
            size = x.shape
            x = x.reshape(-1, *size[-dim:])
            x = f(x, *args, **kwargs)
            x = x.reshape(*size[:-dim], *x.shape[1:])
            return x
        return g
    return decorator


def texnum(x, mfmt='{}', noone=False):
    """
    Convert number into latex
    """
    m, e = "{:e}".format(x).split('e')
    m, e = float(m), int(e)
    mx = mfmt.format(m)
    if e == 0:
        if m == 1:
            return "" if noone else "1"
        return mx
    ex = r"10^{{{}}}".format(e)
    if m == 1:
        return ex
    return r"{}\;{}".format(mx, ex)


@ticker.FuncFormatter
def format_percent(x, _pos=None):
    """
    usage
    plt.gca().yaxis.set_major_formatter(format_percent)
    """
    x = 100 * x
    if abs(x - round(x)) > 0.05:
        return r"${:.1f}\%$".format(x)
    return r"${:.0f}\%$".format(x)
