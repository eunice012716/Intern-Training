from typing import List, Optional
from dataclasses import dataclass

import numpy as np

from IPython import display
from d2l import torch as d2l


@dataclass
class PlotConfig:
    X: float
    Y: Optional[float]
    legend: Optional[List]


def function_(x):
    """
    example for this topic caculus
    f(x) = 3 * x^2 - 4 * x
    """
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    """
    numerical result ( do Differentiation )
    f is what function used
    x is coordinate of x
    h is like dx
    """
    return (f(x + h) - f(x)) / h


def use_svg_display():
    """
    Use the svg format to display a plot in Jupyter.
    """
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set the figure size for matplotlib.
    """
    use_svg_display()
    d2l.plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    Set the axes for matplotlib.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    config: PlotConfig,
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """
    Plot data points.
    return axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend for set_axes()
    """
    if config.legend is None:
        config.legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (
            hasattr(X, "ndim")
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], "__len__")
        )

    if has_one_axis(config.X):
        X = [config.X]
    if config.Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(config.Y):
        Y = [config.Y]
    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla()

    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)


if __name__ == "__main__":
    x = np.arange(0, 3, 0.1)
    h = 0.1  # h is like dx

    plot_config = PlotConfig(
        X=x,
        Y=[function_(x), 2 * x - 3],
        legend=["function_(x)", "Tangent line (x=1)"],
    )

    plot(config=plot_config)

    for _ in range(5):
        print(
            f"h={h:.5f}, numerical limit={numerical_lim(function_, 1, h):.5f}"
        )
        h *= 0.1
