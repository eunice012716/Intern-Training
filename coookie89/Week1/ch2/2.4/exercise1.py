# 1. Plot the function y=f(x)=x^3-1/x and its tangent line when x=1.

import numpy as np

from IPython import display
from d2l import torch as d2l


def function(x):  # 題目給的f(x)函式
    return x ** 3 - (1 / x)


def use_svg_display():  # 顯示圖
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):  # 設定圖大小
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, legend):  # 設定圖要顯示的東西
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    # axes.set_xscale(xscale)
    # axes.set_yscale(yscale)
    # axes.set_xlim(xlim)  # 設定x軸範圍
    # axes.set_ylim(ylim)  # 設定y軸範圍
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):  # 畫出數據(X,Y)的點
    """Plot data points."""
    if legend is None:
        legend = []

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

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()

    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, legend)


if __name__ == "__main__":
    x = np.arange(0.1, 3, 0.1)
    plot(
        x,
        [function(x), 4 * x - 4],
        "x",
        "f(x)",
        legend=["f(x)", "tangent line (x=1)"],
    )
