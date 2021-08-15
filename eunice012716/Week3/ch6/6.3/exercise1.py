import torch
from torch import nn


def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])


def calculate_the_shape(
    X, xh: int, xw: int, kh: int, kw: int, ph: int, pw: int, sh: int, sw: int
):
    width_after_padding = X.shape[1] + pw * 2
    height_after_padding = X.shape[0] + ph * 2
    cal_output_width = 0
    cal_output_height = 0

    for w_pt in range(1, width_after_padding + 1, sw):
        if w_pt + kw <= width_after_padding:
            cal_output_width += 1
        else:
            break

    for h_pt in range(1, height_after_padding + 1, sh):
        if h_pt + kh <= height_after_padding:
            cal_output_height += 1
        else:
            break

    return cal_output_height, cal_output_width


if __name__ == "__main__":
    X = torch.rand(size=(8, 8))
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    cal_height, cal_width = calculate_the_shape(
        X, X.shape[0], X.shape[1], 3, 5, 0, 1, 3, 4
    )
    comp_height = comp_conv2d(conv2d, X).shape[0]
    comp_width = comp_conv2d(conv2d, X).shape[1]
    print(cal_height == comp_height and cal_width == comp_width)
