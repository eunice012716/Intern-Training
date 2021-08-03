# 5.4. Custom Layers

import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        """
        in_units: the number of inputs
        units: the number of outputs
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


if __name__ == "__main__":

    # Layers without Parameters
    print("Layers without Parameters------------------------------")
    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())
    print()

    # Layers with Parameters
    print("Layers with Parameters---------------------------------")
    linear = MyLinear(5, 3)
    print(linear.weight)
    print(linear(torch.rand(2, 5)))  # forward propagation

    net = nn.Sequential(
        MyLinear(64, 8), MyLinear(8, 1)
    )  # construct models using custom layers
    print(net(torch.rand(2, 64)))
