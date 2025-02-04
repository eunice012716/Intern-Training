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
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


if __name__ == "__main__":
    layer = CenteredLayer()
    layer(torch.FloatTensor([1, 2, 3, 4, 5]))
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    Y.mean()
    linear = MyLinear(5, 3)
    linear.weight
    linear(torch.rand(2, 5))
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    net(torch.rand(2, 64))
