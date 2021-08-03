# 5.5. File I/O

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


if __name__ == "__main__":

    x = torch.arange(4)
    torch.save(x, "x-file")

    print("Loading and Saving Tensors-------------------------------")

    # read the data from the stored file back into memory.
    x2 = torch.load("x-file")
    print(x2)
    print()

    # store a list of tensors and read them back into memory.
    y = torch.zeros(4)
    torch.save([x, y], "x-files")
    x2, y2 = torch.load("x-files")
    print((x2, y2))
    print()

    # write and read a dictionary that maps from strings to tensors
    mydict = {"x": x, "y": y}
    torch.save(mydict, "mydict")
    mydict2 = torch.load("mydict")
    print(mydict2)
    print()

    print("Loading and Saving Model Parameters----------------------")

    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)

    # store the parameters of the model as a file with the name “mlp.params”
    torch.save(net.state_dict(), "mlp.params")

    # instantiate a clone of the original MLP model
    clone = MLP()
    clone.load_state_dict(torch.load("mlp.params"))
    print(clone.eval())
    print()

    Y_clone = clone(X)
    print(Y_clone == Y)
