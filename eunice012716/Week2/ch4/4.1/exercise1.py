import torch

if __name__ == "__main__":
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.relu(x)
    y.backward(torch.ones_like(x))
    print(x.grad)
