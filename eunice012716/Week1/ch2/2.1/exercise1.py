import torch

if __name__ == "__main__":
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

    print(x, "\n")
    print(y, "\n")
    print(x == y, "\n")
    print(x < y, "\n")
    print(x > y, "\n")
