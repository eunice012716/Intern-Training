import torch

if __name__ == "__main__":
    A = torch.arange(25).reshape((5, 5))
    print("A+AT:\n", A + A.T)
