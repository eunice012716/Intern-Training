import torch

if __name__ == "__main__":
    A = torch.arange(20).reshape((5, 4))
    print("A == ATT:\n", A == A.T.T)
