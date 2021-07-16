import torch

if __name__ == "__main__":
    A = torch.arange(20).reshape((5, 4))
    B = torch.ones((5, 4))
    print("(A+B)T == AT+BT:\n", (A + B).T == (A.T + B.T))
