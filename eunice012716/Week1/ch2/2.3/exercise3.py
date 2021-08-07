import torch

if __name__ == "__main__":
    A = torch.arange(36).reshape((6, 6))
    print(A, "\n")
    print(A.T, "\n")
    print("A+AT: \n", A + A.T)
