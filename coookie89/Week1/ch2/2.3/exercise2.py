# 2. Given two matrices A and B, show that the sum of transposes is equal to the transpose of a sum: A^T+B^T=(A+B)^T

import torch

if __name__ == "__main__":
    A = torch.arange(12).reshape([3, 4])
    B = torch.arange(5, 17).reshape([3, 4])
    print("(A+B).T==A.T+B.T ==>\n", (A + B).T == A.T + B.T, "\n")
