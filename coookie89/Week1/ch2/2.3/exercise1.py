# 1. Prove that the transpose of a matrix A's transpose is A: (A^T)^T==A

import torch

if __name__ == "__main__":
    A = torch.arange(12).reshape([3, 4])
    print("A.T.T==A ==>\n", A.T.T == A, "\n")
