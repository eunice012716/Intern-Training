# 3. Given any square matrix A, is A+A^T always symmetric? Why?

import torch

if __name__ == "__main__":
    A = torch.arange(9).reshape([3, 3])
    print("A+A.T==(A+A.T).T ==>\n", A + A.T == (A + A.T).T, "\n")
    print("yes, 因為transpose後相加的兩數都一樣")
