# 8. Consider a tensor with shape (2, 3, 4). What are the shapes of the summation outputs along axis 0, 1, and 2?

import torch

if __name__ == "__main__":
    X = torch.arange(24).reshape(2, 3, 4)

    print("X==>\n", X, "\n")
    print("X.sum(axis=0)==>\n", X.sum(axis=0), "\n")
    print("X.sum(axis=1)==>\n", X.sum(axis=1), "\n")
    print("X.sum(axis=2)==>\n", X.sum(axis=2), "\n")
