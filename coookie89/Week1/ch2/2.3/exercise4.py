# 4. We defined the tensor X of shape (2, 3, 4) in this section. What is the output of len(X)?

import torch

if __name__ == "__main__":
    X = torch.arange(24).reshape(2, 3, 4)
    print("len(X) ==>\n", len(X), "\n")
