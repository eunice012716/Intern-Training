# 5. For a tensor X of arbitrary shape, does len(X) always correspond to the length of a certain axis of X? What is that axis?

import torch

if __name__ == "__main__":
    X = torch.arange(24).reshape(2, 3, 4)
    print("len(X) ==>\n", len(X), "\n")

    print("len()取的是第一個dimension")
