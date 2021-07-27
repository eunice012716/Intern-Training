# 9. Feed a tensor with 3 or more axes to the linalg.norm function and observe its output. What does this function compute for tensors of arbitrary shape?


import torch
import math

if __name__ == "__main__":
    Y = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    print("Y ==>\n", Y, "\n")
    print("torch.norm(Y) ==>\n", torch.norm(Y), "\n")

    i = 0
    for j in range(0, 24):
        i += j ** 2

    print(math.sqrt(i) == torch.norm(Y))
    print("torch.norm(variable): 對matrix中每個element平方加起來再開根號")
