import torch

if __name__ == "__main__":
    X = torch.arange(12).reshape(3, 4)
    Y = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [1, 3, 5, 7]])

    print("X ==>\n", X, "\n")
    print("Y ==>\n", Y, "\n")
    print("X==Y ==>\n", X == Y, "\n")
    print("X<Y ==>\n", X < Y, "\n")
    print("X>Y ==>\n", X > Y, "\n")
    print("X!=Y ==>\n", X != Y, "\n")
