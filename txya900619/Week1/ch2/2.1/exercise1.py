import torch

if __name__ == "__main__":
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

    # do == elementwise
    print("X == Y:\n", X == Y)
    print()

    # do < elementwise
    print("X < Y:\n", X < Y)
    print()

    # do > elementwise
    print("X > Y:\n", X > Y)
