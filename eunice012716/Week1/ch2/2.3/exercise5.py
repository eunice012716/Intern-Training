import torch

if __name__ == "__main__":
    X1 = torch.ones((3, 4))
    X2 = torch.ones((2, 6, 8, 1))
    X3 = torch.ones((7, 8, 9))
    print("X1.shape:", X1.shape)
    print("len(X1):", len(X1))
    print()

    print("X2.shape", X2.shape)
    print("len(X2): ", len(X2))
    print()

    print("X3.shape", X3.shape)
    print("len(X3): ", len(X3))
    print()

    print("we find that it is always the size of first axis")
