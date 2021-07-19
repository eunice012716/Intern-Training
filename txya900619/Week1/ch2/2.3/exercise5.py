import torch

if __name__ == "__main__":
    X1 = torch.ones((2, 3, 4))
    X2 = torch.ones((4, 6, 2, 3))
    print("X1.shape: ", X1.shape)
    print("len(X1): ", len(X1))
    print()

    print("X2.shape", X2.shape)
    print("len(X2): ", len(X2))
    print()

    print("always first axis")
