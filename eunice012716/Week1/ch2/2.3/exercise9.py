import torch

if __name__ == "__main__":
    X = torch.ones((2, 3, 4))

    print("X.shape:\n", X.shape)
    print()

    print("torch.linalg.norm(X): ", torch.linalg.norm(X))
    print()

    print("torch.linalg.norm will return L2 for arbitrary shape tensor")
