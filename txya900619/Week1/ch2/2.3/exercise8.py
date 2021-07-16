import torch

if __name__ == "__main__":
    X = torch.ones((2, 3, 4))

    print("X.sum(axis=0).shape: ", X.sum(axis=0).shape)
    print()

    print("X.sum(axis=1).shape: ", X.sum(axis=1).shape)
    print()

    print("X.sum(axis=2).shape: ", X.sum(axis=2).shape)
    print()
