import torch

if __name__ == "__main__":
    X = torch.ones((2, 3, 4))
    print("X.shape:", X.shape)
    print("len(X):", len(X), "[the first axis size]")
