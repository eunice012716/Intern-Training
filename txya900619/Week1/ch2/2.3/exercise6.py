import torch

if __name__ == "__main__":
    A = torch.arange(20).reshape((5, 4))
    print("A/A.sum(axis=1):\n", A / A.sum(axis=1))
    print(
        "A and A.sum(axis=1) not match, because A.sum(axis=1) will reduce to shape (5) and when broadcasting to would be (1, 5), then (5, 4) not match to (1, 5)"
    )
