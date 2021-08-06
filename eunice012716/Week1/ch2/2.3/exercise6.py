import torch

if __name__ == "__main__":
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    print("A:\n", A)
    print()
    print("A.sum(axis=1)\n", A.sum(axis=1))
    print()
    try:
        print("A / A.sum(axis=1):\n", A / A.sum(axis=1))
    except Exception as e:
        print("Error Message: ", e)
