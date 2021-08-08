import torch

if __name__ == "__main__":
    a = torch.arange(12).reshape(2, 6, 1)
    b = torch.arange(36).reshape(2, 6, 3)

    print("a: \n", a, "\n")
    print("b: \n", b, "\n")

    print("a+b: \n", a + b)
