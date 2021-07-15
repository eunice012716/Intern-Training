import torch


if __name__ == "__main__":
    a = torch.arange(15).reshape((3, 1, 5))
    b = torch.arange(30).reshape((3, 2, 5))

    print("a:\n", a)
    print()

    print("b:\n", b)
    print()

    # to trigger broadcasting mechanism, one of a or b is 1, and other axes size are equal
    print("a+b:\n", a + b)
