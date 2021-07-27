import torch

if __name__ == "__main__":
    a = torch.arange(1, 9).reshape((2, 2, 2))
    b = torch.arange(1, 3).reshape((1, 2))

    print("a ==>\n", a, "\n")
    print("b ==>\n", b, "\n")
    print("a+b ==>\n", a + b, "\n")
    print("a-b ==>\n", a - b, "\n")
    print("a*b ==>\n", a * b, "\n")
    print("a/b ==>\n", a / b, "\n")
    print("a**b ==>\n", a ** b, "\n")
