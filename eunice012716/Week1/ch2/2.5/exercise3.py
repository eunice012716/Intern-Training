import torch


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


if __name__ == "__main__":
    a = torch.randn(size=(1, 4), requires_grad=True)
    d = f(a)
    try:
        d.backward()
    except Exception as e:
        print("[Error Message]", e)

    d.backward(torch.Tensor([[1.0, 1.0, 1.0, 1.0]]))
    print(a.grad == d / a)
    print(
        "\nSince a is a vector or a matrix, not a scalar, ",
        "we should add the gradient attribute or it will go wrong",
    )
