import torch

if __name__ == "__main__":
    x = torch.arange(4.0)
    x.requires_grad_(True)
    y = 2 * torch.dot(x, x)
    y.backward()
    print(x.grad)
    try:
        y.backward()
    except Exception as e:
        print("[Error Message]", e)
