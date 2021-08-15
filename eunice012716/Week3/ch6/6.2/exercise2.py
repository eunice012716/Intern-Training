import torch

X = torch.tensor(
    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True
)
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])


def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


if __name__ == "__main__":
    try:
        result = corr2d(X, K)
        result.backward(
            torch.ones(result.shape[0], result.shape[1], dtype=torch.float)
        )
        print(X.grad)
    except Exception as e:
        print("Error Message:", e)

    print("我不知道Error是什麼ＱＡＱ")
