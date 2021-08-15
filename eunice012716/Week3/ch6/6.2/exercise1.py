import torch

X = torch.tensor([[0.0, 1.0, 2.0], [0.0, 4.0, 0.0], [0.0, 7.0, 0.0]])
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
    print("Question 1:")
    print(corr2d(X, K))

    print("Question 2:")
    print(corr2d(X.t(), K))

    print("Question 3:")
    print(corr2d(X, K.t()))

    print("The results of Q2 & Q3 are transposes")
