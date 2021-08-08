# 4.4. Model Selection, Underfitting, and Overfitting
# 4.4.4. Polynomial Regression 多項式回歸

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 4.4.4.2. Training and Testing the Model


def evaluate_loss(net, data_iter, loss_function):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        losses = loss_function(out, y)
        metric.add(losses.sum(), losses.numel())
    return metric[0] / metric[1]


def train(
    train_features,
    test_features,
    train_labels,
    test_labels,
    num_epochs=400,
    lr=0.01,
):
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array(
        (train_features, train_labels.reshape(-1, 1)), batch_size
    )
    test_iter = d2l.load_array(
        (test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False
    )
    trainer = torch.optim.SGD(net.parameters(), lr)
    animator = d2l.Animator(
        xlabel="epoch",
        ylabel="loss",
        yscale="log",
        xlim=[1, num_epochs],
        ylim=[1e-3, 1e2],
        legend=["train", "test"],
    )
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, nn.MSELoss(), trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(
                epoch + 1,
                (
                    evaluate_loss(net, train_iter, nn.MSELoss()),
                    evaluate_loss(net, test_iter, nn.MSELoss()),
                ),
            )
    print("weight:", net[0].weight.data.numpy())


if __name__ == "__main__":

    # 4.4.4.1. Generating the Dataset

    max_degree = 20  # Maximum degree of the polynomial
    data_size_train = 100  # Training dataset sizes
    data_size_test = 100  # test dataset sizes

    true_w = np.zeros(max_degree)  # Allocate lots of empty space
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(data_size_train + data_size_test, 1))
    np.random.shuffle(features)  # 把features順序打亂

    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))

    # make features rescale from x^i to x^i/i! to avoid very large values for large exponents i.
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!

    # polynomial_regression_model==> y=XB+ε
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # Convert from NumPy ndarrays to tensors
    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in [true_w, features, poly_features, labels]
    ]

    print("Third-Order Polynomial Function Fitting (Normal)==>\n")
    # Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the polynomial features
    train(
        poly_features[:data_size_train, :4],
        poly_features[data_size_train:, :4],
        labels[:data_size_train],
        labels[data_size_train:],
    )

    print("Linear Function Fitting (Underfitting)==>\n")
    # Pick the first two dimensions, i.e., 1, x, from the polynomial features
    train(
        poly_features[:data_size_train, :2],
        poly_features[data_size_train:, :2],
        labels[:data_size_train],
        labels[data_size_train:],
    )

    print("Higher-Order Polynomial Function Fitting (Overfitting)==>\n")
    # Pick all the dimensions from the polynomial features
    train(
        poly_features[:data_size_train, :],
        poly_features[data_size_train:, :],
        labels[:data_size_train],
        labels[data_size_train:],
        num_epochs=1500,
    )
