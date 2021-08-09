import math

import torch
import numpy as np
from torch import nn
from d2l import torch as d2l


def evaluate_loss(net, data_iter, loss_function):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        eval_loss = loss_function(out, y)
        metric.add(eval_loss.sum(), eval_loss.numel())
    return metric[0] / metric[1]


def train(
    train_features, test_features, train_labels, test_labels, num_epochs=400
):
    """
    define training function
    """
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array(
        (train_features, train_labels.reshape(-1, 1)), batch_size
    )
    test_iter = d2l.load_array(
        (test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False
    )
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(
        xlabel="epoch",
        ylabel="loss",
        yscale="log",
        xlim=[1, num_epochs],
        ylim=[1e-3, 1e2],
        legend=["train", "test"],
    )
    for epoch in range(1, num_epochs + 1):
        d2l.train_epoch_ch3(net, train_iter, nn.MSELoss, trainer)
        if epoch == 0 or epoch % 20 == 0:
            animator.add(
                epoch,
                (
                    evaluate_loss(net, train_iter, nn.MSELoss),
                    evaluate_loss(net, test_iter, nn.MSELoss),
                ),
            )
    print("weight:", net[0].weight.data.numpy())


if __name__ == "__main__":
    MAX_DEGREE = 20  # Maximum degree of the polynomial
    N_TRAIN, N_TEST = 100, 100  # Training and test dataset sizes

    true_w = np.zeros(MAX_DEGREE)  # Allocate lots of empty space
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(N_TRAIN + N_TEST, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(MAX_DEGREE).reshape(1, -1))
    for i in range(MAX_DEGREE):
        poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
    # Shape of `labels`: (`N_TRAIN` + `N_TEST`,)
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # Convert from NumPy ndarrays to tensors
    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in [true_w, features, poly_features, labels]
    ]

    features[:2], poly_features[:2, :], labels[:2]

    # Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
    # polynomial features
    train(
        poly_features[:N_TRAIN, :4],
        poly_features[N_TRAIN:, :4],
        labels[:N_TRAIN],
        labels[N_TRAIN:],
    )

    # Pick the first two dimensions, i.e., 1, x, from the polynomial features
    train(
        poly_features[:N_TRAIN, :2],
        poly_features[N_TRAIN:, :2],
        labels[:N_TRAIN],
        labels[N_TRAIN:],
    )

    # Pick all the dimensions from the polynomial features
    train(
        poly_features[:N_TRAIN, :],
        poly_features[N_TRAIN:, :],
        labels[:N_TRAIN],
        labels[N_TRAIN:],
        num_epochs=1500,
    )
