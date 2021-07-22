import random

import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    """
    Generate y = Xw + b + noise.
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """
    demonstrate one possible implementation of this functionality
    The function takes a batch size, a matrix of features,
    and a vector of labels, yielding minibatches of the size batch_size.
    Each minibatch consists of a tuple of features and labels.
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """
    The linear regression model.
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    Squared loss.
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    Minibatch stochastic gradient descent.
    """
    with torch.no_grad():
        for param in params:
            param -= (lr * param.grad) / batch_size
            param.grad.zero_()


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print("features:", features[0], "\nlabel:", labels[0])

    d2l.set_figsize()
    # The semicolon is for displaying the plot only
    d2l.plt.scatter(
        features[:, (1)].detach().numpy(), labels.detach().numpy(), 1
    )

    BATCH_SIZE = 10

    for X, y in data_iter(BATCH_SIZE, features, labels):
        print(X, "\n", y)

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    LR = 0.03
    NUM_EPOCHS = 3
    net = linreg
    calculate_loss = squared_loss

    for epoch in range(NUM_EPOCHS):
        for X, y in data_iter(BATCH_SIZE, features, labels):
            loss = calculate_loss(
                net(X, w, b), y
            )  # Minibatch loss in `X` and `y`
            # Compute gradient on `lo` with respect to [`w`, `b`]
            loss.sum().backward()
            sgd(
                [w, b], LR, BATCH_SIZE
            )  # Update parameters using their gradient
        with torch.no_grad():
            train_l = calculate_loss(net(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

    print(f"error in estimating w: {true_w - w.reshape(true_w.shape)}")
    print(f"error in estimating b: {true_b - b}")
