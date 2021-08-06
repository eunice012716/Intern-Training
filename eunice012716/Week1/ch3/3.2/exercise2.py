import random
import torch

LEARNING_RATE = 0.03
BATCH_SIZE = 10
NUM_EPOCHS = 10


def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """shuffle the dataset and access it in minibatches."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)]
        )
        print(batch_indices)
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == "__main__":
    true_w = torch.Tensor([3.3])
    true_b = 0
    currents, voltages = synthetic_data(true_w, true_b, 1000)

    w = torch.normal(0, 0.01, size=(1,), requires_grad=True)
    b = torch.normal(0, 0.01, size=(1,), requires_grad=True)
    net = linreg
    loss = squared_loss

    for epoch in range(1, NUM_EPOCHS + 1):
        for X, y in data_iter(BATCH_SIZE, currents, voltages):
            loss_ = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
            # Compute gradient on `l` with respect to [`w`, `b`]
            loss_.sum().backward()
            sgd(
                [w, b], LEARNING_RATE, BATCH_SIZE
            )  # Update parameters using their gradient
        with torch.no_grad():
            train_l = loss(net(currents, w, b), voltages)
            print(f"epoch {epoch}, loss {float(train_l.mean()):f}")

    print("w: ", w)
    print("b: ", b)
    print("true_w - w:", true_w - w)
    print("true_b - b", true_b - b)
