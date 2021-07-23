import torch
from IPython import display
from d2l import torch as d2l


def softmax(X):
    """
    softmax is the operation define in ch3.6.2
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here


def net(X):
    """
    net defines how the input is mapped to the output through the network.
    """
    return softmax(
        torch.matmul(X.reshape((-1, weight.shape[0])), weight) + biases
    )


def cross_entropy(y_hat, y):
    """
    cross-entropy takes the negative log-likelihood of the predicted probability assigned to the true label
    """
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """
    Compute the number of correct predictions.
    """
    length_gt_1 = len(y_hat.shape) > 1  # gt is greater than
    y_hat_shape_gt_1 = y_hat.shape[1] > 1

    if length_gt_1 and y_hat_shape_gt_1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """
    Compute the accuracy for a model on a dataset.
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    accum_num = 2
    metric = Accumulator(
        accum_num
    )  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """
    The training loop defined in Chapter 3.
    """
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    accum_num = 3
    metric = Accumulator(accum_num)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        lo = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            lo.backward()
            updater.step()
            metric.add(float(lo) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # Using custom built optimizer & loss criterion
            lo.sum().backward()
            updater(X.shape[0])
            metric.add(float(lo.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    Train a model (defined in Chapter 3).
    """
    animator = Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, "train_loss wrong"
    assert train_acc <= 1 and train_acc > 0.7, "train_acc wrong"
    assert test_acc <= 1 and test_acc > 0.7, "test_acc wrong    "


def updater(batch_size):
    LEARNING_RATE = 0.1
    return d2l.sgd([weight, biases], LEARNING_RATE, batch_size)


def predict_ch3(net, test_iter, n=6):
    """
    Predict labels (defined in Chapter 3).
    """
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


class Accumulator:
    """
    For accumulating sums over `var_num` variables.
    """

    def __init__(self, var_num):
        self.data = [0.0] * var_num  # var_num is the number of varialbes

    def add(self, *args):
        self.data = [
            num_correct_prediction + float(num_prediction)
            for num_correct_prediction, num_prediction in zip(self.data, args)
        ]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """
    For plotting data in animation.
    """

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


if __name__ == "__main__":
    BATCH_SIZE = 256
    NUM_INPUTS = 784
    NUM_OUTPUTS = 10
    NUM_EPOCHS = 10

    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)

    weight = torch.normal(
        0, 0.01, size=(NUM_INPUTS, NUM_OUTPUTS), requires_grad=True
    )
    biases = torch.zeros(NUM_OUTPUTS, requires_grad=True)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print(X_prob, X_prob.sum(1))

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat[[0, 1], y])

    print(cross_entropy(y_hat, y))

    print(accuracy(y_hat, y) / len(y))

    print(evaluate_accuracy(net, test_iter))

    train_ch3(net, train_iter, test_iter, cross_entropy, NUM_EPOCHS, updater)

    predict_ch3(net, test_iter)
