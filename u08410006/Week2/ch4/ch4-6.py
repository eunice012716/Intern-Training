import torch
from torch import nn
from d2l import torch as d2l

DROPOUT1, DROPOUT2 = 0.2, 0.5


def dropout_layer(X, dropout):
    """
    drops out the elements in the tensor input X with probability dropout,
    rescaling the remainder as described above:
    dividing the survivors by 1.0-dropout.
    """
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


def init_weights(m):
    """
    initial the weight
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


class Net(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hiddens1,
        num_hiddens2,
        is_training=True,
    ):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training is True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, DROPOUT1)
        H2 = self.relu(self.lin2(H1))
        if self.training is True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, DROPOUT2)
        out = self.lin3(H2)
        return out


if __name__ == "__main__":
    NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDENS1, NUM_HIDDENS2 = 784, 10, 256, 256
    NUM_EPOCHS, LR, BATCH_SIZE = 10, 0.5, 256

    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)
    print(dropout_layer(X, 0.0))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1.0))

    net = Net(NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDENS1, NUM_HIDDENS2)

    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(BATCH_SIZE)
    trainer = torch.optim.SGD(net.parameters(), lr=LR)
    d2l.train_ch3(net, train_iter, test_iter, loss, NUM_EPOCHS, trainer)

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(DROPOUT1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(DROPOUT2),
        nn.Linear(256, 10),
    )

    net.apply(init_weights)

    trainer = torch.optim.SGD(net.parameters(), lr=LR)
    d2l.train_ch3(net, train_iter, test_iter, loss, NUM_EPOCHS, trainer)
