import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

TRUE_W = torch.tensor([2, -3.4])
TRUE_B = 4.2
FEATURES, LABELS = d2l.synthetic_data(TRUE_W, TRUE_B, 1000)
BATCH_SIZE = 10


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    data_iter = load_array((FEATURES, LABELS), BATCH_SIZE)
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            loss_ = loss(net(X), y)
            trainer.zero_grad()
            loss_.backward()
            trainer.step()
        loss_ = loss(net(FEATURES), LABELS)

    print(net[0].weight.grad)
