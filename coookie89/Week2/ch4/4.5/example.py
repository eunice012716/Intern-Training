# 4.5. Weight Decay
# 4.5.2. High-Dimensional Linear Regression

import torch
from torch import nn
from d2l import torch as d2l


def init_params():
    """Initializing Model Parameters"""
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    """L2 Norm Penalty"""
    return torch.sum(w.pow(2)) / 2


def train(lambd, num_epochs, lr):
    """training loop (lambd為weight decay的懲罰項)"""
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss

    animator = d2l.Animator(
        xlabel="epochs",
        ylabel="loss",
        yscale="log",
        xlim=[5, num_epochs],
        legend=["train", "test"],
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting makes `l2_penalty(w)` a vector whose length is `batch_size`
            losses = loss(net(X), y) + lambd * l2_penalty(w)
            losses.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,
                (
                    d2l.evaluate_loss(net, train_iter, loss),
                    d2l.evaluate_loss(net, test_iter, loss),
                ),
            )
    print("L2 norm of w:", torch.norm(w).item())


def train_concise(wd, num_epochs, lr):
    """Concise Implementation"""
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss_function = nn.MSELoss()

    # The bias parameter has not decayed
    trainer = torch.optim.SGD(
        [
            {"params": net[0].weight, "weight_decay": wd},
            {"params": net[0].bias},
        ],
        lr=lr,
    )
    animator = d2l.Animator(
        xlabel="epochs",
        ylabel="loss",
        yscale="log",
        xlim=[5, num_epochs],
        legend=["train", "test"],
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            losses = loss_function(net(X), y)
            losses.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,
                (
                    d2l.evaluate_loss(net, train_iter, loss_function),
                    d2l.evaluate_loss(net, test_iter, loss_function),
                ),
            )
    print("L2 norm of w:", net[0].weight.norm().item())


if __name__ == "__main__":
    data_size_train = 20
    data_size_test = 100
    num_inputs = 200
    batch_size = 5

    num_epochs = 100
    lr = 0.003

    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, data_size_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, data_size_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    print("Training with complecated step-----------------------\n")
    print("without Regularization==>\n")
    train(lambd=0)
    print("using Weight Decay==>\n")
    train(lambd=3)

    print("Training with concise implementation-----------------\n")
    print("without Regularization==>\n")
    train_concise(0, num_epochs, lr)
    print("using Weight Decay==>\n")
    train_concise(3, num_epochs, lr)
