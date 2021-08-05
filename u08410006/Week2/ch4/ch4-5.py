import torch
from torch import nn
from d2l import torch as d2l

def init_params():
    """
    randomly initialize our model parameters
    """
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    """
    square all terms in place and sum them up
    """
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    """
    Defining the Training Loop
    """
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting
            # makes `l2_penalty(w)` a vector whose length is `batch_size`
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())

def train_concise(wd):
    """
    specify the weight decay hyperparameter directly through weight_decay when instantiating our optimizer
    """
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([{
        "params": net[0].weight,
        'weight_decay': wd}, {
            "params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())

if __name__ == '__main__':    
    N_TRAIN, N_TEST, NUM_INPUTS, BATCH_SIZE = 20, 100, 200, 5
    true_w, true_b = torch.ones((NUM_INPUTS, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, N_TRAIN)
    train_iter = d2l.load_array(train_data, BATCH_SIZE)
    test_data = d2l.synthetic_data(true_w, true_b, N_TEST)
    test_iter = d2l.load_array(test_data, BATCH_SIZE, is_train=False)

    train(lambd=0)
    train(lambd=3)
    train_concise(0)
    train_concise(3)
