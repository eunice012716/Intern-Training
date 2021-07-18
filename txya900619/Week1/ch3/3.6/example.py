import torch
from d2l import torch as d2l


def softmax(X: torch.Tensor):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
print(next(iter(train_iter)))
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
