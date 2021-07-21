import torch

x = torch.arange(12)
print(x)
print()

print(x.shape)
print()

print(x.numel())
print()

X = x.reshape(3, 4)
print(X)
print()

print(torch.zeros((2, 3, 4)))
print()

print(torch.ones((2, 3, 4)))
print()

print(torch.randn(3, 4))
print()

print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
print()

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print()

print(torch.exp(x))
print()

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print()

print(X == Y)
print()

print(X.sum())
print()

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print()

print(a + b)
print()

print(X[-1])
print(X[1:3])
print()

X[1, 2] = 9
print(X)
print()

X[0:2, :] = 12
print(X)
print()

before = id(Y)
Y = Y + X
print(id(Y) == before)
print()

Z = torch.zeros_like(Y)
print("id(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))
print()

before = id(X)
X += Y
print(id(X) == before)
print()

A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))
print()

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
print()
