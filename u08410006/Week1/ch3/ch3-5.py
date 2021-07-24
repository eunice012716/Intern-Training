import os

import torch
import torchvision
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms


def get_fashion_mnist_labels(labels):
    """
    Return text labels for the Fashion-MNIST dataset.
    """
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(index)] for index in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles is not None:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers():
    """
    Use 4 processes to read the data.
    """
    cpu_count = os.cpu_count()
    return cpu_count


def load_data_fashion_mnist(batch_size, resize=None):
    """
    Download the Fashion-MNIST dataset and then load it into memory.
    """
    trans = [transforms.ToTensor()]
    if resize is not None:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(
            mnist_train,
            batch_size,
            shuffle=True,
            num_workers=get_dataloader_workers(),
        ),
        data.DataLoader(
            mnist_test,
            batch_size,
            shuffle=False,
            num_workers=get_dataloader_workers(),
        ),
    )


if __name__ == "__main__":

    BATCH_SIZE = 256

    d2l.use_svg_display()

    # `ToTensor` converts the image data from PIL type to 32-bit floating point
    # tensors. It divides all numbers by 255 so that all pixel values are between
    # 0 and 1
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )
    print(len(mnist_train), len(mnist_test))

    print(mnist_train[0][0].shape)
    X, y = next(iter(data.DataLoader(mnist_train, BATCH_SIZE=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    train_iter = data.DataLoader(
        mnist_train,
        BATCH_SIZE,
        shuffle=True,
        num_workers=get_dataloader_workers(),
    )

    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f"{timer.stop():.2f} sec")

    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
