import math
import time

import torch
import numpy as np
from d2l import torch as d2l


def normal(x, mu, sigma):
    """
    the function of Normal Distribution, see ch3.1.3
    """
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


class Timer:
    """
    Record multiple running times.
    """

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """
        Start the timer.
        """
        self.tik = time.time()

    def stop(self):
        """
        Stop the timer and record the time in a list.
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """
        Return the average time.
        """
        return sum(self.times) / len(self.times)

    def sum(self):
        """
        Return the sum of time.
        """
        return sum(self.times)

    def cumsum(self):
        """
        Return the accumulated time.
        """
        return np.array(self.times).cumsum().tolist()


if __name__ == "__main__":
    NUM = 10000
    a = torch.ones(NUM)
    b = torch.ones(NUM)
    c = torch.zeros(NUM)
    timer = Timer()
    for i in range(NUM):
        c[i] = a[i] + b[i]
    print(f"{timer.stop():.5f} sec")

    timer.start()
    d = a + b
    print(f"{timer.stop():.5f} sec")

    # Use numpy again for visualization
    x = np.arange(-7, 7, 0.01)

    # Mean and standard deviation pairs
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(
        x,
        [normal(x, mu, sigma) for mu, sigma in params],
        xlabel="x",
        ylabel="p(x)",
        figsize=(4.5, 2.5),
        legend=[f"mean {mu}, std {sigma}" for mu, sigma in params],
    )
