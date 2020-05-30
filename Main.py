import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    if x < 0:
        return 0
    return x


def gauss(x):
    mu = np.linspace(-30, 30, 6)
    list = []
    s = mu[1] - mu[0]
    for l in range(6):
        list.append(np.exp(-(x[l] - mu[l]) ** 2 / (2 * s ** 2)))

    return np.array(list)


def f(x):
    return -0.05 * x * x + 15


class Model:
    numNuro = 64
    alpha = 0.001
    Yw = np.random.rand(numNuro)
    Yb = np.random.rand(numNuro)
    Zw = np.random.rand(numNuro)
    b = np.random.rand()

    def fit(self):
        for i in range(20000):
            for j in range(len(X)):
                self.learn(X[j], T[j])

    def learn(self, k, t):
        Iy = self.Yw * k + self.Yb
        y = sigmoid(Iy)
        Iz = self.Zw * y
        z = np.sum(Iz) + self.b
        deltab = (z - t)
        self.b = self.b - self.alpha * deltab
        deltaZw = deltab * y
        self.Zw = self.Zw - self.alpha * deltaZw
        deltaY = deltab * self.Zw
        deltaYb = deltaY * y * (1 - y)
        self.Yb = self.Yb - self.alpha * deltaYb
        deltaYw = deltaY * k * y * (1 - y)
        self.Yw = self.Yw - self.alpha * deltaYw
        return 1

    def gosa(self):
        return np.mean((self.func(X) - T) * self.dens1(X)), np.mean(self.func(X) - T)

    def func(self, x):
        ret = []
        print(self.Zw, self.b)
        for i in x:
            Iy = self.Yw * i + self.Yb
            y = sigmoid(Iy)
            Iz = self.Zw * y
            ZZ = np.sum(Iz) + self.b
            ret.append(ZZ)
        return np.array(ret)


m = Model()
np.random.seed(1)
X = np.random.rand(128) * 60 - 30
T = f(X) + np.random.randn(128) * 2

x = np.linspace(-30, 30, 100)
m.fit()

plt.plot(X, T, 'o')
plt.plot(x, m.func(x))
plt.plot(x, f(x))

plt.show()
