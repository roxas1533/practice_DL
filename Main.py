import matplotlib.pyplot as plt
import numpy as np


class Model:
    w = 10
    b = 160
    alpha = 0.0001
    hist = []

    def learn(self):
        for i in range(1000000):
            if max(np.absolute(self.gosa())) < 0.0001:
                break

    def gosa(self):
        sumw = (self.w * X + self.b - T) * X
        sumb = (self.w * X + self.b - T)
        self.w = self.w - self.alpha * np.mean(sumw)
        self.b = self.b - self.alpha * np.mean(sumb)
        self.hist.append([self.w, self.b])
        return np.mean(sumw), np.mean(sumb)

    def f(self, x):
        return self.w * x + self.b


m = Model()
np.random.seed(1)
X = np.random.rand(32) * 25 + 5
T = 3 * X + 1 * np.random.randn(32) * 8

x = np.linspace(5, 30, 100)
m.learn()


def f(x):
    return 3 * x + 1


plt.plot(X, T, 'o')
plt.plot(x, m.f(x))
plt.plot(x,f(x))
# for i in m.hist:
#     plt.plot(x, i[0] * x + i[1])

plt.show()
