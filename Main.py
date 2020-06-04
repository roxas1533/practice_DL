import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    # x = x.astype(np.float32)
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
    return 0.0001*x**4+0.003*x**3+0.0002*x**2+0.001*x+3


class Model:
    alpha = 0.001
    Neuron1 = 128
    Neuron2 = 128
    Wx = np.random.random((Neuron1, 2))
    Wy = np.random.random((Neuron2, Neuron1 + 1))
    Wz = np.random.random((1, Neuron2 + 1))

    def fit(self):
        for i in range(10000):
            for j in range(len(X)):
                self.learn(X[j], T[j])

    def learn(self, k, t):
        out = self.func(k)
        Q, z, Iz, y, Iy, = out[0], out[1], out[2], out[3], out[4]
        deltaQ = (out[0] - t)
        deltaWz = deltaQ * np.vstack((z, np.array([[1]])))

        self.Wz = self.Wz - self.alpha * deltaWz.T
        tempZ = np.delete(self.Wz.T, len(self.Wz) - 1, 0)
        deltaWy = np.dot(deltaQ * (tempZ * (z * (1 - z))), np.vstack((y, np.array([[1]]))).T)
        self.Wy = self.Wy - self.alpha * deltaWy
        # print(self.Wy,y)
        # print(self.Wy,deltaWy)
        tempX = np.delete(self.Wy.T, len(self.Wy) - 1, 0)
        # print(out[4])
        deltaWx = deltaQ * np.dot(np.dot(tempX, tempZ * (z * (1 - z))) * y * (1 - y), np.array([[k, 1]]))
        self.Wx = self.Wx - self.alpha * deltaWx
        return 1

    def func(self, inPut):
        Iy = np.dot(self.Wx, np.array([[inPut], [1]]))
        y = sigmoid(Iy)
        Iz = np.dot(self.Wy, np.vstack((y, np.array([[1]]))))
        z = sigmoid(Iz)
        Q = np.dot(self.Wz, np.vstack((z, np.array([[1]]))))
        return Q, z, Iz, y, Iy

    def out(self, x):
        ret = []
        for i in x:
            ret.append(self.func(i)[0][0])
        return ret


m = Model()
# np.random.seed(1)
X = np.random.rand(128) * 40 - 20
T = f(X)+np.random.randn(128)
x = np.linspace(-20, 20, 100)
m.fit()
plt.plot(X, T, 'o')
plt.plot(x, m.out(x))
print(m.Wz)
plt.plot(x, f(x))
plt.show()
