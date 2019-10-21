import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, P, pas, epsilon=0.01):
        self.P = P
        self.pas = pas
        self.epsilon = epsilon
        self.save = np.array([])

    def __call__(self, x0):
        x = x0
        self.save = []
        self.directions = []
        self.save.append(x)
        while np.linalg.norm(self.P.f.grad(x)) > self.epsilon:
            dx = -1 * self.P.f.grad(x)
            self.directions.append(dx)
            (t, _) = self.pas(self.P.f, x)
            x = x + t * dx
            self.save.append(x)
        self.save = np.array(self.save)
        self.directions = np.array(self.directions)
        return x

    def plot(self):
        if self.save.shape[0] == 0:
            raise Exception("The gradient descent algorithm did not run.")
        if self.P.f.dim == 1:
            plt.figure(figsize=(15, 15))
            x = np.linspace(self.save.min() - 5, self.save.max() + 5, 1000).reshape((1, -1))
            plt.plot(x.reshape((-1)), self.P.f.value(x))
            plt.scatter(self.save[:, 0], self.P.f.value(self.save[:, 0].reshape(1, -1)), 50, c="red")
            plt.grid()
            plt.show()
        elif self.P.f.dim == 2:
            plt.figure(figsize=(15, 15))
            x, y = np.linspace(self.save[:, 0].min() - 5, self.save[:, 0].max() + 5, 200), np.linspace(self.save[:, 1].min() - 5, self.save[:, 1].max() + 5, 200)
            X, Y = np.meshgrid(x, y)
            x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(2, -1)
            plt.contour(X, Y, self.P.f.value(x_y).reshape(200, -1), 15)
            plt.scatter(self.save[:, 0], self.save[:, 1], 50, c="red")
            plt.grid()
            plt.show()
        else:
            raise Exception("Plot in dim", self.P.f.dim, "not implemented")
