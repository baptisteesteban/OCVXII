import numpy as np
import matplotlib.pyplot as plt

def constant(*args):
    return 0.01, 0

def backtracking(f, x, alpha=0.1, beta=0.8):
    t = 1
    desc_d = -1 * f.grad(x)
    while f.value(x + desc_d * t) > f.value(x) + alpha * t * np.dot(f.grad(x).T, desc_d):
        t = beta * t
    return t, desc_d

class Newton:
    def __init__(self, P, pas, epsilon=0.01):
        self.epsilon = epsilon
        self.P = P
        self.pas = pas
        self.save = np.array([])
        self.dirs = np.array([])
        
    def __call__(self, x0):
        self.save = []
        self.dirs = []
        x = x0
        self.save.append(x)
        dxN = -1 * np.dot(np.linalg.inv(self.P.f.hessian(x)), self.P.f.grad(x))
        lmd = -1 * np.dot(self.P.f.grad(x).T, dxN)
        while lmd / 2 > self.epsilon:
            dxN = -1 * np.dot(np.linalg.inv(self.P.f.hessian(x)), self.P.f.grad(x))
            lmd = -1 * np.dot(self.P.f.grad(x).T, dxN)
            (t, dire) = self.pas(self.P.f, x)
            self.dirs.append(dire)
            x = x + t * dxN
            self.save.append(x)
        self.save = np.array(self.save)
        self.dirs = np.array(self.dirs)
        return x
    
    def plot(self):
        if self.save.shape[0] == 0:
            raise Exception("The Newton method algorithm has not been run")
        if self.P.f.dim == 1:
            plt.figure(figsize=(15, 15))
            x = np.linspace(-1 * self.save.max() - 5, self.save.max() + 5, 1000).reshape((1, -1))
            plt.plot(x.reshape((-1)), self.P.f.value(x))
            plt.scatter(self.save[:, 0], self.P.f.value(self.save[:, 0].reshape(1, -1)), 50, c="red")
            plt.grid()
            plt.show()
        elif self.P.f.dim == 2:
            plt.figure(figsize=(15, 15))
            x, y = np.linspace(-1 * self.save[:, 0].max() - 5, self.save[:, 0].max() + 5, 200), np.linspace(- 1 * self.save[:, 1].max() - 5, self.save[:, 1].max() + 5, 200)
            X, Y = np.meshgrid(x, y)
            x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(2, -1)
            plt.contour(X, Y, self.P.f.value(x_y).reshape(200, -1), 15)
            plt.scatter(self.save[:, 0], self.save[:, 1], 50, c="red")
            plt.grid()
            plt.show()
        else:
            raise Exception("Dimension > 2 not implemented")
