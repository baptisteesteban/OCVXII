import numpy as np
import matplotlib.pyplot as plt

class EQNewton:
    def __init__(self, P, pas, epsilon=0.01):
        if P.A is None:
            raise Exception("Probleme should have constraints")
        self.epsilon = epsilon
        self.P = P
        self.pas = pas
        self.save = np.array([])
        
    def _newtonStep(self, x):
        KKT = np.vstack([np.hstack([self.P.f.hessian(x), self.P.A.T]), np.hstack([self.P.A, np.zeros((self.P.A.T.shape[1], self.P.A.T.shape[1])).astype(self.P.A.dtype)])])
        g = self.P.f.grad(x)
        res = np.hstack([-1 * g, np.zeros((KKT.shape[0] - g.shape[0]))])
        x_out = np.linalg.solve(KKT, res)
        return x_out[:self.P.f.dim]
        
    def __call__(self, x0):
        self.save = []
        
        x = x0
        self.save.append(x)
        dxN = self._newtonStep(x)
        lmd = -1 * np.dot(self.P.f.grad(x), dxN)
        while lmd / 2 > self.epsilon:
            dxN = self._newtonStep(x)
            lmd = -1 * np.dot(self.P.f.grad(x), dxN)
            (t, _) = self.pas(self.P.f, x)
            x = x + t * dxN
            self.save.append(x)
        
        self.save = np.array(self.save)
        return x

    def plot(self):
        if self.P.f.dim != 2:
            raise Exception("Plot is only implemented for dim 2")
        plt.figure(figsize=(15, 15))
        x, y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(2, -1)
        z = self.P.f.value(x_y)
        plt.contour(X, Y, z.reshape(100, -1), 15)
        plt.scatter(self.save[:, 0], self.save[:, 1], 50, c="red")
        x_p = np.linspace(-5, 5, 100)
        plt.plot(x_p, -1 * (self.P.A[0, 0] / self.P.A[0, 1]) * x_p + (self.P.b[0] / self.P.A[0, 1]))
        plt.grid()
        plt.show()
