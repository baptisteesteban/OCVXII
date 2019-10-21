import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from ocvx import GradientDescent

from .probleme import Probleme, Function

def eqg_partial(f, x, i=0, dx=1e-6):
    """Computes i-th partial derivative of f at point x.
    
    Args:
        f: objective function.
        x: point at which partial derivative is computed.
        i: coordinate along which derivative is computed.
        dx: slack for finite difference.
        
    Output:
        (float)

    """
    h = np.zeros(f.dim)
    h[i] = dx
    return (f.value(x + h) - f.value(x - h)) / (2*dx)

def eqg_gradient(f, x, dx=1e-6):
    """Computes gradient of f at point x.
    
    Args:
        f: objective function.
        x: point at which gradient is computed.
        dx: slack for finite difference of partial derivatives.
        
    Output:
        (ndarray) of size domain of f.
        
    """
    dim = f.dim
    return np.array([eqg_partial(f, x, i, dx) for i in range(dim)])

def eqgrad_backtracking(f, x, alpha=0.1, beta=0.8):
    t = 1
    desc_d = -1 * eqg_gradient(f, x)
    while f.value(x + desc_d * t) > f.value(x) + alpha * t * np.dot(eqg_gradient(f, x).T, desc_d):
        t = beta * t
    return t, desc_d

class EQGradient:    
    def __init__(self, P, pas, epsilon=0.01):
        self.P_orig = P
        if self.P_orig.A is None:
            raise Exception("The argument P must be an equality constrained optimization problem")
        self.pas = pas
        self.epsilon = epsilon
        self.F = linalg.null_space(self.P_orig.A)
        self.save = np.array([])
        
    def __call__(self, x0):
        n_f_d = {
            "value": lambda z: self.P_orig.f.value(self.F.dot(z) + x0),
            "dim": self.P_orig.f.dim - self.P_orig.rank_A,
            "grad": lambda z: None,
            "hessian": lambda z: None
        }
        self.P = Probleme(Function(**n_f_d))
        
        z = np.zeros((self.P.f.dim))
        self.save = []
        self.save.append(self.F.dot(z) + x0)
        i = 0
        while np.linalg.norm(eqg_gradient(self.P.f, z)) > self.epsilon:
            dz = -1 * eqg_gradient(self.P.f, z)
            (t, _) = self.pas(self.P.f, z)
            z = z + t * dz
            self.save.append(self.F.dot(z) + x0)
        self.save = np.array(self.save)
        return self.F.dot(z) + x0

    def plot(self):
        if self.P_orig.f.dim != 2:
            raise Exception("Plot is only implemented for dim 2")
        if self.save.shape[0] == 0:
            raise Exception("The equality constrained Newton method did not run.")
        droite = lambda x: -1 * (self.P_orig.A[0, 0] / self.P_orig.A[0, 1]) * x + (self.P_orig.b[0] / self.P_orig.A[0, 1])
        plt.figure(figsize=(15, 15))
        x = np.linspace(self.save[:, 0].min() - 5, self.save[:, 0].max() + 5, 200)
        droite_y = droite(x)
        y = np.linspace(min(self.save[:, 1].min() - 5, droite_y.max()), max(self.save[:, 1].max() + 5, droite_y.max()), 200)
        X, Y = np.meshgrid(x, y)
        x_y = np.vstack([X.reshape(1, -1), Y.reshape(1, -1)]).reshape(2, -1)
        z = self.P_orig.f.value(x_y)
        plt.contour(X, Y, z.reshape(200, -1), 15)
        plt.scatter(self.save[:, 0], self.save[:, 1], 50, c="red")
        plt.plot(x, -1 * (self.P_orig.A[0, 0] / self.P_orig.A[0, 1]) * x + (self.P_orig.b[0] / self.P_orig.A[0, 1]))
        plt.grid()
        plt.show()
