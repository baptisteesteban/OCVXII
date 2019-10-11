import numpy as np

class Function:
    def __init__(self, dim, value, grad, hessian):
        self.dim = dim
        self.value = value
        self.grad = grad
        self.hessian = hessian

class Probleme:
    def __init__(self, f, A=None, b=None):
        if A is not None:
            if np.linalg.matrix_rank(A) >= f.dim:
                raise Exception("Dimension error: rg(A) >= f.dim")
        self.f = f
        self.A = A
        self.b = b
