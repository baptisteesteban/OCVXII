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
            self.rank_A = np.linalg.matrix_rank(A)
            if self.rank_A >= f.dim:
                raise Exception("Dimension error: rg(A) >= f.dim")

        if (A is None and b is not None) or (A is not None and b is None):
            raise Exception("Problem have to be equality constrained or not, not half")
        self.f = f
        self.A = A
        self.b = b
