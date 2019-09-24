import numpy as np

class Function:
    def __init__(self, dim, value, grad, hessian):
        self.dim = dim
        self.value = value
        self.grad = grad
        self.hessian = hessian

class Probleme:
    def __init__(self, f):
        self.f = f
