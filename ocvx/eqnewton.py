import numpy as np

class EQNewton:
    def __init__(self, P, pas, epsilon=0.01):
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
