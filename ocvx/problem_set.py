import numpy as np
import pandas as pd

from .probleme import Probleme, Function

def getUnconstrainedProblems():
    quad_d = {
        "value": lambda x: 2 * x[0]**2 + 4 * x[0] + 5,
        "dim": 1,
        "grad": lambda x: np.array([4 * x[0] + 4]),
        "hessian": lambda x: np.diag([4])
    }
    
    quad_2_d = {
        "value": lambda x: 4 * x[0]**2 + 5 * x[1]**2 + 7 * x[1],
        "dim": 2,
        "grad": lambda x: np.array([8 * x[0], 10 * x[1] + 7]),
        "hessian": lambda x: np.diag([8, 10])
    }
    
    trigo_d = {
        "value": lambda x: 4 * np.cos(x[0]),
        "dim": 1,
        "grad": lambda x: np.array([- 4 * np.sin(x[0])]),
        "hessian": lambda x: np.diag([- 4 * np.cos(x[0])])
    }
    
    P_unconstrained = [
        {"name": "quad_1", "probleme": Probleme(Function(**quad_d))},
        {"name": "quad_2", "probleme": Probleme(Function(**quad_2_d))},
        {"name": "trigo", "probleme": Probleme(Function(**trigo_d))},
    ]

    return pd.DataFrame(P_unconstrained)

def getConstrainedProblems():    
    quad_2_d = {
        "value": lambda x: 4 * x[0]**2 + 5 * x[1]**2 + 7 * x[1],
        "dim": 2,
        "grad": lambda x: np.array([8 * x[0], 10 * x[1] + 7]),
        "hessian": lambda x: np.diag([8, 10])
    }
    
    P_constrained = [
        {"name": "quad_1", "probleme": Probleme(Function(**quad_2_d), np.array([[1, 2]]), np.array([3])), "x0": np.array([-17, 10])}
    ]
        
    return pd.DataFrame(P_constrained)