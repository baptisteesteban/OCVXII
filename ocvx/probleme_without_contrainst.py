from .probleme import Function, Probleme

f_d = {
    "dim": 1,
    "value": lambda x: x[0]**2 - 5 * x[0] + 3,
    "grad": lambda x: np.array([2*x[0] - 5]),
    "hessian": lambda x: np.diag([2])
}
f = Function(**f_d)
P = Probleme(f)


f_d_2 = {
    "dim": 1,
    "value": lambda x: x[0]**4 - 10 * x[0] + 3,
    "grad": lambda x: np.array([4*x[0]**3 - 10]),
    "hessian": lambda x: np.diag(np.array([12*x[0]**2]))
}
f_2 = Function(**f_d_2)
P_2 = Probleme(f_2)


f_d_3 = {
    "dim": 1,
    "value": lambda x: x[0]**6 - 10 * x[0]**2,
    "grad": lambda x: np.array([6*x[0]**5 - 20 * x[0]]),
    "hessian": lambda x: np.diag(np.array([30*x[0]**4 - 20]))
}
f_3 = Function(**f_d_3)
P_3 = Probleme(f_3)


f_d_4 = {
    "dim": 1,
    "value": lambda x: 36 * x[0]**2,
    "grad": lambda x: np.array([72 * x[0]]),
    "hessian": lambda x: np.diag([72])
}
f_4 = Function(**f_d_4)
P_4 = Probleme(f_4)


f_d_5 = {
    "dim": 1,
    "value": lambda x: 10 * x[0]**2 - 5 * x[0] + 3,
    "grad": lambda x: np.array([20 * x[0] - 5]),
    "hessian": lambda x: np.diag([20])
}
f_5 = Function(**f_d_5)
P_5 = Probleme(f_5)


f_d_6 = {
    "dim": 1,
    "value": lambda x: 10 * x[0]**6 - 5 * x[0]**5 + 3 * x[0]**4 - 2 * x[0]**3 + 15 * x[0]**2 - 12 * x[0] + 15,
    "grad": lambda x: np.array([60 * x[0]**5 - 25 * x[0]**4 + 12 * x[0]**3 - 6 * x[0]**2 + 30 * x[0] - 12]),
    "hessian": lambda x: np.diag([300 * x[0]**4 - 100 * x[0]**3 + 36 * x[0]**2 - 12 * x[0] + 30])
}
f_6 = Function(**f_d_6)
P_6 = Probleme(f_6)
