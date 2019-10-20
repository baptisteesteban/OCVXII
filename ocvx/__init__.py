from .newton import constant, backtracking, Newton
from .probleme import Function, Probleme
from .eqnewton import EQNewton
from .gradient_descent import GradientDescent
from .eqgradient import EQGradient, eqgrad_backtracking

__all__ = ["constant", "backtracking", "Newton", "Function", "Probleme", "EQNewton", "GradientDescent", "EQGradient", "eqgrad_backtracking"]
