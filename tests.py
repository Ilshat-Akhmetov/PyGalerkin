from SourceCode.GalerkinEllipticEq import GalerkinEllipticSolver
from SourceCode.utilities import get_max_error
from SourceCode.Domain import OneDimDomain, TwoDimDomain
import numpy as np
from math import pi
import time


def test_1d_func():
    n_fs = 6
    funcs = ["x**{}".format(i + 1) for i in range(n_fs)]
    left_part = lambda x, func: func(x, derivative="x") - func(x)
    right_part = lambda x: 0
    left_b = 0
    right_b = 1
    variables = ["x"]
    domain = OneDimDomain(left_b, right_b)
    start = time.time()
    obj = GalerkinEllipticSolver(
        funcs, variables, left_part, right_part, domain, border_func="1"
    )
    obj.calculate_solution()
    print("time passed: {}".format(time.time() - start))
    approx_f = obj.get_solution()
    # print(obj.solution)
    n_points = 100
    dom_vals = domain.get_domain_values(n_points)
    true_sol = lambda x: np.exp(x)
    error = get_max_error(true_sol, approx_f, dom_vals)
    tol = 1e-5
    domain.plot_function(approx_f)
    print("error {}".format(error))
    assert error < tol

    # for i in range(N):
    #     print("x: {} appr: {} true: {}".format(x[i], f(x[i]), np.exp(x[i])))


def test_2d_func():
    def true_solution(x, y):
        n: int = 100
        total_s: float = 0
        for i in range(1, n, 2):
            for j in range(1, n, 2):
                total_s += (
                    (-1) ** ((i + j) // 2 - 1)
                    / (i * j * (i * i + j * j))
                    * np.cos(i * pi / 2 * x)
                    * np.cos(j * pi / 2 * y)
                )
        total_s = total_s * (8 / (pi * pi)) ** 2
        return total_s

    n_fs = 5
    # funcs = ["cos({}*pi/2*x)*cos({}*pi/2*y)".format(i, j) for i in range(1, n_fs, 2) for j in range(1, n_fs, 2)]
    funcs = [
        "(1-x*x)**{}*(1-y*y)**{}".format(i, j)
        for i in range(1, n_fs, 2)
        for j in range(1, n_fs, 2)
    ]
    variables = ["x", "y"]
    left_part = lambda x, y, func: func(x, y, derivative="xx") + func(
        x, y, derivative="yy"
    )
    right_part = lambda x, y: -1
    domain = TwoDimDomain(-1, 1, -1, 1)
    start = time.time()
    obj = GalerkinEllipticSolver(funcs, variables, left_part, right_part, domain)
    obj.calculate_solution()
    print("time passed: {}".format(time.time() - start))
    approx_f = obj.get_solution()
    n_points = 100
    dom_vals = domain.get_domain_values(n_points)
    error = get_max_error(true_solution, approx_f, *dom_vals)
    domain.plot_function(approx_f)
    tol = 1e-2
    print("error {}".format(error))
    assert error < tol


test_1d_func()
test_2d_func()
