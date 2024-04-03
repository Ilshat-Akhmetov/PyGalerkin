import sympy
from typing import List, Callable
import numpy as np
from .Domain import AbstractDomain


class TargetFunction:
    def __init__(self, func_expr: str, f_variables: list):
        self.func = func_expr
        self.get_func = {}
        self.vars = f_variables
        self.get_func[""] = sympy.lambdify(f_variables, func_expr)
        self.func_expr = {}

    def __call__(self, *variables, derivative: str = "") -> Callable:
        # derivative = derivative.strip()
        # derivative = "".join(sorted(derivative))
        if derivative in self.get_func:
            return self.get_func[derivative](*variables)
        curr_f = self.func
        for i, var in enumerate(derivative):
            curr_diff = derivative[: i + 1]
            if curr_diff in self.func_expr:
                curr_f = self.func_expr[curr_diff]
            else:
                curr_f = sympy.diff(curr_f, var)
                self.func_expr[curr_diff] = curr_f
                self.get_func[curr_diff] = sympy.lambdify(self.vars, curr_f)
        return self.get_func[derivative](*variables)


class GalerkinEllipticSolver:
    def __init__(
        self,
        basic_funcs: List[str],
        f_variables: List[str],
        left_part: Callable,
        right_part: Callable,
        domain: AbstractDomain,
        border_func: str = "0",
    ):
        self.basic_funcs = [TargetFunction(func, f_variables) for func in basic_funcs]
        self.left_part = left_part
        self.right_part = right_part
        self.domain = domain
        self.solution = None
        self.border_func = TargetFunction(border_func, f_variables)

    @staticmethod
    def get_scalar_product(func1: Callable, func2: Callable) -> Callable:
        return lambda *variables: func1(*variables) * func2(*variables)

    def get_left_part(self, func) -> Callable:
        return lambda *variables: self.left_part(*variables, func)

    def get_left_part_int(self, col_func: Callable, row_func: Callable) -> float:
        left_part = self.get_left_part(col_func)
        scalar_product = self.get_scalar_product(left_part, row_func)
        integral_val = self.domain.calculate_integral(scalar_product)
        return integral_val

    def calculate_solution(self) -> None:
        n = len(self.basic_funcs)
        a_eq_part = np.zeros((n, n))
        b_eq_part = np.zeros(n)
        for row, row_func in enumerate(self.basic_funcs):
            for col, col_func in enumerate(self.basic_funcs):
                a_eq_part[row, col] = self.get_left_part_int(col_func, row_func)
            scalar_product = self.get_scalar_product(self.right_part, row_func)
            integral_val = self.domain.calculate_integral(scalar_product)
            b_eq_part[row] = integral_val
            b_eq_part[row] -= self.get_left_part_int(self.border_func, row_func)
        self.solution = np.linalg.solve(a_eq_part, b_eq_part)

    def get_solution(self) -> Callable:
        if self.solution is None:
            raise Exception("First run calculate_solution")
        else:
            return lambda *variables: sum(
                coefficient * basis_func(*variables)
                for coefficient, basis_func in zip(self.solution, self.basic_funcs)
            ) + self.border_func(*variables)
