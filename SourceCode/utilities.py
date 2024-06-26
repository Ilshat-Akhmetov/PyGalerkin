from typing import Callable
import numpy as np


def get_max_error(true_sol: Callable, approx: Callable, *domain) -> float:
    """
    true_sol: analytical solution
    approx: approximated solution
    *domain: list of arbitrary number of domain vars like x,y,z
    """
    true_vals = true_sol(*domain)
    approximated_values = approx(*domain)
    return np.max(abs(true_vals - approximated_values))
