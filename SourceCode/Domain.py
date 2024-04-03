import abc
from typing import Union, Callable
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class AbstractDomain:
    @abc.abstractmethod
    def calculate_integral(self, func: Callable) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_domain_values(self, n_poins: int):
        raise NotImplementedError


class OneDimDomain(AbstractDomain):
    def __init__(self, x_left: Union[float, int], x_right: Union[float, int]):
        self.x_left = x_left
        self.x_right = x_right

    def calculate_integral(self, func: Callable) -> float:
        return integrate.quad(func, self.x_left, self.x_right)[0]

    def get_domain_values(self, n_points: int):
        return np.linspace(self.x_left, self.x_right, n_points)

    def plot_function(self, function: Callable):
        n_points = 100
        dom_vals = self.get_domain_values(n_points)
        f_vals = function(dom_vals)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dom_vals, f_vals)
        ax.set_xlabel("X")
        ax.set_ylabel("f_values")
        plt.show()

class TwoDimDomain(AbstractDomain):
    def __init__(
        self,
        x_left: Union[float, int],
        x_right: Union[float, int],
        y_left: Union[float, int],
        y_right: Union[float, int],
    ):
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right

    def calculate_integral(self, func: Callable) -> float:
        return integrate.dblquad(
            func, self.x_left, self.x_right, self.y_left, self.y_right
        )[0]

    def get_domain_values(self, n_points: int):
        x = np.linspace(self.x_left, self.x_right, n_points)
        y = np.linspace(self.x_left, self.x_right, n_points)
        xm, ym = np.meshgrid(x, y)
        return xm, ym

    def plot_function(self, function: Callable):
        n_points = 100
        dom_vals = self.get_domain_values(n_points)
        f_vals = function(*dom_vals)
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = plt.axes(projection='3d')
        ax.plot_surface(*dom_vals, f_vals, cmap='viridis', \
                        edgecolor='green')
        ax.set_title('3D Contour Plot of function(x, y))', fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('f_values', fontsize=12)
        plt.show()




class ThreeDimDomain(AbstractDomain):
    def __init__(
        self,
        x_left: Union[float, int],
        x_right: Union[float, int],
        y_left: Union[float, int],
        y_right: Union[float, int],
        z_left: Union[float, int],
        z_right: Union[float, int]
    ):
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right
        self.z_left = z_left
        self.z_right = z_right

    def calculate_integral(self, func: Callable) -> float:
        return integrate.tplquad(
            func, self.x_left, self.x_right, self.y_left, self.y_right,
            self.z_left, self.z_right
        )[0]

    def get_domain_values(self, n_points: int):
        x = np.linspace(self.x_left, self.x_right, n_points)
        y = np.linspace(self.y_left, self.y_right, n_points)
        z = np.linspace(self.z_left, self.z_right, n_points)
        xm, ym, zm = np.meshgrid(x, y, z, indexing='ij')
        return xm, ym, zm

    def plot_function(self, function: Callable):
        raise NotImplementedError
