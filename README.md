# PyGalerkin
Python implementation of the classical Bubnov-Galerkin method for solving differential equations.
The current implementation supports solving elliptic equation in 1,2 and 3-dimensional cases, 
although program was designed to be easily extendable for 4,5,6... dimensional cases.

Currently, the implementation also supports only a simple rectangular domain.

A non-stationary problem (for example, parabolic heat equation) also can be represented 
as a stationary one by considering time variable as a space variable like x, y, z.

## How to use?

First of all, make sure you have python3.8 or newer and all libraries from *requirements.txt*

Here are some examples how to use this program for your equations. API is pretty simple and intuitive.
You just have to define set of basis functions so that all of them equal to zero at the equation's domains boundary, 
optionally pass so-called *boundary_function*
(in order to make the approximating function to satisfy the boundary conditions, its coefficient are not calculated), l
eft and right parts of the equation (left part depends on target function $u$ and right part does not) 
as lambda expressions or functions.

The solution will be represented as a linear combination of basis functions 
plus boundary function with its coefficient equals 1.

The program can work with arbitrary linear elliptic equations with continuous coefficients and right part. 
You don't have to calculate 
derivatives of the basis functions, program will do it for you using sympy's symbolic derivative calculations.

You can also compare the solution obtained with this program with the exact solution or compare in terms of deviance
from exact solution with a solution you calculated with some other method, 
for example finite element or finite difference method.

First example, say you need to solve the equation

$$ y'- y = 0 $$

$$ y(0) = 1 $$

```python
from SourceCode import *
import numpy as np

n_fs = 6
basis_funcs = ["x**{}".format(i + 1) for i in range(n_fs)]
left_eq_part = lambda x, func: func(x, derivative="x") - func(x)
right_eq_part = lambda x: 0
left_b = 0
right_b = 1
variables = ["x"]
domain = OneDimDomain(left_b, right_b)
obj = GalerkinEllipticSolver(
    basis_funcs, variables, left_eq_part, right_eq_part, domain, boundary_function="1"
)
obj.calculate_solution()
f_approx = obj.get_solution()
print("coefficients: {}".format(obj.solution))
n_points = 100
dom_vals = domain.get_domain_values(n_points)
exact_sol = lambda x: np.exp(x)
error = get_max_error(exact_sol, f_approx, dom_vals)
print("max error {}".format(error))
```
second example

$$ f_{yy} + f_{xx} = -1 $$

$$ f(-1,y)=f(1,y)=f(x,-1)=f(x,1)=0 $$

```python
from SourceCode import *
import numpy as np
from math import pi


def exact_solution(x, y):
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
# basis_funcs = ["cos({}*pi/2*x)*cos({}*pi/2*y)".format(i, j) for i in range(1, n_fs, 2) for j in range(1, n_fs, 2)]
basis_funcs = [
    "(1-x*x)**{}*(1-y*y)**{}".format(i, j)
    for i in range(1, n_fs, 2)
    for j in range(1, n_fs, 2)
]
variables = ["x", "y"]
left_eq_part = lambda x, y, func: func(x, y, derivative="xx") + func(
    x, y, derivative="yy"
)
right_eq_part = lambda x, y: -1
domain = TwoDimDomain(-1, 1, -1, 1)
obj = GalerkinEllipticSolver(basis_funcs, variables, left_eq_part, right_eq_part, domain)
obj.calculate_solution()
f_approx = obj.get_solution()
n_points = 100
dom_vals = domain.get_domain_values(n_points)
error = get_max_error(exact_solution, f_approx, *dom_vals)
print("error {}".format(error))
domain.plot_function(f_approx)
```

As you can see you may also compare obtained solution with the exact solution if you have one.

In case of discontinues coefficients or a discontinuous right part you still may get a correct solution 
if you choose proper basis functions.

Current implementation was tested on Dirichlet problems, but I think it should work fine also with
Neumann and even mixed boundary conditions if boundary function and basis functions are properly chosen.

In the future I have plans to extend current program for non-stationary (parabolic and hyperbolic) equations.

## Short documentation
class *GalerkinEllipticSolver* has the following init parameters:
* **basic_funcs: List[str]** - set of basis functions. The solution will be represented as their linear combination
* **f_variables: List[str]** - variables which are present in basis functions or boundary_function
* **left_eq_part: Callable** - left part of the equation. For example, if L[u]=f(x) then L[u] is the left part. 
Here L - is the differential operator.
* **right_eq_part: Callable** - right part of the equation. In the example above f(x) is the right part.
* **domain: AbstractDomain** - a domain your function is defined on. Can be OneDimDomain, TwoDim or ThreeDim
* **boundary_function: str = "0"** - optional function, required to satisfy boundary conditions. 

In **the left_eq_part** to show where the target function is you have to use the following syntax:
function(*vars, derivative="xy..x"). 

Here *vars is an arbitrary number of variables the target function depends on, for example x, y, z.
By default, *derivative* equals to an empty string which means derivatives should not be calculated. 

For example, if you set derivative = "xxx" it means that here function should calculate 3-order derivative by x.
As an example if
```python
left_eq_part = lambda x, y, func: func(x,y, derivative='xy')
right_eq_part = lambda x,y: x*y
```
then it means that we have an equation $ u_{xy}=xy $.

*GalerkinEllipticSolver* has several methods, but you need only two of them:
* **calculate_solution(self)** - after init you should execute it to calcuate coefficients of the basis functions
* **get_solution(self)** - get callable function as a linear comb of basis functions plus boundary function. Can be
executed only after **calculate_solution(self)** is used

class *OneDimDomain*. Here are its init parameters:
* **x_left** - left boundary of the domain
* **x_right** - right boundary of the domain


*TwoDimDomain* and *ThreeDimDomain* init params are similar, you just also need for example 
to specify *y_left* and *y_right* . After initialization, you should pass an instance of domain 
class to the *GalerkinEllipticSolver* as the init param.

It has one method you may be interested in: *plot_function(self, function: Callable)*. 
It gets a callable function (probably true solution) 
as a param and makes a plot.






