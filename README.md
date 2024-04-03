# PyGalerkin
Python implementation of the classical Bubnov-Galerkin method for solving differential equations.

How to use?
Current implementation can be applied only to linear stationary (elliptic) equations in 1,2,3 dimensions,
although program was designed to be easily extendable to 4,5,6.. dimensional cases.

Here are some examples how to use this program for your equations. API is pretty simple and intuitive.
You just have to define set of basis function so that all of them equals to zero at border, optionally 
pass so-called border function (it is required to satisfy boundary conditions, its coefficient is not calculated)
and left and right part of the equation (left part depends on target function $u$ and right part does not)

first example, say you need to solve the equation
$$ y'-y=0 $$
$$ y(0)=1$$
```python
n_fs = 6
funcs = ["x**{}".format(i + 1) for i in range(n_fs)]
left_part = lambda x, func: func(x, derivative="x") - func(x)
right_part = lambda x: 0
left_b = 0
right_b = 1
variables = ["x"]
domain = OneDimDomain(left_b, right_b)
obj = GalerkinEllipticSolver(
    funcs, variables, left_part, right_part, domain, border_func="1"
)
obj.calculate_solution()
f_approx = obj.get_solution()
print("coeffs: {}".format(obj.solution))
n_points = 100
dom_vals = domain.get_domain_values(n_points)
true_sol = lambda x: np.exp(x)
error = get_max_error(true_sol, f_approx, dom_vals)
print("max error {}".format(error))
```
second example
$$ f_{yy} + f_{xx} = -1 $$
$$ f(-1,y)=f(1,y)=f(x,-1)=f(x,1)=0 $$
```python
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
obj = GalerkinEllipticSolver(funcs, variables, left_part, right_part, domain)
obj.calculate_solution()
f_approx = obj.get_solution()
n_points = 100
dom_vals = domain.get_domain_values(n_points)
error = get_max_error(true_solution, f_approx, *dom_vals)
print("error {}".format(error))
domain.plot_function(f_approx)
```

As you can see you may also compare obtained solution with the true solution if you have one.
But program should work correctly with any linear equation with continuous left part's coefficients and a continuous right part.

You may find more examples in *Presentations.py*