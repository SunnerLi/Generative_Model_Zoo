"""This script defines solver and integral function to solve DDPM/flow matching"""

from torchdiffeq._impl.odeint import SOLVERS

from torchdiffeq._impl.solvers import FixedGridODESolver
from torchdiffeq._impl.misc import _flat_to_shape, _check_inputs
import torch


class DiffusionSolver(FixedGridODESolver):
    """Implementation of DDPM pipeline with torchdiffeq API"""

    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return f0

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(
            len(t) + 1,
            *self.y0.shape,
            dtype=self.y0.dtype,
            device=self.y0.device,
        )
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for _t in time_grid:
            y0 = self._step_func(self.func, _t, None, None, y0)
            solution[j] = y0
            j += 1

        return solution


# Register DDPM solver
SOLVERS.update(
    {
        "diffusion": DiffusionSolver,
    }
)


# We have rewritten the odeint function from the torchdiffeq package.
# The reason is that the timesteps in the Diffusers noise scheduler are
# arranged in descending order, whereas the odeint function in torchdiffeq
# by default expects the time grid to be in ascending order.
# As a result, the reverse callback function used in odeint causes
# the timesteps' signs to be incorrect, leading to inference errors.
#
# Modified from: https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py
def odeint(
    func,
    y0,
    t,
    *,
    rtol=1e-7,
    atol=1e-9,
    method=None,
    options=None,
    event_fn=None,
):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`, in either increasing or decreasing order. The first element of
            this sequence is taken to be the initial time point.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
            event_fn evaluates to zero. If this is not None, all but the first elements of
            `t` are ignored.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """
    if method == "diffusion":
        # Skip parameter check in diffusers setting
        shapes = None

        solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol)
    else:
        (
            shapes,
            func,
            y0,
            t,
            rtol,
            atol,
            method,
            options,
            event_fn,
            t_is_reversed,
        ) = _check_inputs(
            func, y0, t, rtol, atol, method, options, event_fn, SOLVERS
        )

        solver = SOLVERS[method](
            func=func, y0=y0, rtol=rtol, atol=atol, **options
        )

    if event_fn is None:
        solution = solver.integrate(t)
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.to(t)
        # if t_is_reversed:
        #     event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution
