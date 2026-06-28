"""Basic time integrators for Newtonian dynamics.

AI-assisted implementation: this module was drafted with ChatGPT/Codex and
reviewed for the project requirements.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


Array = np.ndarray
AccelerationFunc = Callable[[Array], Array]


def euler_step(
    position: Array,
    velocity: Array,
    dt: float,
    acceleration_func: AccelerationFunc,
) -> tuple[Array, Array]:
    """Advance one step with the first-order explicit Euler method.

    The position and velocity are updated from values at the beginning of the
    time step. This method is simple but not symplectic, so it usually shows a
    clear long-term energy drift in orbital problems.
    """
    acceleration = acceleration_func(position)
    new_position = position + velocity * dt
    new_velocity = velocity + acceleration * dt
    return new_position, new_velocity


def rk4_step(
    position: Array,
    velocity: Array,
    dt: float,
    acceleration_func: AccelerationFunc,
) -> tuple[Array, Array]:
    """Advance one step with the classical fourth-order Runge-Kutta method.

    The second-order Newton equation is written as a first-order system:

        dr/dt = v,    dv/dt = a(r)

    RK4 has high local accuracy, but it is not symplectic, so very long orbital
    integrations can still accumulate secular energy errors.
    """

    def derivative(pos: Array, vel: Array) -> tuple[Array, Array]:
        return vel, acceleration_func(pos)

    k1_pos, k1_vel = derivative(position, velocity)
    k2_pos, k2_vel = derivative(
        position + 0.5 * dt * k1_pos,
        velocity + 0.5 * dt * k1_vel,
    )
    k3_pos, k3_vel = derivative(
        position + 0.5 * dt * k2_pos,
        velocity + 0.5 * dt * k2_vel,
    )
    k4_pos, k4_vel = derivative(
        position + dt * k3_pos,
        velocity + dt * k3_vel,
    )

    new_position = position + (dt / 6.0) * (
        k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos
    )
    new_velocity = velocity + (dt / 6.0) * (
        k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel
    )
    return new_position, new_velocity


def velocity_verlet_step(
    position: Array,
    velocity: Array,
    dt: float,
    acceleration_func: AccelerationFunc,
) -> tuple[Array, Array]:
    """Advance one step with the symplectic Velocity Verlet method.

    Velocity Verlet uses the acceleration at both ends of the time step. For a
    conservative gravitational system this usually keeps the total energy
    bounded and oscillatory instead of producing a strong monotonic drift.
    """
    acceleration = acceleration_func(position)
    new_position = position + velocity * dt + 0.5 * acceleration * dt**2
    new_acceleration = acceleration_func(new_position)
    new_velocity = velocity + 0.5 * (acceleration + new_acceleration) * dt
    return new_position, new_velocity
