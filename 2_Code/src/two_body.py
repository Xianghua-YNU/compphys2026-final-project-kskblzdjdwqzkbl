"""Two-body orbital dynamics utilities.

AI-assisted implementation: this module was drafted with ChatGPT/Codex and
reviewed for the project requirements.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from integrators import euler_step, rk4_step, velocity_verlet_step


Array = np.ndarray
StepFunc = Callable[[Array, Array, float, Callable[[Array], Array]], tuple[Array, Array]]


INTEGRATORS: dict[str, StepFunc] = {
    "euler": euler_step,
    "rk4": rk4_step,
    "velocity_verlet": velocity_verlet_step,
    "verlet": velocity_verlet_step,
}


def gravitational_acceleration(
    position: Array,
    central_mass: float = 1.0,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
) -> Array:
    """Return gravitational acceleration from a fixed central mass.

    The equation is

        a(r) = -G M r / |r|^3

    ``position`` can be a single vector with shape ``(dim,)`` or an array of
    vectors with shape ``(..., dim)``. The ``softening`` parameter is kept
    optional for later N-body experiments where close encounters may need
    numerical regularization; it is zero for the present two-body comparison.
    """
    position = np.asarray(position, dtype=float)
    radius_squared = np.sum(position * position, axis=-1, keepdims=True) + softening**2
    radius = np.sqrt(radius_squared)

    if np.any(radius == 0.0):
        raise ValueError("gravitational_acceleration is undefined at r = 0")

    return -gravitational_constant * central_mass * position / radius**3


def total_energy(
    position: Array,
    velocity: Array,
    central_mass: float = 1.0,
    planet_mass: float = 1.0e-3,
    gravitational_constant: float = 1.0,
) -> Array:
    """Compute the mechanical energy of a planet orbiting a fixed central mass.

    E = 1/2 m |v|^2 - G M m / |r|
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    speed_squared = np.sum(velocity * velocity, axis=-1)
    radius = np.linalg.norm(position, axis=-1)
    if np.any(radius == 0.0):
        raise ValueError("total_energy is undefined at r = 0")

    kinetic_energy = 0.5 * planet_mass * speed_squared
    potential_energy = -gravitational_constant * central_mass * planet_mass / radius
    return kinetic_energy + potential_energy


def simulate_two_body(
    integrator: str | StepFunc,
    initial_position: Array,
    initial_velocity: Array,
    t_end: float,
    dt: float,
    central_mass: float = 1.0,
    planet_mass: float = 1.0e-3,
    gravitational_constant: float = 1.0,
) -> dict[str, Array]:
    """Simulate a planet orbiting a fixed central mass.

    Parameters are deliberately explicit so later experiments can reuse this
    function with different masses, units, and time steps. The returned arrays
    include time, position, velocity, and total energy at each stored step.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")

    if isinstance(integrator, str):
        key = integrator.lower()
        if key not in INTEGRATORS:
            valid_names = ", ".join(sorted(INTEGRATORS))
            raise ValueError(f"Unknown integrator '{integrator}'. Valid names: {valid_names}")
        step_func = INTEGRATORS[key]
    else:
        step_func = integrator

    position0 = np.asarray(initial_position, dtype=float)
    velocity0 = np.asarray(initial_velocity, dtype=float)
    if position0.shape != velocity0.shape:
        raise ValueError("initial_position and initial_velocity must have the same shape")

    n_steps = int(round(t_end / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    positions = np.empty((n_steps + 1, *position0.shape), dtype=float)
    velocities = np.empty((n_steps + 1, *velocity0.shape), dtype=float)
    energies = np.empty(n_steps + 1, dtype=float)

    positions[0] = position0
    velocities[0] = velocity0
    energies[0] = total_energy(
        positions[0],
        velocities[0],
        central_mass=central_mass,
        planet_mass=planet_mass,
        gravitational_constant=gravitational_constant,
    )

    def acceleration_func(position: Array) -> Array:
        return gravitational_acceleration(
            position,
            central_mass=central_mass,
            gravitational_constant=gravitational_constant,
        )

    for step_index in range(n_steps):
        positions[step_index + 1], velocities[step_index + 1] = step_func(
            positions[step_index],
            velocities[step_index],
            dt,
            acceleration_func,
        )
        energies[step_index + 1] = total_energy(
            positions[step_index + 1],
            velocities[step_index + 1],
            central_mass=central_mass,
            planet_mass=planet_mass,
            gravitational_constant=gravitational_constant,
        )

    return {
        "time": times,
        "position": positions,
        "velocity": velocities,
        "energy": energies,
    }
