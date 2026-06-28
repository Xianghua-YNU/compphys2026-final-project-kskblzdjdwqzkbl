"""Simplified two-dimensional N-body open-cluster dynamics.

AI-assisted implementation: this module was drafted with ChatGPT/Codex and
reviewed for the project requirements.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


Array = np.ndarray


def generate_cluster_initial_conditions(
    n_particles: int = 300,
    sigma_r: float = 0.5,
    sigma_v: float = 0.15,
    v_cluster: tuple[float, float] = (0.0, 0.0),
    random_seed: int | None = 42,
) -> tuple[Array, Array, Array]:
    """Generate initial positions, velocities, and equal particle masses.

    The model is intentionally dimensionless and two-dimensional. Positions are
    sampled from a Gaussian cluster, while velocities represent a moderate
    internal dispersion. The mass-weighted center of mass and bulk velocity are
    removed so the cluster starts near the origin with zero net momentum.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if sigma_r < 0.0:
        raise ValueError("sigma_r must be non-negative")
    if sigma_v < 0.0:
        raise ValueError("sigma_v must be non-negative")

    rng = np.random.default_rng(random_seed)
    positions = rng.normal(loc=0.0, scale=sigma_r, size=(n_particles, 2))
    velocities = rng.normal(loc=0.0, scale=sigma_v, size=(n_particles, 2))
    velocities += np.asarray(v_cluster, dtype=float)

    masses = np.full(n_particles, 1.0 / n_particles, dtype=float)
    positions -= np.average(positions, axis=0, weights=masses)
    velocities -= np.average(velocities, axis=0, weights=masses)
    return positions, velocities, masses


def compute_accelerations(
    positions: Array,
    masses: Array,
    gravitational_constant: float = 1.0,
    softening: float = 0.08,
    tidal_strength: float = 0.002,
) -> Array:
    """Compute softened self-gravity plus a simple external tidal field.

    The pairwise softened acceleration is

        a_i = G sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2).

    The diagonal terms are explicitly set to zero so particles do not exert a
    force on themselves. The tide is a linearized background field used to
    stretch the cluster along x and compress it weakly along y.
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have shape (N,)")
    if softening <= 0.0:
        raise ValueError("softening must be positive")

    separation = positions[None, :, :] - positions[:, None, :]
    distance_squared = np.sum(separation * separation, axis=2) + softening**2
    inverse_distance_cubed = distance_squared ** (-1.5)
    np.fill_diagonal(inverse_distance_cubed, 0.0)

    self_gravity = gravitational_constant * np.sum(
        separation * inverse_distance_cubed[:, :, None] * masses[None, :, None],
        axis=1,
    )

    tidal_acceleration = np.empty_like(positions)
    tidal_acceleration[:, 0] = tidal_strength * positions[:, 0]
    tidal_acceleration[:, 1] = -0.5 * tidal_strength * positions[:, 1]

    return self_gravity + tidal_acceleration


def total_energy_cluster(
    positions: Array,
    velocities: Array,
    masses: Array,
    gravitational_constant: float = 1.0,
    softening: float = 0.08,
    return_components: bool = False,
) -> float | tuple[float, float, float]:
    """Return kinetic plus softened self-gravitational potential energy.

    The external tidal potential is deliberately not included. This makes the
    result a useful diagnostic of the cluster's internal energy scale, but it is
    not an exactly conserved quantity when the tidal field is active.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have shape (N,)")
    if softening <= 0.0:
        raise ValueError("softening must be positive")

    speed_squared = np.sum(velocities * velocities, axis=1)
    kinetic_energy = 0.5 * np.sum(masses * speed_squared)

    i_upper, j_upper = np.triu_indices(positions.shape[0], k=1)
    pair_separation = positions[i_upper] - positions[j_upper]
    pair_distance = np.sqrt(np.sum(pair_separation * pair_separation, axis=1) + softening**2)
    potential_energy = -gravitational_constant * np.sum(
        masses[i_upper] * masses[j_upper] / pair_distance
    )

    total_energy = kinetic_energy + potential_energy
    if return_components:
        return float(total_energy), float(kinetic_energy), float(potential_energy)
    return float(total_energy)


def velocity_verlet_cluster(
    positions: Array,
    velocities: Array,
    masses: Array,
    dt: float,
    n_steps: int,
    save_every: int = 100,
    gravitational_constant: float = 1.0,
    softening: float = 0.08,
    tidal_strength: float = 0.002,
    extra_save_steps: Iterable[int] | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Advance the cluster with the Velocity Verlet method.

    Snapshots are saved at step 0, every ``save_every`` steps, and at the final
    step. Optional ``extra_save_steps`` make it convenient to store selected
    analysis times such as the leading/trailing tail comparison epochs.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if save_every <= 0:
        raise ValueError("save_every must be positive")

    current_positions = np.asarray(positions, dtype=float).copy()
    current_velocities = np.asarray(velocities, dtype=float).copy()
    masses = np.asarray(masses, dtype=float)

    if current_positions.ndim != 2 or current_positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if current_velocities.shape != current_positions.shape:
        raise ValueError("velocities must have the same shape as positions")
    if masses.shape != (current_positions.shape[0],):
        raise ValueError("masses must have shape (N,)")

    save_steps = set(range(0, n_steps + 1, save_every))
    save_steps.add(0)
    save_steps.add(n_steps)
    if extra_save_steps is not None:
        for step in extra_save_steps:
            if 0 <= step <= n_steps:
                save_steps.add(int(step))

    times: list[float] = []
    position_snapshots: list[Array] = []
    velocity_snapshots: list[Array] = []
    energy_history: list[float] = []

    def save_snapshot(step: int) -> None:
        times.append(step * dt)
        position_snapshots.append(current_positions.copy())
        velocity_snapshots.append(current_velocities.copy())
        energy_history.append(
            total_energy_cluster(
                current_positions,
                current_velocities,
                masses,
                gravitational_constant=gravitational_constant,
                softening=softening,
            )
        )

    save_snapshot(0)
    acceleration = compute_accelerations(
        current_positions,
        masses,
        gravitational_constant=gravitational_constant,
        softening=softening,
        tidal_strength=tidal_strength,
    )

    for step in range(1, n_steps + 1):
        current_positions += current_velocities * dt + 0.5 * acceleration * dt**2
        new_acceleration = compute_accelerations(
            current_positions,
            masses,
            gravitational_constant=gravitational_constant,
            softening=softening,
            tidal_strength=tidal_strength,
        )
        current_velocities += 0.5 * (acceleration + new_acceleration) * dt
        acceleration = new_acceleration

        if step in save_steps:
            save_snapshot(step)

    return (
        np.asarray(times, dtype=float),
        np.asarray(position_snapshots, dtype=float),
        np.asarray(velocity_snapshots, dtype=float),
        np.asarray(energy_history, dtype=float),
    )
