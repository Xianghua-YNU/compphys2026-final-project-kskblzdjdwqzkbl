"""Run simplified open-cluster N-body experiments.

AI-assisted implementation: this script was drafted with ChatGPT/Codex and
reviewed for the project requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_CODE_DIR / "src"
OUTPUT_DIR = PROJECT_CODE_DIR / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nbody_cluster import (  # noqa: E402
    generate_cluster_initial_conditions,
    velocity_verlet_cluster,
)


def relative_energy_error(energy: np.ndarray) -> np.ndarray:
    """Compute |E(t) - E(0)| / |E(0)| for one energy history."""
    initial_energy = energy[0]
    if initial_energy == 0.0:
        raise ValueError("Initial energy is zero, so relative error is undefined")
    return np.abs(energy - initial_energy) / abs(initial_energy)


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return the mass-weighted center of mass of one snapshot."""
    return np.average(positions, axis=0, weights=masses)


def to_comoving_positions(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Shift one snapshot into the cluster center-of-mass frame."""
    return positions - center_of_mass(positions, masses)


def select_snapshot_indices(times: np.ndarray, target_times: list[float]) -> list[int]:
    """Find the stored snapshot nearest to each requested plotting time."""
    return [int(np.argmin(np.abs(times - target_time))) for target_time in target_times]


def plot_isolated_energy_error(
    times: np.ndarray,
    energy_history: np.ndarray,
    output_path: Path,
) -> None:
    """Save the energy conservation diagnostic for the isolated cluster."""
    error = relative_energy_error(energy_history)
    plotted_error = np.maximum(error, 1.0e-16)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.semilogy(times, plotted_error, marker="o", markersize=3.5, linewidth=1.2)
    ax.set_xlabel("Time [model units]")
    ax.set_ylabel("Relative total energy error")
    ax.set_title("Energy conservation test for isolated cluster")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tidal_evolution_comoving(
    times: np.ndarray,
    position_snapshots: np.ndarray,
    masses: np.ndarray,
    target_times: list[float],
    output_path: Path,
) -> None:
    """Save selected snapshots in the center-of-mass frame."""
    snapshot_indices = select_snapshot_indices(times, target_times)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True, sharey=True)
    for ax, snapshot_index in zip(axes.flat, snapshot_indices):
        positions = to_comoving_positions(position_snapshots[snapshot_index], masses)
        ax.scatter(positions[:, 0], positions[:, 1], s=8, alpha=0.68, edgecolors="none")
        ax.scatter([0.0], [0.0], marker="+", s=120, color="black", label="Center of mass")
        ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t = {times[snapshot_index]:.1f}")
        ax.set_xlabel("x - x_cm [model units]")
        ax.set_ylabel("y - y_cm [model units]")
        ax.grid(True, linestyle="--", alpha=0.30)

    axes.flat[0].legend(loc="upper right")
    fig.suptitle("Comoving tidal evolution of a simplified open cluster", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def classify_tail_candidates(
    comoving_positions: np.ndarray,
    x_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """Classify particles as core, leading-tail, or trailing-tail candidates."""
    leading_mask = comoving_positions[:, 0] > x_threshold
    trailing_mask = comoving_positions[:, 0] < -x_threshold
    core_mask = ~(leading_mask | trailing_mask)

    n_lead = int(np.sum(leading_mask))
    n_trail = int(np.sum(trailing_mask))
    tail_total = n_lead + n_trail
    if tail_total == 0:
        tail_asymmetry = 0.0
    else:
        tail_asymmetry = (n_lead - n_trail) / tail_total

    return core_mask, leading_mask, trailing_mask, n_lead, n_trail, float(tail_asymmetry)


def plot_tail_classification(
    comoving_positions: np.ndarray,
    core_mask: np.ndarray,
    leading_mask: np.ndarray,
    trailing_mask: np.ndarray,
    tail_asymmetry: float,
    output_path: Path,
) -> None:
    """Save the final-snapshot tail classification figure."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(
        comoving_positions[core_mask, 0],
        comoving_positions[core_mask, 1],
        s=12,
        alpha=0.70,
        label="Cluster core candidate",
    )
    ax.scatter(
        comoving_positions[leading_mask, 0],
        comoving_positions[leading_mask, 1],
        s=24,
        marker=">",
        alpha=0.85,
        label="Leading tail candidate",
    )
    ax.scatter(
        comoving_positions[trailing_mask, 0],
        comoving_positions[trailing_mask, 1],
        s=24,
        marker="<",
        alpha=0.85,
        label="Trailing tail candidate",
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axvline(-1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
    ax.set_xlabel("x - x_cm [model units]")
    ax.set_ylabel("y - y_cm [model units]")
    ax.set_title("Final tail candidate classification")
    ax.text(
        0.02,
        0.95,
        f"A_tail = {tail_asymmetry:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_isolated_energy_test(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run experiment A: an isolated-cluster energy conservation test."""
    dt = 0.005
    t_end = 20.0
    save_every = 200
    n_steps = int(round(t_end / dt))

    times, _, _, energy_history = velocity_verlet_cluster(
        positions=initial_positions,
        velocities=initial_velocities,
        masses=masses,
        dt=dt,
        n_steps=n_steps,
        save_every=save_every,
        tidal_strength=0.0,
    )

    return times, energy_history


def run_tidal_evolution(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run experiment B: gentle tidal evolution of the cluster."""
    dt = 0.01
    t_end = 80.0
    save_every = 1000
    target_times = [0.0, 20.0, 40.0, 80.0]
    n_steps = int(round(t_end / dt))
    extra_save_steps = [int(round(target_time / dt)) for target_time in target_times]

    return velocity_verlet_cluster(
        positions=initial_positions,
        velocities=initial_velocities,
        masses=masses,
        dt=dt,
        n_steps=n_steps,
        save_every=save_every,
        tidal_strength=0.002,
        extra_save_steps=extra_save_steps,
    )


def main() -> None:
    """Run the third-stage cluster experiments and save paper-ready figures."""
    n_particles = 300
    target_times = [0.0, 20.0, 40.0, 80.0]

    initial_positions, initial_velocities, masses = generate_cluster_initial_conditions(
        n_particles=n_particles,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    isolated_times, isolated_energy = run_isolated_energy_test(
        initial_positions,
        initial_velocities,
        masses,
    )
    isolated_energy_path = OUTPUT_DIR / "isolated_cluster_energy_error.png"
    plot_isolated_energy_error(isolated_times, isolated_energy, isolated_energy_path)

    tidal_times, tidal_positions, tidal_velocities, _ = run_tidal_evolution(
        initial_positions,
        initial_velocities,
        masses,
    )
    tidal_evolution_path = OUTPUT_DIR / "cluster_tidal_evolution_comoving.png"
    plot_tidal_evolution_comoving(
        tidal_times,
        tidal_positions,
        masses,
        target_times,
        tidal_evolution_path,
    )

    final_comoving_positions = to_comoving_positions(tidal_positions[-1], masses)
    (
        core_mask,
        leading_mask,
        trailing_mask,
        n_lead,
        n_trail,
        tail_asymmetry,
    ) = classify_tail_candidates(final_comoving_positions)
    tail_classification_path = OUTPUT_DIR / "tail_classification.png"
    plot_tail_classification(
        final_comoving_positions,
        core_mask,
        leading_mask,
        trailing_mask,
        tail_asymmetry,
        tail_classification_path,
    )

    max_isolated_error = float(np.max(relative_energy_error(isolated_energy)))
    final_speed_mean = float(np.mean(np.linalg.norm(tidal_velocities[-1], axis=1)))

    print("Cluster experiments finished.")
    print("Experiment A: isolated cluster energy conservation")
    print("tidal_strength = 0.0, t_end = 20.0, dt = 0.005, save_every = 200")
    print(f"Max relative energy error: {max_isolated_error:.6e}")
    print("Experiment B: gentle tidal evolution")
    print("tidal_strength = 0.002, t_end = 80.0, dt = 0.01, save_every = 1000")
    print(f"Mean final speed: {final_speed_mean:.6f}")
    print(f"N_lead = {n_lead}")
    print(f"N_trail = {n_trail}")
    print(f"A_tail = {tail_asymmetry:.6f}")
    print(f"Saved: {isolated_energy_path}")
    print(f"Saved: {tidal_evolution_path}")
    print(f"Saved: {tail_classification_path}")


if __name__ == "__main__":
    main()
