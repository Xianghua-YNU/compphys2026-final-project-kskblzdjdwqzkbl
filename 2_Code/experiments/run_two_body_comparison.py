"""Compare Euler, RK4, and Velocity Verlet for a two-body orbit.

AI-assisted implementation: this script was drafted with ChatGPT/Codex and
reviewed for the project requirements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_CODE_DIR / "src"
OUTPUT_DIR = PROJECT_CODE_DIR / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from two_body import simulate_two_body  # noqa: E402


def relative_energy_error(energy: np.ndarray) -> np.ndarray:
    """Compute |E(t) - E(0)| / |E(0)| for one simulated orbit."""
    initial_energy = energy[0]
    if initial_energy == 0.0:
        raise ValueError("Initial energy is zero, so relative error is undefined")
    return np.abs(energy - initial_energy) / abs(initial_energy)


def plot_orbits(results: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    """Save the orbit comparison figure."""
    plt.figure(figsize=(7.0, 7.0))
    for method_name, data in results.items():
        position = data["position"]
        plt.plot(position[:, 0], position[:, 1], label=method_name, linewidth=1.2)

    plt.scatter([0.0], [0.0], color="gold", edgecolor="black", s=120, label="Central mass")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Two-body orbit comparison")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_energy_errors(results: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    """Save the relative energy error comparison figure."""
    plt.figure(figsize=(8.0, 5.0))
    for method_name, data in results.items():
        error = relative_energy_error(data["energy"])
        plt.semilogy(data["time"], error, label=method_name, linewidth=1.2)

    plt.xlabel("Time")
    plt.ylabel("Relative energy error")
    plt.title("Energy conservation comparison")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    """Run the first-stage two-body integrator comparison."""
    # Dimensionless units for a simplified Sun-planet system.
    gravitational_constant = 1.0
    central_mass = 1.0
    planet_mass = 1.0e-3

    initial_position = np.array([1.0, 0.0], dtype=float)
    initial_velocity = np.array([0.0, 1.0], dtype=float)
    t_end = 50.0
    dt = 0.01

    methods = {
        "Euler": "euler",
        "RK4": "rk4",
        "Velocity Verlet": "velocity_verlet",
    }

    results = {}
    for label, integrator_name in methods.items():
        results[label] = simulate_two_body(
            integrator=integrator_name,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            t_end=t_end,
            dt=dt,
            central_mass=central_mass,
            planet_mass=planet_mass,
            gravitational_constant=gravitational_constant,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_orbits(results, OUTPUT_DIR / "orbit_comparison.png")
    plot_energy_errors(results, OUTPUT_DIR / "energy_error_comparison.png")

    print("Two-body comparison finished.")
    print(f"Saved: {OUTPUT_DIR / 'orbit_comparison.png'}")
    print(f"Saved: {OUTPUT_DIR / 'energy_error_comparison.png'}")


if __name__ == "__main__":
    main()
