"""Step-size convergence test for Velocity Verlet in a two-body orbit.

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

from two_body import simulate_two_body, total_energy  # noqa: E402


def relative_energy_error(energy: np.ndarray, initial_energy: float) -> np.ndarray:
    """Compute |E(t) - E(0)| / |E(0)| for one time-step experiment."""
    if initial_energy == 0.0:
        raise ValueError("Initial energy is zero, so relative error is undefined")
    return np.abs(energy - initial_energy) / abs(initial_energy)


def plot_stepsize_convergence(
    results: dict[float, dict[str, np.ndarray]],
    initial_energy: float,
    output_path: Path,
) -> None:
    """Save the Velocity Verlet step-size convergence figure."""
    plt.figure(figsize=(8.0, 5.0))
    for dt, data in results.items():
        error = relative_energy_error(data["energy"], initial_energy)
        plt.semilogy(data["time"], error, label=f"dt = {dt:g}", linewidth=1.2)

    plt.xlabel("Time")
    plt.ylabel("Relative energy error")
    plt.title("Step-size convergence of Velocity Verlet")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    """Run the Velocity Verlet time-step convergence experiment."""
    gravitational_constant = 1.0
    central_mass = 1.0
    planet_mass = 1.0e-3

    initial_position = np.array([1.0, 0.0], dtype=float)
    initial_velocity = np.array([0.0, 1.0], dtype=float)
    t_end = 50.0
    time_steps = [0.04, 0.02, 0.01, 0.005]

    initial_energy = total_energy(
        initial_position,
        initial_velocity,
        central_mass=central_mass,
        planet_mass=planet_mass,
        gravitational_constant=gravitational_constant,
    )

    results = {}
    for dt in time_steps:
        results[dt] = simulate_two_body(
            integrator="velocity_verlet",
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            t_end=t_end,
            dt=dt,
            central_mass=central_mass,
            planet_mass=planet_mass,
            gravitational_constant=gravitational_constant,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "verlet_stepsize_convergence.png"
    plot_stepsize_convergence(results, initial_energy, output_path)

    print("Velocity Verlet step-size convergence finished.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
