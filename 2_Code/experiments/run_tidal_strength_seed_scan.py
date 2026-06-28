"""Multi-seed tidal-strength scan for open-cluster tail statistics.

AI-assisted implementation: this script was drafted with ChatGPT/Codex and
reviewed for the project requirements.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - fallback for minimal environments.
    pd = None


PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_CODE_DIR / "src"
OUTPUT_DIR = PROJECT_CODE_DIR / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nbody_cluster import (  # noqa: E402
    generate_cluster_initial_conditions,
    velocity_verlet_cluster,
)


N_PARTICLES = 300
DT = 0.01
T_END = 80.0
SAVE_EVERY = 1000
SOFTENING = 0.08
X_TAIL_THRESHOLD = 1.0
TIDAL_STRENGTHS = [0.0005, 0.001, 0.002, 0.004]
SEEDS = range(10)


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return the mass-weighted center of mass for one snapshot."""
    return np.average(positions, axis=0, weights=masses)


def to_comoving_positions(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Shift coordinates into the center-of-mass frame."""
    return positions - center_of_mass(positions, masses)


def classify_tail_candidates(
    positions_rel: np.ndarray,
    x_threshold: float = X_TAIL_THRESHOLD,
) -> dict[str, float]:
    """Count core, leading-tail, and trailing-tail candidates."""
    leading_mask = positions_rel[:, 0] > x_threshold
    trailing_mask = positions_rel[:, 0] < -x_threshold
    core_mask = ~(leading_mask | trailing_mask)

    n_lead = int(np.sum(leading_mask))
    n_trail = int(np.sum(trailing_mask))
    n_core = int(np.sum(core_mask))
    n_tail = n_lead + n_trail

    if n_tail == 0:
        tail_asymmetry = 0.0
    else:
        tail_asymmetry = (n_lead - n_trail) / n_tail

    escaped_fraction = n_tail / positions_rel.shape[0]

    return {
        "N_core": n_core,
        "N_lead": n_lead,
        "N_trail": n_trail,
        "A_tail": float(tail_asymmetry),
        "escaped_fraction": float(escaped_fraction),
    }


def run_one_simulation(tidal_strength: float, seed: int) -> dict[str, float]:
    """Run one seeded simulation and return final tail statistics."""
    initial_positions, initial_velocities, masses = generate_cluster_initial_conditions(
        n_particles=N_PARTICLES,
        random_seed=seed,
    )

    n_steps = int(round(T_END / DT))
    _, position_snapshots, _, _ = velocity_verlet_cluster(
        positions=initial_positions,
        velocities=initial_velocities,
        masses=masses,
        dt=DT,
        n_steps=n_steps,
        save_every=SAVE_EVERY,
        softening=SOFTENING,
        tidal_strength=tidal_strength,
    )

    positions_rel = to_comoving_positions(position_snapshots[-1], masses)
    stats = classify_tail_candidates(positions_rel)

    return {
        "tidal_strength": float(tidal_strength),
        "seed": int(seed),
        "N_core": int(stats["N_core"]),
        "N_lead": int(stats["N_lead"]),
        "N_trail": int(stats["N_trail"]),
        "A_tail": float(stats["A_tail"]),
        "escaped_fraction": float(stats["escaped_fraction"]),
    }


def summarize_by_strength(raw_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Compute mean and standard deviation for each tidal strength."""
    summary_rows: list[dict[str, float]] = []

    for tidal_strength in TIDAL_STRENGTHS:
        selected = [
            row for row in raw_rows if np.isclose(row["tidal_strength"], tidal_strength)
        ]
        a_tail_values = np.array([row["A_tail"] for row in selected], dtype=float)
        escaped_values = np.array([row["escaped_fraction"] for row in selected], dtype=float)

        summary_rows.append(
            {
                "tidal_strength": float(tidal_strength),
                "mean_A_tail": float(np.mean(a_tail_values)),
                "std_A_tail": float(np.std(a_tail_values, ddof=1)),
                "mean_escaped_fraction": float(np.mean(escaped_values)),
                "std_escaped_fraction": float(np.std(escaped_values, ddof=1)),
            }
        )

    return summary_rows


def save_csv(rows: list[dict[str, float]], output_path: Path, fieldnames: list[str]) -> None:
    """Save rows to CSV, using pandas when available."""
    if pd is not None:
        pd.DataFrame(rows, columns=fieldnames).to_csv(output_path, index=False)
        return

    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_errorbar_metric(
    summary_rows: list[dict[str, float]],
    mean_key: str,
    std_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot mean values with one-standard-deviation error bars."""
    tidal_strengths = np.array([row["tidal_strength"] for row in summary_rows], dtype=float)
    mean_values = np.array([row[mean_key] for row in summary_rows], dtype=float)
    std_values = np.array([row[std_key] for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(
        tidal_strengths,
        mean_values,
        yerr=std_values,
        fmt="o-",
        linewidth=1.5,
        markersize=5.5,
        capsize=4,
    )
    ax.set_xlabel("tidal_strength")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the multi-seed scan and save raw data, summaries, and figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, float]] = []
    total_runs = len(TIDAL_STRENGTHS) * len(list(SEEDS))
    run_index = 0

    for tidal_strength in TIDAL_STRENGTHS:
        for seed in SEEDS:
            run_index += 1
            print(
                f"[{run_index:02d}/{total_runs}] "
                f"tidal_strength={tidal_strength:.4g}, seed={seed}",
                flush=True,
            )
            row = run_one_simulation(tidal_strength, seed)
            raw_rows.append(row)
            print(
                f"    N_core={row['N_core']}, "
                f"N_lead={row['N_lead']}, "
                f"N_trail={row['N_trail']}, "
                f"A_tail={row['A_tail']:.6f}, "
                f"escaped_fraction={row['escaped_fraction']:.6f}",
                flush=True,
            )

    summary_rows = summarize_by_strength(raw_rows)

    raw_fieldnames = [
        "tidal_strength",
        "seed",
        "N_core",
        "N_lead",
        "N_trail",
        "A_tail",
        "escaped_fraction",
    ]
    summary_fieldnames = [
        "tidal_strength",
        "mean_A_tail",
        "std_A_tail",
        "mean_escaped_fraction",
        "std_escaped_fraction",
    ]

    raw_csv_path = OUTPUT_DIR / "tidal_strength_seed_scan_raw.csv"
    summary_csv_path = OUTPUT_DIR / "tidal_strength_seed_scan_summary.csv"
    a_tail_plot_path = OUTPUT_DIR / "A_tail_vs_tidal_strength_errorbar.png"
    escaped_plot_path = OUTPUT_DIR / "escaped_fraction_vs_tidal_strength_errorbar.png"

    save_csv(raw_rows, raw_csv_path, raw_fieldnames)
    save_csv(summary_rows, summary_csv_path, summary_fieldnames)
    plot_errorbar_metric(
        summary_rows,
        mean_key="mean_A_tail",
        std_key="std_A_tail",
        y_label="mean_A_tail",
        title="Tail asymmetry versus tidal strength",
        output_path=a_tail_plot_path,
    )
    plot_errorbar_metric(
        summary_rows,
        mean_key="mean_escaped_fraction",
        std_key="std_escaped_fraction",
        y_label="mean_escaped_fraction",
        title="Escaped fraction versus tidal strength",
        output_path=escaped_plot_path,
    )

    print("Multi-seed tidal strength scan finished.", flush=True)
    print(f"Saved: {raw_csv_path}", flush=True)
    print(f"Saved: {summary_csv_path}", flush=True)
    print(f"Saved: {a_tail_plot_path}", flush=True)
    print(f"Saved: {escaped_plot_path}", flush=True)


if __name__ == "__main__":
    main()
