"""Scan tidal strength for simplified open-cluster tail statistics.

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
RANDOM_SEED = 42
X_TAIL_THRESHOLD = 1.0
TIDAL_STRENGTHS = [0.0005, 0.001, 0.002, 0.004]
CLASSIFICATION_STRENGTH = 0.002


def center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return the mass-weighted center of mass for one snapshot."""
    return np.average(positions, axis=0, weights=masses)


def to_comoving_positions(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Shift particle coordinates into the cluster center-of-mass frame."""
    return positions - center_of_mass(positions, masses)


def classify_tail_candidates(
    positions_rel: np.ndarray,
    x_threshold: float = X_TAIL_THRESHOLD,
) -> dict[str, object]:
    """Classify core, leading-tail, and trailing-tail candidates."""
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
        "core_mask": core_mask,
        "leading_mask": leading_mask,
        "trailing_mask": trailing_mask,
        "N_core": n_core,
        "N_lead": n_lead,
        "N_trail": n_trail,
        "A_tail": float(tail_asymmetry),
        "escaped_fraction": float(escaped_fraction),
    }


def run_one_strength(tidal_strength: float) -> tuple[dict[str, float], np.ndarray, dict[str, object]]:
    """Run one tidal-strength simulation and return final tail statistics."""
    initial_positions, initial_velocities, masses = generate_cluster_initial_conditions(
        n_particles=N_PARTICLES,
        random_seed=RANDOM_SEED,
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

    final_positions_rel = to_comoving_positions(position_snapshots[-1], masses)
    classification = classify_tail_candidates(final_positions_rel)
    row = {
        "tidal_strength": float(tidal_strength),
        "N_core": int(classification["N_core"]),
        "N_lead": int(classification["N_lead"]),
        "N_trail": int(classification["N_trail"]),
        "A_tail": float(classification["A_tail"]),
        "escaped_fraction": float(classification["escaped_fraction"]),
    }

    return row, final_positions_rel, classification


def save_scan_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    """Save scan statistics to CSV, using pandas when available."""
    fieldnames = [
        "tidal_strength",
        "N_core",
        "N_lead",
        "N_trail",
        "A_tail",
        "escaped_fraction",
    ]

    if pd is not None:
        pd.DataFrame(rows, columns=fieldnames).to_csv(output_path, index=False)
        return

    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scan_metric(
    rows: list[dict[str, float]],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot one scan metric as a point-line figure."""
    tidal_strengths = np.array([row["tidal_strength"] for row in rows], dtype=float)
    metric_values = np.array([row[metric_key] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(tidal_strengths, metric_values, marker="o", linewidth=1.5)
    ax.set_xlabel("tidal_strength")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def annotation_text(row: dict[str, float]) -> str:
    """Return a compact annotation block for the classification plots."""
    return (
        f"A_tail = {row['A_tail']:.3f}\n"
        f"N_lead = {int(row['N_lead'])}\n"
        f"N_trail = {int(row['N_trail'])}\n"
        f"escaped_fraction = {row['escaped_fraction']:.3f}"
    )


def plot_tail_classification(
    positions_rel: np.ndarray,
    classification: dict[str, object],
    row: dict[str, float],
    output_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot final tail classes for one tidal-strength simulation."""
    core_mask = classification["core_mask"]
    leading_mask = classification["leading_mask"]
    trailing_mask = classification["trailing_mask"]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(
        positions_rel[core_mask, 0],
        positions_rel[core_mask, 1],
        s=12,
        alpha=0.70,
        label="core candidate",
    )
    ax.scatter(
        positions_rel[leading_mask, 0],
        positions_rel[leading_mask, 1],
        s=28,
        marker=">",
        alpha=0.85,
        label="leading tail candidate",
    )
    ax.scatter(
        positions_rel[trailing_mask, 0],
        positions_rel[trailing_mask, 1],
        s=28,
        marker="<",
        alpha=0.85,
        label="trailing tail candidate",
    )

    ax.axvline(X_TAIL_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axvline(-X_TAIL_THRESHOLD, color="black", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("x - x_cm [model units]")
    ax.set_ylabel("y - y_cm [model units]")
    ax.set_title(f"Tail classification at tidal_strength = {row['tidal_strength']:.4g}")
    ax.text(
        0.02,
        0.96,
        annotation_text(row),
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the tidal-strength scan and save CSV plus paper-ready figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    classification_positions: np.ndarray | None = None
    classification_masks: dict[str, object] | None = None
    classification_row: dict[str, float] | None = None

    for tidal_strength in TIDAL_STRENGTHS:
        row, positions_rel, classification = run_one_strength(tidal_strength)
        rows.append(row)
        print(
            f"tidal_strength={tidal_strength:.4g}: "
            f"N_core={row['N_core']}, "
            f"N_lead={row['N_lead']}, "
            f"N_trail={row['N_trail']}, "
            f"A_tail={row['A_tail']:.6f}, "
            f"escaped_fraction={row['escaped_fraction']:.6f}"
        )

        if np.isclose(tidal_strength, CLASSIFICATION_STRENGTH):
            classification_positions = positions_rel
            classification_masks = classification
            classification_row = row

    csv_path = OUTPUT_DIR / "tidal_strength_scan.csv"
    a_tail_path = OUTPUT_DIR / "A_tail_vs_tidal_strength.png"
    escaped_path = OUTPUT_DIR / "escaped_fraction_vs_tidal_strength.png"
    full_path = OUTPUT_DIR / "tail_classification_full.png"
    zoom_path = OUTPUT_DIR / "tail_classification_zoom.png"

    save_scan_csv(rows, csv_path)
    plot_scan_metric(
        rows,
        metric_key="A_tail",
        y_label="A_tail",
        title="Tail asymmetry versus tidal strength",
        output_path=a_tail_path,
    )
    plot_scan_metric(
        rows,
        metric_key="escaped_fraction",
        y_label="escaped_fraction",
        title="Escaped fraction versus tidal strength",
        output_path=escaped_path,
    )

    if classification_positions is None or classification_masks is None or classification_row is None:
        raise RuntimeError("Classification strength result was not found in the scan")

    plot_tail_classification(
        classification_positions,
        classification_masks,
        classification_row,
        full_path,
    )
    plot_tail_classification(
        classification_positions,
        classification_masks,
        classification_row,
        zoom_path,
        xlim=(-15.0, 15.0),
        ylim=(-8.0, 8.0),
    )

    print("Tidal strength scan finished.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {a_tail_path}")
    print(f"Saved: {escaped_path}")
    print(f"Saved: {full_path}")
    print(f"Saved: {zoom_path}")


if __name__ == "__main__":
    main()
