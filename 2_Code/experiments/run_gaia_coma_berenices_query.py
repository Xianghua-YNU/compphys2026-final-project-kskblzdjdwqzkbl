"""Query lightweight Gaia DR3 candidate members around Coma Berenices.

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
import pandas as pd


PROJECT_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "3_Data"
OUTPUT_DIR = PROJECT_CODE_DIR / "outputs"

CANDIDATE_CSV = DATA_DIR / "coma_berenices_gaia_dr3_candidates.csv"

RA_CENTER = 186.8110
DEC_CENTER = 25.8112
RADIUS_DEG = 5.0
COMA_PARALLAX_MAS = 11.65

QUERY_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "bp_rp",
    "ruwe",
]


def build_candidate_query() -> str:
    """Return the lightweight ADQL query for candidate members only."""
    columns = ",\n        ".join(QUERY_COLUMNS)
    return f"""
    SELECT TOP 2000
        {columns}
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {RA_CENTER}, {DEC_CENTER}, {RADIUS_DEG})
    )
    AND parallax BETWEEN 10.5 AND 12.8
    AND parallax_error < 0.5
    AND pmra BETWEEN -15 AND 0
    AND pmdec BETWEEN -15 AND 5
    AND phot_g_mean_mag < 18
    AND bp_rp IS NOT NULL
    AND bp_rp BETWEEN -0.2 AND 4.0
    AND ruwe < 1.4
    """


def angular_separation_deg(dataframe: pd.DataFrame) -> np.ndarray:
    """Compute angular separation from the adopted Coma Berenices center."""
    ra = np.deg2rad(dataframe["ra"].to_numpy(dtype=float))
    dec = np.deg2rad(dataframe["dec"].to_numpy(dtype=float))
    ra0 = np.deg2rad(RA_CENTER)
    dec0 = np.deg2rad(DEC_CENTER)

    cos_sep = (
        np.sin(dec0) * np.sin(dec)
        + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    )
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_sep))


def apply_local_candidate_filter(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reapply the ADQL box cuts locally for cached or manually downloaded CSVs."""
    filtered = dataframe[QUERY_COLUMNS].copy()
    mask = (
        (angular_separation_deg(filtered) <= RADIUS_DEG)
        & (filtered["parallax"].between(10.5, 12.8))
        & (filtered["parallax_error"] < 0.5)
        & (filtered["pmra"].between(-15.0, 0.0))
        & (filtered["pmdec"].between(-15.0, 5.0))
        & (filtered["phot_g_mean_mag"] < 18.0)
        & (filtered["bp_rp"].notna())
        & (filtered["bp_rp"].between(-0.2, 4.0))
        & (filtered["ruwe"] < 1.4)
    )
    return filtered.loc[mask].reset_index(drop=True)


def query_candidate_members() -> pd.DataFrame:
    """Run one synchronous Gaia query for the candidate-member sample."""
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is required for Gaia queries. Install dependencies with "
            "`pip install -r 2_Code/requirements.txt`."
        ) from exc

    print("Submitting lightweight Gaia DR3 candidate query...")
    job = Gaia.launch_job(build_candidate_query(), dump_to_file=False)
    table = job.get_results()
    dataframe = table.to_pandas()
    return dataframe[QUERY_COLUMNS]


def load_or_query_candidates() -> pd.DataFrame:
    """Query candidates; if the network fails, explain the manual CSV fallback."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        candidates = query_candidate_members()
        candidates = apply_local_candidate_filter(candidates)
        candidates.to_csv(CANDIDATE_CSV, index=False)
        print(f"Saved candidate CSV: {CANDIDATE_CSV}")
        return candidates
    except Exception as exc:
        message = (
            "Gaia candidate query failed. Manually download the ADQL result from "
            "Gaia Archive and save it as "
            f"{CANDIDATE_CSV}, then rerun this script to generate the figures."
        )
        print(message, file=sys.stderr)
        if CANDIDATE_CSV.exists():
            print(f"Using existing candidate CSV after local validation: {CANDIDATE_CSV}")
            cached = pd.read_csv(CANDIDATE_CSV)
            return apply_local_candidate_filter(cached)
        raise RuntimeError(message) from exc


def plot_sky_distribution(candidates: pd.DataFrame, output_path: Path) -> None:
    """Plot sky positions of candidate members."""
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(
        candidates["ra"],
        candidates["dec"],
        s=18,
        color="tab:blue",
        alpha=0.85,
        linewidths=0,
        label="Candidate members",
    )
    ax.scatter(
        [RA_CENTER],
        [DEC_CENTER],
        marker="*",
        s=180,
        color="tab:red",
        edgecolor="black",
        linewidth=0.5,
        label="Literature center",
    )
    ax.invert_xaxis()
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Gaia DR3 candidate distribution around Coma Berenices")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pm_parallax(candidates: pd.DataFrame, output_path: Path) -> None:
    """Plot proper motions and parallax histogram for candidate members."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    axes[0].scatter(
        candidates["pmra"],
        candidates["pmdec"],
        s=18,
        color="tab:blue",
        alpha=0.85,
        linewidths=0,
        label="Candidate members",
    )
    axes[0].set_xlabel("pmra [mas/yr]")
    axes[0].set_ylabel("pmdec [mas/yr]")
    axes[0].set_title("Proper-motion candidate selection")
    axes[0].grid(True, linestyle="--", alpha=0.30)
    axes[0].legend(loc="best")

    axes[1].hist(
        candidates["parallax"],
        bins=20,
        color="tab:blue",
        alpha=0.75,
        label="Candidate members",
    )
    axes[1].axvline(
        COMA_PARALLAX_MAS,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Coma Ber parallax ~ 11.65 mas",
    )
    axes[1].set_xlabel("parallax [mas]")
    axes[1].set_ylabel("Number of stars")
    axes[1].set_title("Candidate parallax distribution")
    axes[1].grid(True, linestyle="--", alpha=0.30)
    axes[1].legend(loc="best")

    fig.suptitle("Gaia DR3 proper motion and parallax candidates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cmd(candidates: pd.DataFrame, output_path: Path) -> None:
    """Plot the color-magnitude diagram for candidate members."""
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    ax.scatter(
        candidates["bp_rp"],
        candidates["phot_g_mean_mag"],
        s=18,
        color="tab:blue",
        alpha=0.85,
        linewidths=0,
        label="Candidate members",
    )
    ax.invert_yaxis()
    ax.set_xlabel("bp_rp [mag]")
    ax.set_ylabel("phot_g_mean_mag [mag]")
    ax.set_title("Color-magnitude diagram of candidate members")
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Query candidate members and save Gaia comparison plots."""
    candidates = load_or_query_candidates()

    sky_path = OUTPUT_DIR / "coma_berenices_sky_candidates.png"
    pm_parallax_path = OUTPUT_DIR / "coma_berenices_pm_parallax.png"
    cmd_path = OUTPUT_DIR / "coma_berenices_cmd.png"

    plot_sky_distribution(candidates, sky_path)
    plot_pm_parallax(candidates, pm_parallax_path)
    plot_cmd(candidates, cmd_path)

    print("Gaia DR3 Coma Berenices candidate comparison finished.")
    print("Candidate selection is a simple course-project box cut, not a membership probability.")
    print(f"Candidate size: {len(candidates)}")
    print(f"Saved: {CANDIDATE_CSV}")
    print(f"Saved: {sky_path}")
    print(f"Saved: {pm_parallax_path}")
    print(f"Saved: {cmd_path}")


if __name__ == "__main__":
    main()
