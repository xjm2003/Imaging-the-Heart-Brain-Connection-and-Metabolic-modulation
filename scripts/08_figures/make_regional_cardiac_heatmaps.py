#!/usr/bin/env python3
"""Compute regional brain-cardiac Spearman associations and heatmaps.

The script analyzes regional FA and normalized regional volume against a
predefined panel of cardiac phenotypes. Canonical cardiac variables are taken
from the master table and replace stale cardiac columns in the brain-feature
file. Publication-style stars use global modality-wide BH-FDR.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

CARDIAC_ORDER = [
    "cardiac_Diastolic_LV_Volume",
    "cardiac_Systolic_LV_Volume",
    "cardiac_Stroke_Volume",
    "cardiac_Ejection_Fraction",
    "cardiac_Cardiac_Output",
    "cardiac_Diastolic_RV",
    "cardiac_Systolic_RV",
    "cardiac_RV_Stroke_Volume",
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LA",
    "cardiac_Diastolic_RA",
    "cardiac_Systolic_RA",
    "cardiac_Diastolic_Myo",
    "cardiac_Systolic_Myo",
    "cardiac_MYO_MASS",
    "cardiac_Heart_Rate",
]

CARDIAC_LABELS = {
    "cardiac_Diastolic_LV_Volume": "LV EDV",
    "cardiac_Systolic_LV_Volume": "LV ESV",
    "cardiac_Stroke_Volume": "LV SV",
    "cardiac_Ejection_Fraction": "EF",
    "cardiac_Cardiac_Output": "CO",
    "cardiac_Diastolic_RV": "RV EDV",
    "cardiac_Systolic_RV": "RV ESV",
    "cardiac_RV_Stroke_Volume": "RV SV",
    "cardiac_Diastolic_LA": "LA diastolic",
    "cardiac_Systolic_LA": "LA systolic",
    "cardiac_Diastolic_RA": "RA diastolic",
    "cardiac_Systolic_RA": "RA systolic",
    "cardiac_Diastolic_Myo": "Myo diastolic",
    "cardiac_Systolic_Myo": "Myo systolic",
    "cardiac_MYO_MASS": "Myo mass",
    "cardiac_Heart_Rate": "Heart rate",
}

CARDIAC_GROUPS = [
    ("LV", 0, 5),
    ("RV", 5, 8),
    ("Atria", 8, 12),
    ("Myocardium", 12, 15),
    ("Rate", 15, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-features", required=True, type=Path)
    parser.add_argument("--master-table", required=True, type=Path)
    parser.add_argument("--roi-mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-roi", type=int, default=28)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--vmax", type=float, default=0.8)
    return parser.parse_args()


def detect_id_column(df: pd.DataFrame, label: str) -> str:
    for candidate in ["mouse_id", "ID", "ARunno", "Mouse_ID", "subject_id"]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"No mouse ID column detected in {label}.")


def bh_fdr(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty(m)
    restored[order] = adjusted
    q[valid] = restored
    return q


def extract_roi_index(column: str) -> int | None:
    """Map modality-specific ROI labels to common sequential indices 1-332."""
    column = str(column)
    match = re.search(r"ROI[_-]?(\d+)", column, flags=re.I)
    if match is None:
        return None
    label = int(match.group(1))

    if column.startswith(("vol_norm__", "vol_abs__")):
        return label if 1 <= label <= 332 else None

    if column.startswith(("FA__", "MD__", "AD__", "RD__")):
        if 1 <= label <= 166:
            return label
        if 1001 <= label <= 1166:
            return label - 834
        return None

    return label if 1 <= label <= 332 else None


def global_star(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    brain = pd.read_csv(args.brain_features)
    master = pd.read_csv(args.master_table)
    roi_map = pd.read_csv(args.roi_mapping)

    brain_id = detect_id_column(brain, "brain features")
    master_id = detect_id_column(master, "master table")
    brain = brain.rename(columns={brain_id: "mouse_id"})
    master = master.rename(columns={master_id: "mouse_id"})
    brain["mouse_id"] = brain["mouse_id"].astype(str).str.strip()
    master["mouse_id"] = master["mouse_id"].astype(str).str.strip()

    if brain["mouse_id"].duplicated().any() or master["mouse_id"].duplicated().any():
        raise ValueError("Mouse IDs must be unique in both input tables.")
    if set(brain["mouse_id"]) != set(master["mouse_id"]):
        raise ValueError("Brain-feature and master tables contain different mice.")

    stale = [
        column for column in brain.columns
        if column.startswith("cardiac_")
        or column in {
            "Cardiac_ID", "Cardiac_ID_norm", "cardiac_table_ID",
            "cardiac_table_Animal_Code", "Weight",
        }
    ]
    canonical = [
        column for column in master.columns
        if column.startswith("cardiac_")
        or column in {
            "Cardiac_ID", "Cardiac_ID_norm", "cardiac_table_ID",
            "cardiac_table_Animal_Code", "Weight",
        }
    ]
    data = brain.drop(columns=stale, errors="ignore").merge(
        master[["mouse_id"] + canonical],
        on="mouse_id",
        how="left",
        validate="one_to_one",
    )

    roi_map["ROI_index"] = pd.to_numeric(
        roi_map["ROI_index"], errors="raise"
    ).astype(int)
    if len(roi_map) != 332 or roi_map["ROI_index"].nunique() != 332:
        raise ValueError("ROI mapping must contain 332 unique ROI indices.")

    if "structure_display_name" not in roi_map.columns:
        roi_map["structure_display_name"] = (
            roi_map["atlas_abbrev"].astype(str)
            + " — "
            + roi_map["anatomical_name"].astype(str)
        )
    roi_map["hemisphere_short"] = roi_map["hemisphere"].map(
        {"hemisphere_block_A": "A", "hemisphere_block_B": "B"}
    )
    roi_map["roi_display_name"] = (
        roi_map["hemisphere_short"].astype(str)
        + " · "
        + roi_map["structure_display_name"].astype(str)
    )
    return data, roi_map


def calculate_associations(
    data: pd.DataFrame,
    roi_map: pd.DataFrame,
    feature_columns: list[str],
    cardiac_columns: list[str],
    modality: str,
    min_n: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for feature in feature_columns:
        roi_index = extract_roi_index(feature)
        x_all = pd.to_numeric(data[feature], errors="coerce")
        for outcome in cardiac_columns:
            y_all = pd.to_numeric(data[outcome], errors="coerce")
            valid = x_all.notna() & y_all.notna()
            n = int(valid.sum())
            rho = np.nan
            p_value = np.nan
            if (
                n >= min_n
                and x_all[valid].nunique() >= 3
                and y_all[valid].nunique() >= 3
            ):
                rho, p_value = spearmanr(x_all[valid], y_all[valid])
            records.append(
                {
                    "modality": modality,
                    "feature_column": feature,
                    "ROI_index": roi_index,
                    "cardiac_feature": outcome,
                    "cardiac_label": CARDIAC_LABELS.get(outcome, outcome),
                    "n": n,
                    "rho": rho,
                    "p_value": p_value,
                }
            )

    result = pd.DataFrame(records)
    result["q_global_modality"] = bh_fdr(result["p_value"])
    result["q_by_outcome"] = np.nan
    for _, indices in result.groupby("cardiac_feature").groups.items():
        result.loc[indices, "q_by_outcome"] = bh_fdr(
            result.loc[indices, "p_value"]
        )

    metadata = [
        "ROI_index", "raw_atlas_label_id", "hemisphere", "hemisphere_short",
        "atlas_abbrev", "anatomical_name", "structure_display_name",
        "roi_display_name",
    ]
    result = result.merge(
        roi_map[metadata], on="ROI_index", how="left", validate="many_to_one"
    )
    result["abs_rho"] = result["rho"].abs()
    result["global_star"] = result["q_global_modality"].map(global_star)
    return result


def select_rois(
    result: pd.DataFrame,
    cardiac_columns: list[str],
    n_roi: int,
) -> list[int]:
    selected: set[int] = set()
    for outcome in cardiac_columns:
        subset = result.loc[
            result["cardiac_feature"].eq(outcome) & result["rho"].notna()
        ].sort_values(
            ["q_global_modality", "q_by_outcome", "p_value", "abs_rho"],
            ascending=[True, True, True, False],
        )
        selected.update(subset.head(2)["ROI_index"].astype(int))

    ranking = (
        result.groupby("ROI_index", as_index=False)
        .agg(
            min_global_q=("q_global_modality", "min"),
            min_outcome_q=("q_by_outcome", "min"),
            min_p=("p_value", "min"),
            max_abs_rho=("abs_rho", "max"),
        )
        .sort_values(
            ["min_global_q", "min_outcome_q", "min_p", "max_abs_rho"],
            ascending=[True, True, True, False],
        )
    )
    for roi_index in ranking["ROI_index"]:
        selected.add(int(roi_index))
        if len(selected) >= n_roi:
            break

    return (
        ranking.loc[ranking["ROI_index"].isin(selected)]
        .head(n_roi)["ROI_index"]
        .astype(int)
        .tolist()
    )


def cluster_rows(matrix: pd.DataFrame) -> list[int]:
    values = matrix.fillna(0).to_numpy(dtype=float)
    if len(matrix) < 3:
        return matrix.index.tolist()
    try:
        distance = pdist(values, metric="correlation")
        if not np.all(np.isfinite(distance)):
            raise ValueError
        tree = linkage(distance, method="average")
    except Exception:
        tree = linkage(values, method="average", metric="euclidean")
    return matrix.index.to_numpy()[leaves_list(tree)].tolist()


def shorten(value: str, limit: int = 48) -> str:
    value = str(value).replace("_", " ").strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def draw_heatmap(
    result: pd.DataFrame,
    cardiac_columns: list[str],
    title: str,
    output_stem: Path,
    n_roi: int,
    vmax: float,
) -> plt.Figure:
    selected = select_rois(result, cardiac_columns, n_roi)
    rho = result.pivot(
        index="ROI_index", columns="cardiac_feature", values="rho"
    ).reindex(index=selected, columns=cardiac_columns)
    stars = result.pivot(
        index="ROI_index", columns="cardiac_feature", values="global_star"
    ).reindex(index=selected, columns=cardiac_columns)

    order = cluster_rows(rho)
    rho = rho.loc[order]
    stars = stars.loc[order]
    metadata = result.drop_duplicates("ROI_index").set_index("ROI_index")
    labels = [
        shorten(metadata.loc[index, "roi_display_name"])
        for index in order
    ]
    rho.index = labels
    stars.index = labels
    rho.columns = [CARDIAC_LABELS.get(column, column) for column in rho.columns]
    stars.columns = rho.columns

    height = max(10.5, 0.34 * len(rho) + 3.0)
    fig, ax = plt.subplots(figsize=(14.5, height))
    sns.heatmap(
        rho,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=stars,
        fmt="",
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        linewidths=0.30,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.72, "pad": 0.025},
    )

    for group_name, start, end in CARDIAC_GROUPS:
        ax.text(
            (start + end) / 2,
            1.025,
            group_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
        if end < len(cardiac_columns):
            ax.axvline(end, color="black", linewidth=1.0)

    ax.set_title(
        f"{title}\nTop {len(rho)} ROIs; stars indicate global BH-FDR significance",
        fontsize=15,
        fontweight="bold",
        pad=30,
    )
    ax.set_xlabel("Cardiac phenotype")
    ax.set_ylabel("Atlas ROI")
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8.2)
    fig.text(
        0.01,
        0.008,
        "* qglobal < 0.05; ** qglobal < 0.01; *** qglobal < 0.001. "
        "A/B denote atlas hemisphere blocks; anatomical left/right remains unresolved.",
        fontsize=8,
        ha="left",
    )
    plt.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    result.loc[result["ROI_index"].isin(selected)].to_csv(
        output_stem.with_name(output_stem.name + "_selected_associations.csv"),
        index=False,
    )
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, roi_map = load_data(args)

    cardiac_columns = [column for column in CARDIAC_ORDER if column in data.columns]
    if len(cardiac_columns) != len(CARDIAC_ORDER):
        missing = [column for column in CARDIAC_ORDER if column not in data.columns]
        raise ValueError(f"Missing cardiac columns: {missing}")

    fa_columns = [
        column for column in data.columns
        if column.startswith("FA__") and extract_roi_index(column) is not None
    ]
    volume_columns = [
        column for column in data.columns
        if column.startswith("vol_norm__") and extract_roi_index(column) is not None
    ]
    if len(fa_columns) != 332:
        raise ValueError(f"Expected 332 FA columns, found {len(fa_columns)}")
    if len(volume_columns) != 332:
        raise ValueError(
            f"Expected 332 normalized-volume columns, found {len(volume_columns)}"
        )

    print(f"Mice: {data['mouse_id'].nunique()}")
    print(f"Regional FA columns: {len(fa_columns)}")
    print(f"Normalized-volume columns: {len(volume_columns)}")
    for outcome in cardiac_columns:
        n = pd.to_numeric(data[outcome], errors="coerce").notna().sum()
        print(f"{outcome}: n={n}")

    fa = calculate_associations(
        data, roi_map, fa_columns, cardiac_columns, "regional_FA", args.min_n
    )
    volume = calculate_associations(
        data,
        roi_map,
        volume_columns,
        cardiac_columns,
        "normalized_regional_volume",
        args.min_n,
    )
    fa.to_csv(args.output_dir / "regional_FA_cardiac_spearman_named.csv", index=False)
    volume.to_csv(
        args.output_dir / "regional_volume_cardiac_spearman_named.csv", index=False
    )

    sns.set_theme(context="paper", style="white", font_scale=0.95)
    fa_fig = draw_heatmap(
        fa,
        cardiac_columns,
        "Regional fractional anisotropy × cardiac phenotypes",
        args.output_dir / "regional_FA_cardiac_publication",
        args.n_roi,
        args.vmax,
    )
    volume_fig = draw_heatmap(
        volume,
        cardiac_columns,
        "Normalized regional brain volume × cardiac phenotypes",
        args.output_dir / "regional_volume_cardiac_publication",
        args.n_roi,
        args.vmax,
    )

    paired = args.output_dir / "regional_FA_volume_cardiac_paired.pdf"
    with PdfPages(paired) as pdf:
        pdf.savefig(fa_fig, bbox_inches="tight")
        pdf.savefig(volume_fig, bbox_inches="tight")
    plt.close(fa_fig)
    plt.close(volume_fig)
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
