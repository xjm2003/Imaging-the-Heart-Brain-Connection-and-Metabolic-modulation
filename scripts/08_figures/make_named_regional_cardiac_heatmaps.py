from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


# =============================================================================
# Paths
# =============================================================================

ROOT = Path("/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1")

TABLE_DIR = (
    ROOT
    / "analysis_ready_regional_FA_volume_20260624_plus_B25122904"
    / "tables"
)

BRAIN_FILE = (
    TABLE_DIR
    / "advanced_brain_heart_visuals"
    / "advanced_brain_heart_merged_features.csv"
)

MASTER_FILE = Path(
    "/mnt/newStor/paros/paros_WORK/jimin/data+/"
    "GLP1_26/data/metadata/GLP1_MRI.csv"
)

ROI_MAPPING_FILE = (
    ROOT
    / "analysis_ready_regional_FA_volume_20260624_plus_B25122904"
    / "docs"
    / "ROI_mapping_332_named.csv"
)

OUT = TABLE_DIR / "regional_cardiac_heatmaps_named"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Settings
# =============================================================================

MIN_N = 20

# 每个 cardiac outcome 至少提取多少个候选 ROI
TOP_PER_OUTCOME = 4

# Top heatmap 最多显示多少个 ROI
MAX_TOP_ROWS = 45

# 至少显示多少个 ROI
MIN_TOP_ROWS = 30

# FDR 星号使用 q_by_outcome
STAR_Q = {
    0.001: "***",
    0.01: "**",
    0.05: "*",
    0.10: "·",
}


CARDIAC_ORDER = [
    "cardiac_Diastolic_LV_Volume",
    "cardiac_Systolic_LV_Volume",
    "cardiac_Diastolic_RV",
    "cardiac_Systolic_RV",
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LA",
    "cardiac_Diastolic_RA",
    "cardiac_Systolic_RA",
    "cardiac_Diastolic_Myo",
    "cardiac_Systolic_Myo",
    "cardiac_Stroke_Volume",
    "cardiac_RV_Stroke_Volume",
    "cardiac_Ejection_Fraction",
    "cardiac_Cardiac_Output",
    "cardiac_Heart_Rate",
    "cardiac_MYO_MASS",
]


CARDIAC_LABELS = {
    "cardiac_Diastolic_LV_Volume": "LV EDV",
    "cardiac_Systolic_LV_Volume": "LV ESV",
    "cardiac_Diastolic_RV": "RV EDV",
    "cardiac_Systolic_RV": "RV ESV",
    "cardiac_Diastolic_LA": "LA diastolic",
    "cardiac_Systolic_LA": "LA systolic",
    "cardiac_Diastolic_RA": "RA diastolic",
    "cardiac_Systolic_RA": "RA systolic",
    "cardiac_Diastolic_Myo": "Myo diastolic",
    "cardiac_Systolic_Myo": "Myo systolic",
    "cardiac_Stroke_Volume": "LV stroke volume",
    "cardiac_RV_Stroke_Volume": "RV stroke volume",
    "cardiac_Ejection_Fraction": "Ejection fraction",
    "cardiac_Cardiac_Output": "Cardiac output",
    "cardiac_Heart_Rate": "Heart rate",
    "cardiac_MYO_MASS": "Myocardial mass",
}


# =============================================================================
# Utilities
# =============================================================================

def detect_id_column(df: pd.DataFrame, name: str) -> str:
    candidates = [
        "mouse_id",
        "ID",
        "ARunno",
        "Mouse_ID",
        "subject_id",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        f"Cannot identify mouse ID column in {name}. "
        f"First columns: {df.columns[:20].tolist()}"
    )


def bh_fdr(values) -> np.ndarray:
    """Benjamini-Hochberg adjustment with NaN preservation."""

    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

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
    """
    Convert feature-column ROI labels to the common ROI_index 1–332.

    Regional volume columns use sequential indices:
        vol_norm__ROI_1 ... vol_norm__ROI_332
        vol_abs__ROI_1  ... vol_abs__ROI_332

    Diffusion metric columns use raw atlas labels:
        FA/MD/AD/RD ROI_1–166       -> ROI_index 1–166
        FA/MD/AD/RD ROI_1001–1166   -> ROI_index 167–332
    """
    column = str(column)

    match = re.search(r"ROI[_-]?(\d+)", column, flags=re.I)

    if match is None:
        return None

    label = int(match.group(1))

    # Volume tables already use sequential ROI indices 1–332
    if column.startswith(("vol_norm__", "vol_abs__")):
        if 1 <= label <= 332:
            return label
        return None

    # Diffusion metrics use raw atlas labels
    if column.startswith(("FA__", "MD__", "AD__", "RD__")):
        if 1 <= label <= 166:
            return label

        if 1001 <= label <= 1166:
            return label - 834

        return None

    # Conservative fallback
    if 1 <= label <= 332:
        return label

    return None


def clean_level(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).replace("_", " ").strip()


def significance_symbol(q: float) -> str:
    if not np.isfinite(q):
        return ""

    for cutoff in sorted(STAR_Q):
        if q < cutoff:
            return STAR_Q[cutoff]

    return ""


# =============================================================================
# Read data
# =============================================================================

for required in [BRAIN_FILE, MASTER_FILE, ROI_MAPPING_FILE]:
    if not required.exists():
        raise FileNotFoundError(required)


brain = pd.read_csv(BRAIN_FILE)
master = pd.read_csv(MASTER_FILE)
roi_map = pd.read_csv(ROI_MAPPING_FILE)


brain_id = detect_id_column(brain, "brain table")
master_id = detect_id_column(master, "master table")

brain = brain.rename(columns={brain_id: "mouse_id"})
master = master.rename(columns={master_id: "mouse_id"})

brain["mouse_id"] = brain["mouse_id"].astype(str).str.strip()
master["mouse_id"] = master["mouse_id"].astype(str).str.strip()


if brain["mouse_id"].duplicated().any():
    raise RuntimeError("Duplicate mouse IDs in advanced brain table.")

if master["mouse_id"].duplicated().any():
    raise RuntimeError("Duplicate mouse IDs in GLP1_MRI.csv.")


brain_ids = set(brain["mouse_id"])
master_ids = set(master["mouse_id"])

if brain_ids != master_ids:
    raise RuntimeError(
        "Mouse IDs differ between brain and master tables.\n"
        f"Only brain: {sorted(brain_ids - master_ids)}\n"
        f"Only master: {sorted(master_ids - brain_ids)}"
    )


# =============================================================================
# Replace stale cardiac variables with canonical master-table variables
# =============================================================================

stale_cardiac_columns = [
    c for c in brain.columns
    if c.startswith("cardiac_")
    or c in {
        "Cardiac_ID",
        "Cardiac_ID_norm",
        "cardiac_table_ID",
        "cardiac_table_Animal_Code",
        "Weight",
    }
]

brain_without_cardiac = brain.drop(
    columns=stale_cardiac_columns,
    errors="ignore",
)


canonical_columns = [
    c for c in master.columns
    if c.startswith("cardiac_")
    or c in {
        "Cardiac_ID",
        "Cardiac_ID_norm",
        "cardiac_table_ID",
        "cardiac_table_Animal_Code",
        "Weight",
    }
]

data = brain_without_cardiac.merge(
    master[["mouse_id"] + canonical_columns],
    on="mouse_id",
    how="left",
    validate="one_to_one",
)


print("=" * 100)
print("Input validation")
print("=" * 100)
print("Brain table:", BRAIN_FILE)
print("Canonical cardiac table:", MASTER_FILE)
print("Merged shape:", data.shape)


check_columns = [
    "Cardiac_ID",
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LV_Volume",
    "Weight",
]

available_check = [c for c in check_columns if c in data.columns]

print(
    "\nCorrected mapping rows:\n",
    data.loc[
        data["mouse_id"].isin(
            ["B26010604", "B26010605", "B26010606"]
        ),
        ["mouse_id"] + available_check,
    ].to_string(index=False),
)


# =============================================================================
# ROI mapping metadata
# =============================================================================

roi_map["ROI_index"] = pd.to_numeric(
    roi_map["ROI_index"],
    errors="raise",
).astype(int)

if len(roi_map) != 332:
    raise RuntimeError(
        f"ROI mapping should have 332 rows, found {len(roi_map)}."
    )


if "structure_display_name" not in roi_map.columns:
    roi_map["structure_display_name"] = (
        roi_map["atlas_abbrev"].astype(str)
        + " — "
        + roi_map["anatomical_name"].astype(str)
    )


block_short = {
    "hemisphere_block_A": "A",
    "hemisphere_block_B": "B",
}

roi_map["hemisphere_short"] = (
    roi_map["hemisphere"]
    .map(block_short)
    .fillna(roi_map["hemisphere"].astype(str))
)

roi_map["roi_display_name"] = (
    roi_map["hemisphere_short"].astype(str)
    + " | "
    + roi_map["structure_display_name"].astype(str)
)

roi_map["ontology_level_1_clean"] = (
    roi_map["ontology_level_1"].map(clean_level)
)

roi_map["ontology_level_2_clean"] = (
    roi_map["ontology_level_2"].map(clean_level)
)


# =============================================================================
# Identify features
# =============================================================================

fa_columns = [
    c for c in data.columns
    if c.startswith("FA__")
    and extract_roi_index(c) is not None
]

volume_columns = [
    c for c in data.columns
    if c.startswith("vol_norm__")
    and extract_roi_index(c) is not None
]


if len(fa_columns) != 332:
    raise RuntimeError(
        f"Expected 332 regional FA columns, found {len(fa_columns)}."
    )

if len(volume_columns) != 332:
    raise RuntimeError(
        "Expected 332 normalized regional-volume columns, "
        f"found {len(volume_columns)}."
    )


cardiac_columns = [
    c for c in CARDIAC_ORDER
    if c in data.columns
]

if not cardiac_columns:
    raise RuntimeError("No requested cardiac outcome columns found.")


print("\nRegional FA columns:", len(fa_columns))
print("Normalized volume columns:", len(volume_columns))
print("Cardiac outcomes:", len(cardiac_columns))

for outcome in cardiac_columns:
    n = pd.to_numeric(
        data[outcome],
        errors="coerce",
    ).notna().sum()

    print(f"{outcome}: n={n}")


# =============================================================================
# Association calculation
# =============================================================================

def calculate_associations(
    feature_columns: list[str],
    modality: str,
) -> pd.DataFrame:

    records = []

    for feature_column in feature_columns:
        roi_index = extract_roi_index(feature_column)

        x_all = pd.to_numeric(
            data[feature_column],
            errors="coerce",
        )

        for cardiac_column in cardiac_columns:
            y_all = pd.to_numeric(
                data[cardiac_column],
                errors="coerce",
            )

            valid = x_all.notna() & y_all.notna()

            x = x_all[valid]
            y = y_all[valid]
            n = int(valid.sum())

            rho = np.nan
            p = np.nan

            if (
                n >= MIN_N
                and x.nunique() >= 3
                and y.nunique() >= 3
            ):
                rho, p = spearmanr(x, y)

            records.append(
                {
                    "modality": modality,
                    "feature_column": feature_column,
                    "ROI_index": roi_index,
                    "cardiac_feature": cardiac_column,
                    "cardiac_label": CARDIAC_LABELS.get(
                        cardiac_column,
                        cardiac_column,
                    ),
                    "n": n,
                    "rho": rho,
                    "p_value": p,
                }
            )

    result = pd.DataFrame(records)

    # Global FDR across all ROI × cardiac pairs in this modality
    result["q_global_modality"] = bh_fdr(
        result["p_value"].to_numpy()
    )

    # FDR separately within each cardiac outcome
    result["q_by_outcome"] = np.nan

    for cardiac_feature, indices in result.groupby(
        "cardiac_feature"
    ).groups.items():
        result.loc[
            indices,
            "q_by_outcome",
        ] = bh_fdr(
            result.loc[indices, "p_value"].to_numpy()
        )

    result = result.merge(
        roi_map[
            [
                "ROI_index",
                "raw_atlas_label_id",
                "ontology_value",
                "hemisphere",
                "hemisphere_short",
                "atlas_abbrev",
                "anatomical_name",
                "structure_display_name",
                "roi_display_name",
                "ontology_level_1_clean",
                "ontology_level_2_clean",
            ]
        ],
        on="ROI_index",
        how="left",
        validate="many_to_one",
    )

    result["abs_rho"] = result["rho"].abs()
    result["significance"] = result[
        "q_by_outcome"
    ].map(significance_symbol)

    return result


volume_results = calculate_associations(
    volume_columns,
    modality="normalized_regional_volume",
)

fa_results = calculate_associations(
    fa_columns,
    modality="regional_FA",
)


volume_results.to_csv(
    OUT / "regional_volume_cardiac_spearman_named.csv",
    index=False,
)

fa_results.to_csv(
    OUT / "regional_FA_cardiac_spearman_named.csv",
    index=False,
)


# =============================================================================
# Balanced selection of top regions
# =============================================================================

def select_balanced_top_regions(
    result: pd.DataFrame,
) -> list[int]:

    selected = set()

    # Include strongest regions for every cardiac outcome
    for cardiac_feature in cardiac_columns:
        subset = result.loc[
            result["cardiac_feature"].eq(cardiac_feature)
            & result["rho"].notna()
        ].copy()

        subset = subset.sort_values(
            [
                "q_by_outcome",
                "p_value",
                "abs_rho",
            ],
            ascending=[True, True, False],
        )

        selected.update(
            subset.head(TOP_PER_OUTCOME)["ROI_index"].tolist()
        )

    # Overall ROI score
    roi_score = (
        result.groupby("ROI_index")
        .agg(
            min_q=("q_by_outcome", "min"),
            min_p=("p_value", "min"),
            max_abs_rho=("abs_rho", "max"),
            n_q05=(
                "q_by_outcome",
                lambda x: int((x < 0.05).sum()),
            ),
            n_q10=(
                "q_by_outcome",
                lambda x: int((x < 0.10).sum()),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "n_q05",
                "n_q10",
                "min_q",
                "min_p",
                "max_abs_rho",
            ],
            ascending=[False, False, True, True, False],
        )
    )

    # Fill to minimum size
    for roi_index in roi_score["ROI_index"]:
        selected.add(int(roi_index))

        if len(selected) >= MIN_TOP_ROWS:
            break

    # Cap size while retaining best overall ROIs
    if len(selected) > MAX_TOP_ROWS:
        selected_score = roi_score.loc[
            roi_score["ROI_index"].isin(selected)
        ].head(MAX_TOP_ROWS)

        selected = set(
            selected_score["ROI_index"].astype(int)
        )

    return sorted(selected)


# =============================================================================
# Row clustering
# =============================================================================

def cluster_row_order(matrix: pd.DataFrame) -> list[int]:
    if matrix.shape[0] <= 2:
        return list(matrix.index)

    values = matrix.fillna(0).to_numpy(dtype=float)

    try:
        distance = pdist(values, metric="correlation")

        if not np.all(np.isfinite(distance)):
            raise ValueError("Non-finite correlation distances.")

        tree = linkage(distance, method="average")
        order = leaves_list(tree)

    except Exception:
        tree = linkage(values, method="average", metric="euclidean")
        order = leaves_list(tree)

    return matrix.index.to_numpy()[order].tolist()


# =============================================================================
# Plot heatmaps
# =============================================================================

sns.set_theme(
    context="paper",
    style="white",
    font_scale=0.90,
)


def prepare_matrices(
    result: pd.DataFrame,
    roi_indices: list[int] | None,
):
    rho = result.pivot(
        index="ROI_index",
        columns="cardiac_feature",
        values="rho",
    ).reindex(columns=cardiac_columns)

    stars = result.pivot(
        index="ROI_index",
        columns="cardiac_feature",
        values="significance",
    ).reindex(columns=cardiac_columns)

    if roi_indices is not None:
        rho = rho.loc[rho.index.intersection(roi_indices)]
        stars = stars.reindex(rho.index)

    ordered_rows = cluster_row_order(rho)

    rho = rho.loc[ordered_rows]
    stars = stars.loc[ordered_rows]

    display_lookup = (
        roi_map.set_index("ROI_index")["roi_display_name"]
        .to_dict()
    )

    rho.index = [
        display_lookup.get(i, f"ROI_{i}")
        for i in rho.index
    ]

    stars.index = rho.index

    rho.columns = [
        CARDIAC_LABELS.get(c, c)
        for c in rho.columns
    ]

    stars.columns = rho.columns

    return rho, stars


def draw_top_heatmap(
    result: pd.DataFrame,
    modality_title: str,
    output_stem: str,
):
    selected = select_balanced_top_regions(result)

    rho, stars = prepare_matrices(
        result,
        roi_indices=selected,
    )

    vmax = np.nanmax(np.abs(rho.to_numpy()))

    # Prevent extremely narrow color scales
    vmax = max(0.50, min(1.0, float(vmax)))

    figure_height = max(
        10,
        0.34 * rho.shape[0] + 3.5,
    )

    fig, ax = plt.subplots(
        figsize=(16, figure_height)
    )

    sns.heatmap(
        rho,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=stars,
        fmt="",
        annot_kws={
            "fontsize": 8,
            "fontweight": "bold",
        },
        linewidths=0.35,
        linecolor="white",
        cbar_kws={
            "label": "Spearman ρ",
            "shrink": 0.70,
            "pad": 0.02,
        },
    )

    ax.set_title(
        f"{modality_title} × cardiac phenotypes\n"
        f"Balanced top-region view; "
        f"stars use BH-FDR within each cardiac outcome",
        fontsize=15,
        pad=16,
        fontweight="bold",
    )

    ax.set_xlabel("Cardiac phenotype", fontsize=11)
    ax.set_ylabel("Atlas ROI", fontsize=11)

    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=9,
    )

    ax.tick_params(
        axis="y",
        labelrotation=0,
        labelsize=8,
    )

    note = (
        "*** q<0.001   ** q<0.01   * q<0.05   · q<0.10\n"
        "A/B denote atlas hemisphere blocks; anatomical left/right "
        "has not yet been assigned."
    )

    fig.text(
        0.01,
        0.006,
        note,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.035, 1, 1])

    fig.savefig(
        OUT / f"{output_stem}_top_balanced.png",
        dpi=350,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{output_stem}_top_balanced.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    selected_table = (
        result.loc[result["ROI_index"].isin(selected)]
        .sort_values(
            ["q_by_outcome", "p_value", "abs_rho"],
            ascending=[True, True, False],
        )
    )

    selected_table.to_csv(
        OUT / f"{output_stem}_top_balanced_associations.csv",
        index=False,
    )


def draw_all_roi_overview(
    result: pd.DataFrame,
    modality_title: str,
    output_stem: str,
):
    rho, _ = prepare_matrices(
        result,
        roi_indices=None,
    )

    vmax = np.nanmax(np.abs(rho.to_numpy()))
    vmax = max(0.50, min(1.0, float(vmax)))

    fig, ax = plt.subplots(
        figsize=(15, 22)
    )

    sns.heatmap(
        rho,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        yticklabels=False,
        linewidths=0,
        cbar_kws={
            "label": "Spearman ρ",
            "shrink": 0.45,
            "pad": 0.02,
        },
    )

    ax.set_title(
        f"{modality_title} × cardiac phenotypes\n"
        "All 332 atlas ROIs, hierarchically ordered",
        fontsize=15,
        pad=16,
        fontweight="bold",
    )

    ax.set_xlabel("Cardiac phenotype", fontsize=11)
    ax.set_ylabel("332 atlas ROIs", fontsize=11)

    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=9,
    )

    plt.tight_layout()

    fig.savefig(
        OUT / f"{output_stem}_all_332_overview.png",
        dpi=350,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{output_stem}_all_332_overview.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


draw_top_heatmap(
    volume_results,
    modality_title="Normalized regional brain volume",
    output_stem="regional_volume_cardiac",
)

draw_all_roi_overview(
    volume_results,
    modality_title="Normalized regional brain volume",
    output_stem="regional_volume_cardiac",
)

draw_top_heatmap(
    fa_results,
    modality_title="Regional fractional anisotropy",
    output_stem="regional_FA_cardiac",
)

draw_all_roi_overview(
    fa_results,
    modality_title="Regional fractional anisotropy",
    output_stem="regional_FA_cardiac",
)


# =============================================================================
# Overall top associations summary
# =============================================================================

top_volume = (
    volume_results
    .sort_values(
        ["q_global_modality", "q_by_outcome", "p_value"],
        ascending=True,
    )
    .head(50)
)

top_fa = (
    fa_results
    .sort_values(
        ["q_global_modality", "q_by_outcome", "p_value"],
        ascending=True,
    )
    .head(50)
)

top_summary = pd.concat(
    [top_volume, top_fa],
    ignore_index=True,
)

top_summary.to_csv(
    OUT / "top_100_regional_cardiac_associations_named.csv",
    index=False,
)


# =============================================================================
# Print concise results
# =============================================================================

print("\n" + "=" * 100)
print("TOP NORMALIZED-VOLUME ASSOCIATIONS")
print("=" * 100)

print(
    volume_results.sort_values(
        ["q_global_modality", "p_value"],
        ascending=True,
    )[
        [
            "roi_display_name",
            "cardiac_label",
            "n",
            "rho",
            "p_value",
            "q_by_outcome",
            "q_global_modality",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 100)
print("TOP REGIONAL-FA ASSOCIATIONS")
print("=" * 100)

print(
    fa_results.sort_values(
        ["q_global_modality", "p_value"],
        ascending=True,
    )[
        [
            "roi_display_name",
            "cardiac_label",
            "n",
            "rho",
            "p_value",
            "q_by_outcome",
            "q_global_modality",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nOutputs saved to:")
print(OUT)
