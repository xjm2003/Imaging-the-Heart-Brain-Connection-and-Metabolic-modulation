from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1/"
    "analysis_ready_regional_FA_volume_20260624_plus_B25122904/"
    "tables/regional_cardiac_heatmaps_named"
)

OUT = ROOT / "publication_style"
OUT.mkdir(parents=True, exist_ok=True)

N_ROI = 28
VMIN = -0.8
VMAX = 0.8


CARDIAC_ORDER = [
    # LV
    "cardiac_Diastolic_LV_Volume",
    "cardiac_Systolic_LV_Volume",
    "cardiac_Stroke_Volume",
    "cardiac_Ejection_Fraction",
    "cardiac_Cardiac_Output",

    # RV
    "cardiac_Diastolic_RV",
    "cardiac_Systolic_RV",
    "cardiac_RV_Stroke_Volume",

    # Atria
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LA",
    "cardiac_Diastolic_RA",
    "cardiac_Systolic_RA",

    # Myocardium
    "cardiac_Diastolic_Myo",
    "cardiac_Systolic_Myo",
    "cardiac_MYO_MASS",

    # Heart rate
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

# start inclusive, end exclusive
CARDIAC_GROUPS = [
    ("LV", 0, 5),
    ("RV", 5, 8),
    ("Atria", 8, 12),
    ("Myocardium", 12, 15),
    ("Rate", 15, 16),
]


def global_star(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def shorten_name(value, maximum=43):
    value = str(value).replace("_", " ").strip()
    if len(value) <= maximum:
        return value
    return value[:maximum - 1].rstrip() + "…"


def make_roi_label(row):
    block = str(row.get("hemisphere_short", "")).strip()
    abbrev = str(row.get("atlas_abbrev", "")).strip()
    name = shorten_name(row.get("anatomical_name", ""))

    return f"{block} · {abbrev} — {name}"


def select_rois(result):
    """
    Balanced selection:
    take two strongest ROIs per cardiac outcome,
    then rank and retain 28 total ROIs.
    """
    result = result.copy()
    result["abs_rho"] = result["rho"].abs()

    selected = set()

    for outcome in CARDIAC_ORDER:
        subset = result.loc[
            result["cardiac_feature"].eq(outcome)
            & result["rho"].notna()
        ].sort_values(
            [
                "q_global_modality",
                "q_by_outcome",
                "p_value",
                "abs_rho",
            ],
            ascending=[True, True, True, False],
        )

        selected.update(
            subset.head(2)["ROI_index"].astype(int)
        )

    ranking = (
        result.groupby("ROI_index", as_index=False)
        .agg(
            min_global_q=("q_global_modality", "min"),
            min_outcome_q=("q_by_outcome", "min"),
            min_p=("p_value", "min"),
            max_abs_rho=("abs_rho", "max"),
        )
        .sort_values(
            [
                "min_global_q",
                "min_outcome_q",
                "min_p",
                "max_abs_rho",
            ],
            ascending=[True, True, True, False],
        )
    )

    # Fill to 28
    for roi_index in ranking["ROI_index"]:
        selected.add(int(roi_index))
        if len(selected) >= N_ROI:
            break

    # Cap at 28
    ranking = ranking.loc[
        ranking["ROI_index"].isin(selected)
    ].head(N_ROI)

    return ranking["ROI_index"].astype(int).tolist()


def cluster_rows(matrix):
    if len(matrix) < 3:
        return matrix.index.tolist()

    values = matrix.fillna(0).to_numpy(dtype=float)

    try:
        distance = pdist(values, metric="correlation")

        if not np.all(np.isfinite(distance)):
            raise ValueError

        tree = linkage(distance, method="average")
    except Exception:
        tree = linkage(
            values,
            method="average",
            metric="euclidean",
        )

    return matrix.index.to_numpy()[
        leaves_list(tree)
    ].tolist()


def prepare_heatmap(result):
    result = result.copy()

    available_outcomes = [
        c for c in CARDIAC_ORDER
        if c in set(result["cardiac_feature"])
    ]

    selected = select_rois(result)

    rho = (
        result.pivot(
            index="ROI_index",
            columns="cardiac_feature",
            values="rho",
        )
        .reindex(index=selected, columns=available_outcomes)
    )

    result["global_star"] = result[
        "q_global_modality"
    ].map(global_star)

    stars = (
        result.pivot(
            index="ROI_index",
            columns="cardiac_feature",
            values="global_star",
        )
        .reindex(index=selected, columns=available_outcomes)
    )

    row_order = cluster_rows(rho)
    rho = rho.loc[row_order]
    stars = stars.loc[row_order]

    metadata = (
        result.drop_duplicates("ROI_index")
        .set_index("ROI_index")
    )

    labels = [
        make_roi_label(metadata.loc[roi_index])
        for roi_index in row_order
    ]

    rho.index = labels
    stars.index = labels

    rho.columns = [
        CARDIAC_LABELS.get(c, c)
        for c in rho.columns
    ]
    stars.columns = rho.columns

    return rho, stars, selected


def add_group_labels(ax):
    for name, start, end in CARDIAC_GROUPS:
        center = (start + end) / 2

        ax.text(
            center,
            1.025,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )

        if end < len(CARDIAC_ORDER):
            ax.axvline(
                end,
                color="black",
                linewidth=1.0,
            )


def draw_heatmap(result_file, title, output_stem):
    result = pd.read_csv(result_file)

    rho, stars, selected = prepare_heatmap(result)

    height = max(10.5, 0.34 * len(rho) + 3.0)

    fig, ax = plt.subplots(
        figsize=(14.5, height)
    )

    sns.heatmap(
        rho,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=VMIN,
        vmax=VMAX,
        annot=stars,
        fmt="",
        annot_kws={
            "fontsize": 9,
            "fontweight": "bold",
        },
        linewidths=0.30,
        linecolor="white",
        cbar_kws={
            "label": "Spearman ρ",
            "shrink": 0.72,
            "pad": 0.025,
        },
    )

    add_group_labels(ax)

    ax.set_title(
        f"{title}\n"
        f"Top {len(rho)} ROIs; stars indicate global BH-FDR significance",
        fontsize=15,
        fontweight="bold",
        pad=30,
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
        labelsize=8.2,
    )

    fig.text(
        0.01,
        0.008,
        "* qglobal < 0.05; ** qglobal < 0.01; "
        "*** qglobal < 0.001. "
        "A/B denote atlas hemisphere blocks; "
        "anatomical left/right remains unresolved.",
        fontsize=8,
        ha="left",
    )

    plt.tight_layout(rect=[0, 0.035, 1, 0.96])

    png = OUT / f"{output_stem}.png"
    pdf = OUT / f"{output_stem}.pdf"

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    selected_table = result.loc[
        result["ROI_index"].isin(selected)
    ].copy()

    selected_table.to_csv(
        OUT / f"{output_stem}_selected_associations.csv",
        index=False,
    )

    print("Saved:", png)
    print("Saved:", pdf)

    return fig


sns.set_theme(
    context="paper",
    style="white",
    font_scale=0.95,
)

fa_figure = draw_heatmap(
    ROOT / "regional_FA_cardiac_spearman_named.csv",
    "Regional fractional anisotropy × cardiac phenotypes",
    "regional_FA_cardiac_publication",
)

volume_figure = draw_heatmap(
    ROOT / "regional_volume_cardiac_spearman_named.csv",
    "Normalized regional brain volume × cardiac phenotypes",
    "regional_volume_cardiac_publication",
)

# Two-page paired PDF
paired_pdf = OUT / "regional_FA_volume_cardiac_paired.pdf"

with PdfPages(paired_pdf) as pdf:
    pdf.savefig(fa_figure, bbox_inches="tight")
    pdf.savefig(volume_figure, bbox_inches="tight")

plt.close(fa_figure)
plt.close(volume_figure)

print("\nPaired PDF:", paired_pdf)
print("Output directory:", OUT)
