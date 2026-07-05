from pathlib import Path
import os, glob, re
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    from scipy import stats
except Exception:
    stats = None

BASE = Path("analysis_ready_regional_FA_volume_20260624")
TABLES = BASE / "tables"
DOCS = BASE / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

def read_csv(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None

meta = read_csv("GLP1_mouse_metadata_25.csv")
fa = read_csv("GLP1_regional_FA_25.csv")
vol_abs = read_csv("GLP1_regional_volume_absolute_mm3_25.csv")
vol_norm = read_csv("GLP1_regional_volume_normalized_25.csv")
vol_qc = read_csv("GLP1_regional_volume_QC_25.csv")
analysis = read_csv("GLP1_connectome_cardiac_mwm_analysis_table_25.csv")
conn_qc = read_csv("GLP1_connectome_global_QC_25.csv")

# ---------- ROI mapping ----------
roi_rows = []
for roi in range(1, 333):
    if roi <= 166:
        raw = roi
        hemi_block = "raw_label_1_166"
        bilateral_region = roi
        homologous_roi = roi + 166
    else:
        raw = 1000 + (roi - 166)
        hemi_block = "raw_label_1001_1166"
        bilateral_region = roi - 166
        homologous_roi = roi - 166
    roi_rows.append({
        "ROI_index": roi,
        "ROI_column_prefix": f"ROI_{roi}",
        "raw_atlas_label_id": raw,
        "atlas_label_block": hemi_block,
        "bilateral_region_index": bilateral_region,
        "homologous_ROI_index": homologous_roi,
        "anatomical_name": "not_available_in_current_tables"
    })
roi_map = pd.DataFrame(roi_rows)
roi_map.to_csv(DOCS / "ROI_mapping_332.csv", index=False)

# ---------- Data dictionary ----------
dd = [
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "ID", "unit": "text", "definition": "Unique mouse identifier used across analysis-ready tables."},
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "TreatmentGroup / Group / NameGroup", "unit": "category", "definition": "Treatment/group variables available in the metadata table."},
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "Genotype", "unit": "category", "definition": "Mouse genotype, used for genotype-comparison statistics."},
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "Sex", "unit": "category", "definition": "Mouse sex."},
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "Diet", "unit": "category", "definition": "Diet assignment where available."},
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "Age_Months / Age_Imaging", "unit": "months or source-specific age unit", "definition": "Age variables available in metadata."},
    {"file": "GLP1_mouse_metadata_25.csv", "variable": "Weight", "unit": "source metadata unit", "definition": "Body mass / weight variable from metadata."},
    {"file": "GLP1_regional_volume_absolute_mm3_25.csv", "variable": "ROI_1_volume_mm3 ... ROI_332_volume_mm3", "unit": "mm3", "definition": "Regional volume extracted from native T2-space atlas labels after compact mapping from raw labels 1-166 and 1001-1166 to ROI_1-ROI_332."},
    {"file": "GLP1_regional_volume_normalized_25.csv", "variable": "ROI_1_volume_fraction ... ROI_332_volume_fraction", "unit": "fraction", "definition": "Regional volume divided by total labeled brain volume for that mouse."},
    {"file": "GLP1_regional_volume_absolute_mm3_25.csv / GLP1_regional_volume_normalized_25.csv", "variable": "total_brain_label_volume_mm3", "unit": "mm3", "definition": "Total labeled brain volume calculated as the sum of all compact 332 ROI volumes."},
    {"file": "GLP1_regional_volume_bilateral_166_25.csv", "variable": "Region_1_bilateral_volume_mm3 ... Region_166_bilateral_volume_mm3", "unit": "mm3", "definition": "Bilateral regional volume after summing homologous ROI pairs whose raw atlas labels differ by 1000."},
    {"file": "GLP1_regional_FA_25.csv", "variable": "ROI_1_FA ... ROI_332_FA", "unit": "unitless", "definition": "Regional fractional anisotropy measurement for each compact ROI."},
    {"file": "GLP1_regional_MD_25.csv", "variable": "ROI_1_MD ... ROI_332_MD", "unit": "diffusion metric source unit", "definition": "Regional mean diffusivity measurement for each compact ROI."},
    {"file": "GLP1_regional_AD_25.csv", "variable": "ROI_1_AD ... ROI_332_AD", "unit": "diffusion metric source unit", "definition": "Regional axial diffusivity measurement for each compact ROI."},
    {"file": "GLP1_regional_RD_25.csv", "variable": "ROI_1_RD ... ROI_332_RD", "unit": "diffusion metric source unit", "definition": "Regional radial diffusivity measurement for each compact ROI."},
    {"file": "GLP1_regional_volume_QC_25.csv", "variable": "n_unexpected_labels", "unit": "count", "definition": "Number of atlas labels outside expected raw label ranges 1-166 and 1001-1166."},
    {"file": "GLP1_regional_volume_QC_25.csv", "variable": "normalized_sum", "unit": "fraction", "definition": "Sum of normalized 332 ROI volumes; expected to be 1.0 for successful volume extraction."},
    {"file": "ROI_mapping_332.csv", "variable": "raw_atlas_label_id", "unit": "atlas label ID", "definition": "Original atlas label ID before compact remapping."},
]
pd.DataFrame(dd).to_csv(DOCS / "data_dictionary.csv", index=False)

# ---------- Overlap and missingness ----------
def id_set(df):
    if df is None or "ID" not in df.columns:
        return set()
    return set(df["ID"].astype(str))

all_ids = sorted(
    id_set(meta) | id_set(fa) | id_set(vol_abs) | id_set(vol_norm) |
    id_set(analysis) | id_set(vol_qc) | id_set(conn_qc)
)

overlap = pd.DataFrame({"ID": all_ids})
sources = {
    "has_metadata": meta,
    "has_regional_FA": fa,
    "has_regional_volume_absolute": vol_abs,
    "has_regional_volume_normalized": vol_norm,
    "has_volume_QC": vol_qc,
    "has_connectome_cardiac_mwm_table": analysis,
    "has_connectome_QC": conn_qc,
}
for col, df in sources.items():
    overlap[col] = overlap["ID"].isin(id_set(df))

if meta is not None:
    keep = [c for c in ["ID","TreatmentGroup","Group","NameGroup","Diet","Sex","Genotype","Weight","Age_Months","Age_Imaging"] if c in meta.columns]
    overlap = overlap.merge(meta[keep], on="ID", how="left")

# cardiac availability from merged table
cardiac_cols = []
if analysis is not None:
    patterns = ["cardiac", "diastolic", "systolic", "LV", "RV", "LA", "RA", "heart", "stroke", "ejection", "output", "myocard", "MWM"]
    for c in analysis.columns:
        if c == "ID":
            continue
        if any(p.lower() in c.lower() for p in patterns):
            cardiac_cols.append(c)
    tmp = analysis[["ID"] + cardiac_cols].copy() if cardiac_cols else analysis[["ID"]].copy()
    if cardiac_cols:
        tmp["has_cardiac_data"] = tmp[cardiac_cols].notna().any(axis=1)
    else:
        tmp["has_cardiac_data"] = True
    tmp["ID"] = tmp["ID"].astype(str)
    overlap = overlap.merge(tmp[["ID","has_cardiac_data"]], on="ID", how="left")
else:
    overlap["has_cardiac_data"] = False

overlap["has_cardiac_data"] = overlap["has_cardiac_data"].fillna(False)
overlap["has_volume_and_FA"] = overlap["has_regional_FA"] & overlap["has_regional_volume_absolute"] & overlap["has_regional_volume_normalized"]
overlap["has_volume_FA_and_cardiac"] = overlap["has_volume_and_FA"] & overlap["has_cardiac_data"]
overlap.to_csv(DOCS / "overlap_mouse_level.csv", index=False)

group_cols = [c for c in ["TreatmentGroup","Genotype","Sex"] if c in overlap.columns]
if group_cols:
    overlap_summary = overlap.groupby(group_cols, dropna=False).agg(
        n_mice=("ID","count"),
        n_with_volume=("has_regional_volume_absolute","sum"),
        n_with_FA=("has_regional_FA","sum"),
        n_with_cardiac=("has_cardiac_data","sum"),
        n_with_volume_FA_cardiac=("has_volume_FA_and_cardiac","sum")
    ).reset_index()
else:
    overlap_summary = pd.DataFrame({"summary": ["No grouping variables available"]})
overlap_summary.to_csv(DOCS / "overlap_summary_by_treatment_genotype_sex.csv", index=False)

unique_rows = []
dataset_flags = ["has_metadata","has_regional_FA","has_regional_volume_absolute","has_regional_volume_normalized","has_cardiac_data","has_connectome_QC"]
for flag in dataset_flags:
    ids = set(overlap.loc[overlap[flag], "ID"])
    others = set(overlap.loc[overlap[[f for f in dataset_flags if f != flag]].any(axis=1), "ID"])
    unique_rows.append({
        "dataset_flag": flag,
        "n_present": len(ids),
        "n_unique_to_this_dataset": len(ids - others),
        "unique_IDs": ";".join(sorted(ids - others))
    })
pd.DataFrame(unique_rows).to_csv(DOCS / "unique_to_dataset_summary.csv", index=False)

missing_rows = []
key_vars = [c for c in ["TreatmentGroup","Group","NameGroup","Diet","Sex","Genotype","Weight","Age_Months","Age_Imaging"] if c in overlap.columns]
for c in key_vars:
    missing_rows.append({
        "variable": c,
        "n_missing": int(overlap[c].isna().sum()),
        "n_available": int(overlap[c].notna().sum()),
        "missing_IDs": ";".join(overlap.loc[overlap[c].isna(), "ID"].astype(str))
    })
for flag in ["has_regional_FA","has_regional_volume_absolute","has_regional_volume_normalized","has_cardiac_data"]:
    missing_rows.append({
        "variable": flag,
        "n_missing": int((~overlap[flag]).sum()),
        "n_available": int(overlap[flag].sum()),
        "missing_IDs": ";".join(overlap.loc[~overlap[flag], "ID"].astype(str))
    })
pd.DataFrame(missing_rows).to_csv(DOCS / "missing_data_summary.csv", index=False)

pd.DataFrame({"candidate_cardiac_columns": cardiac_cols}).to_csv(DOCS / "candidate_cardiac_columns.csv", index=False)

# ---------- independent brain mask volume sanity check ----------
mask_rows = []
if nib is not None and meta is not None and vol_abs is not None:
    label_total = vol_abs[["ID","total_brain_label_volume_mm3"]].copy()
    for sid in meta["ID"].astype(str):
        candidates = []
        for pat in [
            f"preproc_{sid}/*T2*mask*.nii*",
            f"preproc_{sid}/*brain*mask*.nii*",
            f"preproc_{sid}/*mask*.nii*",
        ]:
            candidates.extend(glob.glob(pat))
        candidates = [c for c in sorted(set(candidates)) if "label" not in Path(c).name.lower()]
        chosen = candidates[0] if candidates else ""
        row = {"ID": sid, "mask_file": chosen, "mask_status": "missing_mask"}
        if chosen:
            try:
                img = nib.load(chosen)
                data = np.asanyarray(img.dataobj)
                voxel = abs(np.prod(img.header.get_zooms()[:3]))
                mask_vol = float((data > 0).sum() * voxel)
                row.update({"mask_status": "ok", "brain_mask_volume_mm3": mask_vol})
            except Exception as e:
                row.update({"mask_status": f"error: {e}", "brain_mask_volume_mm3": np.nan})
        mask_rows.append(row)
    mask_df = pd.DataFrame(mask_rows).merge(label_total, on="ID", how="left")
    if "brain_mask_volume_mm3" in mask_df.columns:
        mask_df["mask_minus_label_volume_mm3"] = mask_df["brain_mask_volume_mm3"] - mask_df["total_brain_label_volume_mm3"]
        mask_df["relative_difference_vs_label"] = mask_df["mask_minus_label_volume_mm3"] / mask_df["total_brain_label_volume_mm3"]
else:
    mask_df = pd.DataFrame({"note": ["nibabel, metadata, or volume table not available"]})
mask_df.to_csv(DOCS / "brain_mask_volume_sanity_check.csv", index=False)

# ---------- genotype-comparison stats ----------
def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order)+1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out

def glp1_filter(df):
    cols = [c for c in ["TreatmentGroup","Group","NameGroup"] if c in df.columns]
    if not cols:
        return pd.Series(True, index=df.index)
    text = df[cols].astype(str).agg(" ".join, axis=1)
    mask = text.str.contains("GLP", case=False, na=False)
    return mask if mask.any() else pd.Series(True, index=df.index)

def choose_age_col(df):
    for c in ["Age_Months","Age_Imaging","Age"]:
        if c in df.columns and df[c].notna().sum() > 0:
            return c
    return None

def fit_one(df, y_col, genotype_col, covariates):
    d = df[[y_col, genotype_col] + covariates].copy()
    d = d.dropna(subset=[y_col, genotype_col])
    d[genotype_col] = d[genotype_col].astype(str)
    geno_levels = sorted(d[genotype_col].dropna().unique())
    if len(geno_levels) < 2:
        return []
    ref = geno_levels[0]

    X_parts = [pd.Series(1.0, index=d.index, name="Intercept")]
    term_names = ["Intercept"]

    for g in geno_levels[1:]:
        s = (d[genotype_col] == g).astype(float).rename(f"Genotype:{g}_vs_{ref}")
        X_parts.append(s); term_names.append(s.name)

    used_covs = []
    for c in covariates:
        if c not in d.columns:
            continue
        s = d[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) < 2:
            continue
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            if vals.notna().sum() == 0 or vals.nunique(dropna=True) < 2:
                continue
            d[c] = vals
            X_parts.append(vals.rename(c)); term_names.append(c); used_covs.append(c)
        else:
            cats = sorted(s.dropna().astype(str).unique())
            if len(cats) < 2:
                continue
            for cat in cats[1:]:
                name = f"{c}:{cat}_vs_{cats[0]}"
                X_parts.append((s.astype(str) == cat).astype(float).rename(name))
                term_names.append(name)
            used_covs.append(c)

    model_df = pd.concat([d[[y_col, genotype_col]], *X_parts], axis=1).dropna()
    y = model_df[y_col].astype(float).to_numpy()
    X = model_df[term_names].astype(float).to_numpy()
    n, p = X.shape
    if n <= p + 1:
        return []

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df_resid = n - p
    sigma2 = float((resid @ resid) / df_resid)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    rows = []
    geno_counts = model_df[genotype_col].value_counts().to_dict()
    for i, term in enumerate(term_names):
        if not term.startswith("Genotype:"):
            continue
        comp = term.replace("Genotype:", "").split("_vs_")[0]
        tval = beta[i] / se[i] if se[i] > 0 else np.nan
        pval = 2 * stats.t.sf(abs(tval), df_resid) if stats is not None and np.isfinite(tval) else np.nan
        rows.append({
            "outcome": y_col,
            "reference_genotype": ref,
            "comparison_genotype": comp,
            "n_total_model": n,
            "n_reference": geno_counts.get(ref, 0),
            "n_comparison": geno_counts.get(comp, 0),
            "effect_estimate": beta[i],
            "standard_error": se[i],
            "t_statistic": tval,
            "p_value": pval,
            "model_df_resid": df_resid,
            "covariates_used": ";".join(used_covs) if used_covs else "none"
        })
    return rows

def run_genotype_stats(measure_df, outcome_cols, out_name, model_label, extra_covariates=None):
    if measure_df is None or meta is None or "Genotype" not in meta.columns:
        pd.DataFrame({"note": ["Missing measurement table, metadata table, or Genotype column"]}).to_csv(DOCS / out_name, index=False)
        return

    d = meta.merge(measure_df, on="ID", how="inner")
    d = d.loc[glp1_filter(d)].copy()

    age_col = choose_age_col(d)
    covs = []
    if "Sex" in d.columns: covs.append("Sex")
    if age_col: covs.append(age_col)
    if "Weight" in d.columns: covs.append("Weight")
    if extra_covariates:
        covs.extend([c for c in extra_covariates if c in d.columns])

    rows = []
    for y in outcome_cols:
        rows.extend(fit_one(d, y, "Genotype", covs))

    res = pd.DataFrame(rows)
    if len(res):
        res.insert(0, "model", model_label)
        res["FDR_q_value"] = bh_fdr(res["p_value"].values)
    else:
        res = pd.DataFrame({"note": [f"No valid genotype model could be fit for {model_label}. Check genotype levels, sample size, and missingness."]})
    res.to_csv(DOCS / out_name, index=False)

fa_cols = [c for c in (fa.columns if fa is not None else []) if re.match(r"ROI_\d+_FA$", c)]
vol_norm_cols = [c for c in (vol_norm.columns if vol_norm is not None else []) if re.match(r"ROI_\d+_volume_fraction$", c)]
vol_abs_cols = [c for c in (vol_abs.columns if vol_abs is not None else []) if re.match(r"ROI_\d+_volume_mm3$", c)]

run_genotype_stats(
    fa, fa_cols,
    "genotype_comparison_regional_FA_GLP1.csv",
    "Regional FA ~ Genotype + Sex + Age + Weight within GLP-1-treated group"
)

run_genotype_stats(
    vol_norm, vol_norm_cols,
    "genotype_comparison_regional_volume_normalized_GLP1.csv",
    "Normalized regional volume ~ Genotype + Sex + Age + Weight within GLP-1-treated group"
)

run_genotype_stats(
    vol_abs, vol_abs_cols,
    "genotype_comparison_regional_volume_absolute_TBV_adjusted_GLP1.csv",
    "Absolute regional volume ~ Genotype + total brain label volume + Sex + Age + Weight within GLP-1-treated group",
    extra_covariates=["total_brain_label_volume_mm3"]
)

# ---------- README and report ----------
n_meta = len(meta) if meta is not None else 0
n_fa = len(fa) if fa is not None else 0
n_vol = len(vol_abs) if vol_abs is not None else 0
n_vfc = int(overlap["has_volume_FA_and_cardiac"].sum()) if "has_volume_FA_and_cardiac" in overlap else 0
vol_status = vol_qc["status"].value_counts().to_dict() if vol_qc is not None and "status" in vol_qc else {}

readme = f"""GLP-1 mouse imaging analysis-ready tables

Current folder:
{BASE.resolve()}

Included tables:
- tables/GLP1_mouse_metadata_25.csv: cleaned mouse metadata
- tables/GLP1_regional_volume_absolute_mm3_25.csv: 332-ROI regional brain volumes in mm3
- tables/GLP1_regional_volume_normalized_25.csv: 332-ROI regional brain volumes normalized by total labeled brain volume
- tables/GLP1_regional_volume_bilateral_166_25.csv: bilateral 166-region volume table combining homologous left/right ROIs
- tables/GLP1_regional_volume_QC_25.csv: QC table for regional volume extraction
- tables/GLP1_regional_FA_25.csv: regional FA measurements
- tables/GLP1_regional_MD_25.csv: regional MD measurements
- tables/GLP1_regional_AD_25.csv: regional AD measurements
- tables/GLP1_regional_RD_25.csv: regional RD measurements
- tables/GLP1_connectome_cardiac_mwm_analysis_table_25.csv: merged connectome/cardiac/MWM analysis table
- tables/GLP1_connectome_global_QC_25.csv: global connectome QC table

Included documentation and summaries:
- docs/data_dictionary.csv
- docs/ROI_mapping_332.csv
- docs/overlap_mouse_level.csv
- docs/overlap_summary_by_treatment_genotype_sex.csv
- docs/unique_to_dataset_summary.csv
- docs/missing_data_summary.csv
- docs/brain_mask_volume_sanity_check.csv
- docs/genotype_comparison_regional_FA_GLP1.csv
- docs/genotype_comparison_regional_volume_normalized_GLP1.csv
- docs/genotype_comparison_regional_volume_absolute_TBV_adjusted_GLP1.csv
- docs/report_to_Alexandra.md
- docs/email_update_draft.txt

ROI coding:
- Original atlas labels 1-166 were mapped to ROI_1-ROI_166.
- Original atlas labels 1001-1166 were mapped to ROI_167-ROI_332.
- Homologous left/right regions differ by 1000 in the original atlas label coding.

Volume definitions:
- Absolute volume is measured in mm3 from native T2-space atlas labels.
- Normalized volume is regional volume divided by total labeled brain volume for that mouse.
- total_brain_label_volume_mm3 is the sum of compact 332 ROI volumes.

Current counts:
- metadata rows: {n_meta}
- regional FA rows: {n_fa}
- regional volume rows: {n_vol}
- mice with volume + FA + cardiac data: {n_vfc}
- volume QC status: {vol_status}
"""
(BASE / "README.txt").write_text(readme)

report = f"""# GLP-1 analysis-ready regional volume and FA deliverable

## Summary

I assembled the current analysis-ready regional brain-volume and FA dataset using the working BJ-derived data. The package is located at:

`{BASE.resolve()}`

The current dataset includes {n_meta} mice with metadata, {n_fa} mice with regional FA measurements, and {n_vol} mice with compact 332-ROI regional volume measurements. The regional volume extraction QC status is: {vol_status}. The overlap table indicates {n_vfc} mice currently have regional volume, regional FA, and cardiac data available.

## Regional volume and FA tables

The main analysis-ready tables are:

- `tables/GLP1_regional_volume_absolute_mm3_25.csv`
- `tables/GLP1_regional_volume_normalized_25.csv`
- `tables/GLP1_regional_FA_25.csv`
- `tables/GLP1_mouse_metadata_25.csv`

The volume tables use compact 332-ROI coding. Original atlas labels 1-166 map to ROI_1-ROI_166, and labels 1001-1166 map to ROI_167-ROI_332. The total brain label volume is calculated as the sum of all compact ROI volumes.

## QC and sanity checks

Volume QC is provided in `tables/GLP1_regional_volume_QC_25.csv`. The key checks are unexpected atlas labels and the sum of normalized regional volumes.

An independent brain-mask sanity check is provided in `docs/brain_mask_volume_sanity_check.csv`. If a usable brain mask was found for a mouse, the mask-derived volume is compared with the summed regional-label volume. If no mask was found, that mouse is flagged as missing mask volume.

## Overlap across datasets

Mouse-level dataset membership is provided in:

- `docs/overlap_mouse_level.csv`
- `docs/overlap_summary_by_treatment_genotype_sex.csv`
- `docs/unique_to_dataset_summary.csv`

These files identify which mice have regional volume, FA, metadata, connectome/QC, and cardiac data, and summarize overlap by treatment group, genotype, and sex.

## Missing data

Missing variables and missing mouse identifiers are summarized in:

- `docs/missing_data_summary.csv`

This file lists missingness for key experimental variables and major measurement tables.

## Genotype-comparison statistics

Genotype-comparison results within the GLP-1-treated group are provided in:

- `docs/genotype_comparison_regional_FA_GLP1.csv`
- `docs/genotype_comparison_regional_volume_normalized_GLP1.csv`
- `docs/genotype_comparison_regional_volume_absolute_TBV_adjusted_GLP1.csv`

Models used:

- Regional FA: `FA ~ Genotype + Sex + Age + Weight`
- Normalized regional volume: `volume_fraction ~ Genotype + Sex + Age + Weight`
- Absolute regional volume: `volume_mm3 ~ Genotype + total_brain_label_volume_mm3 + Sex + Age + Weight`

For normalized volume analyses, total brain volume is not included as a covariate because each regional volume is already normalized by total labeled brain volume. For absolute volume analyses, total brain label volume is included as a covariate.

Each result table includes genotype sample sizes, effect estimates, standard errors, uncorrected p-values, and FDR-adjusted q-values.

## Dataset consistency

Mouse identifiers are standardized using the `ID` column. Regional definitions are consistent across the current compact 332-ROI volume and FA tables after mapping raw atlas labels 1-166 and 1001-1166 to ROI_1-ROI_332. The anatomical region names are not currently available in the extracted tables; `docs/ROI_mapping_332.csv` records the ROI-to-raw-label mapping, and anatomical names can be added once the atlas lookup table is available.

## Items requiring follow-up

- Anatomical ROI names for the 332 compact ROIs are not available in the current extracted tables.
- Independent brain-mask volume is available only where a usable mask file is detected; missing mask-derived volumes should be followed up with the imaging-processing team.
- Revised data from Aidan can be incorporated later; the current deliverable uses the working BJ-derived dataset so that analyses can proceed.
- Any missing experimental variables identified in `docs/missing_data_summary.csv` should be requested from the appropriate metadata owner.
"""
(DOCS / "report_to_Alexandra.md").write_text(report)

email = f"""Dear Alexandra,

I have assembled the current analysis-ready regional brain-volume and FA package for the GLP-1 study using the working BJ-derived dataset.

The folder is:
{BASE.resolve()}

It includes the 332-ROI regional absolute and normalized volume tables, regional FA/MD/AD/RD tables, metadata, QC files, ROI mapping documentation, overlap summaries, missing-data summaries, brain-mask volume sanity checks where masks were available, and genotype-comparison statistics within the GLP-1-treated group.

The main documentation files are:
- README.txt
- docs/report_to_Alexandra.md
- docs/data_dictionary.csv
- docs/overlap_mouse_level.csv
- docs/overlap_summary_by_treatment_genotype_sex.csv
- docs/missing_data_summary.csv
- docs/genotype_comparison_regional_FA_GLP1.csv
- docs/genotype_comparison_regional_volume_normalized_GLP1.csv
- docs/genotype_comparison_regional_volume_absolute_TBV_adjusted_GLP1.csv

For the regional volume tables, I mapped the original atlas labels 1-166 and 1001-1166 into a compact 332-ROI coding so that the volume columns align with the existing ROI_1 to ROI_332 regional FA table. Absolute volumes are in mm3, and normalized volumes are divided by total labeled brain volume.

A few items still require follow-up: anatomical ROI names are not available in the current extracted tables, and independent brain-mask volume is only available where a usable mask file was detected. I listed these items and any missing variables in the report and missing-data summary.

Best,
Jimin
"""
(DOCS / "email_update_draft.txt").write_text(email)

print("Final deliverables written.")
print("README:", BASE / "README.txt")
print("Report:", DOCS / "report_to_Alexandra.md")
print("Email draft:", DOCS / "email_update_draft.txt")
print("Main docs:")
for p in sorted(DOCS.glob("*.csv")):
    print(" -", p)
