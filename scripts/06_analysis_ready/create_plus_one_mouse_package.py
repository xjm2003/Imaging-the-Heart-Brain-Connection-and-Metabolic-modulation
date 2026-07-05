from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = Path("/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1")
OLD = ROOT / "analysis_ready_regional_FA_volume_20260624"
NEW = ROOT / "analysis_ready_regional_FA_volume_20260624_plus_B25122904"
TABLES = NEW / "tables"
DOCS = NEW / "docs"
ID = "B25122904"

if not TABLES.exists():
    shutil.copytree(OLD, NEW, dirs_exist_ok=True)

TABLES.mkdir(exist_ok=True, parents=True)
DOCS.mkdir(exist_ok=True, parents=True)

# ---------- metadata ----------
old_meta = pd.read_csv(OLD / "tables/GLP1_mouse_metadata_25.csv")
src = pd.read_csv(ROOT / "metadata_sources/metadata_appended.csv")

hit = src[src.astype(str).apply(lambda r: r.str.contains(ID, na=False).any(), axis=1)]
if hit.empty:
    raise RuntimeError(f"No metadata row found for {ID}")

r = hit.iloc[0]
new_row = {c: np.nan for c in old_meta.columns}

for c in old_meta.columns:
    if c in src.columns:
        new_row[c] = r[c]

new_row["ID"] = ID
new_row["ARunno"] = r.get("ARunno", ID)
new_row["BadeaID"] = r.get("BadeaID", np.nan)
new_row["Genotype"] = r.get("Genotype", np.nan)
new_row["Sex"] = r.get("Sex", np.nan)
new_row["Diet"] = r.get("Diet", np.nan)
new_row["Weight"] = r.get("Weight", np.nan)
new_row["Age_Months"] = r.get("Age_Months", np.nan)

if "TreatmentGroup" in new_row and pd.isna(new_row["TreatmentGroup"]):
    new_row["TreatmentGroup"] = r.get("Group", r.get("Diet", np.nan))
if "Group" in new_row and pd.isna(new_row["Group"]):
    new_row["Group"] = r.get("Group", r.get("Diet", np.nan))
if "NameGroup" in new_row and pd.isna(new_row["NameGroup"]):
    new_row["NameGroup"] = r.get("Group", r.get("Diet", np.nan))

meta26 = old_meta[old_meta["ID"].astype(str) != ID].copy()
meta26 = pd.concat([meta26, pd.DataFrame([new_row])], ignore_index=True)
meta26.to_csv(TABLES / "GLP1_mouse_metadata_26_plus_B25122904.csv", index=False)

# ---------- regional diffusion append ----------
GDIR = ROOT / "connectomes_global_25"
if not GDIR.exists():
    GDIR = ROOT / "connectomes_global"

gmap = pd.read_csv(GDIR / "global_roi_mapping.csv")
global_old_labels = gmap["old_label"].astype(int).tolist()

pre = ROOT / f"preproc_{ID}"
label_f = pre / f"{ID}_label_T2_ants_fixed_in_b0_compact.nii.gz"
map_f = pre / f"{ID}_label_fixed_compact_mapping.csv"

lab = nib.load(label_f).get_fdata().astype(int)
local_map = pd.read_csv(map_f)
local_map.columns = [c.strip() for c in local_map.columns]
if "new_label" not in local_map.columns or "old_label" not in local_map.columns:
    local_map.columns = ["new_label", "old_label"]

old_to_new = dict(zip(local_map["old_label"].astype(int), local_map["new_label"].astype(int)))

metric_suffix = {
    "FA": "fa_T2mask_clipped.nii.gz",
    "MD": "md_T2mask.nii.gz",
    "AD": "ad_T2mask.nii.gz",
    "RD": "rd_T2mask.nii.gz",
}

for metric, suffix in metric_suffix.items():
    old_table = pd.read_csv(OLD / f"tables/GLP1_regional_{metric}_25.csv")

    img_f = pre / f"{ID}_{suffix}"
    if metric == "FA" and not img_f.exists():
        img_f = pre / f"{ID}_fa_T2mask.nii.gz"

    img = nib.load(img_f).get_fdata().astype(float)
    if metric == "FA":
        img = np.clip(img, 0, 1)

    row = {"ID": ID}

    for col in old_table.columns:
        if col == "ID":
            continue

        m = re.match(r"ROI_(\d+)_" + metric + r"$", col)
        if not m:
            row[col] = np.nan
            continue

        roi_code = int(m.group(1))

        candidate_old_labels = [roi_code]
        if 167 <= roi_code <= 332:
            candidate_old_labels.append(1000 + (roi_code - 166))

        val = np.nan
        for old_label in candidate_old_labels:
            if old_label in old_to_new:
                new_label = old_to_new[old_label]
                vals = img[lab == new_label]
                vals = vals[np.isfinite(vals)]
                vals = vals[vals != 0]
                val = float(vals.mean()) if vals.size > 0 else np.nan
                break

        row[col] = val

    out = old_table[old_table["ID"].astype(str) != ID].copy()
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(TABLES / f"GLP1_regional_{metric}_26_plus_B25122904.csv", index=False)

# ---------- regional volume append from native T2 labels ----------
native_label = pre / f"{ID}_label_T2_ants_nativeT2_fixed.nii.gz"
img = nib.load(native_label)
data = np.asanyarray(img.dataobj)
voxel_mm3 = abs(np.prod(img.header.get_zooms()[:3]))

def raw_to_roi(label):
    label = int(label)
    if 1 <= label <= 166:
        return label
    if 1001 <= label <= 1166:
        return 166 + (label - 1000)
    return None

labels, counts = np.unique(data, return_counts=True)
roi_vol = {i: 0.0 for i in range(1, 333)}
unexpected = []

for lab_id, cnt in zip(labels, counts):
    lab_id = int(lab_id)
    if lab_id == 0:
        continue
    roi = raw_to_roi(lab_id)
    if roi is None:
        unexpected.append(lab_id)
    else:
        roi_vol[roi] += int(cnt) * voxel_mm3

total = sum(roi_vol.values())

abs_row = {"ID": ID, "total_brain_label_volume_mm3": total}
norm_row = {"ID": ID, "total_brain_label_volume_mm3": total}
bilat_row = {"ID": ID, "total_brain_label_volume_mm3": total}

for roi in range(1, 333):
    abs_row[f"ROI_{roi}_volume_mm3"] = roi_vol[roi]
    norm_row[f"ROI_{roi}_volume_fraction"] = roi_vol[roi] / total if total else np.nan

for k in range(1, 167):
    v = roi_vol[k] + roi_vol[166 + k]
    bilat_row[f"Region_{k}_bilateral_volume_mm3"] = v
    bilat_row[f"Region_{k}_bilateral_volume_fraction"] = v / total if total else np.nan

qc_row = {
    "ID": ID,
    "status": "ok" if not unexpected else "unexpected_labels",
    "label_file": str(native_label.relative_to(ROOT)),
    "voxel_volume_mm3": voxel_mm3,
    "n_expected_roi_nonzero": sum(v > 0 for v in roi_vol.values()),
    "n_unexpected_labels": len(unexpected),
    "unexpected_labels": ";".join(map(str, unexpected)),
    "total_brain_label_volume_mm3": total,
    "normalized_sum": sum((roi_vol[i] / total) for i in roi_vol) if total else np.nan,
}

volume_files = [
    ("GLP1_regional_volume_absolute_mm3_25.csv", "GLP1_regional_volume_absolute_mm3_26_plus_B25122904.csv", abs_row),
    ("GLP1_regional_volume_normalized_25.csv", "GLP1_regional_volume_normalized_26_plus_B25122904.csv", norm_row),
    ("GLP1_regional_volume_bilateral_166_25.csv", "GLP1_regional_volume_bilateral_166_26_plus_B25122904.csv", bilat_row),
    ("GLP1_regional_volume_QC_25.csv", "GLP1_regional_volume_QC_26_plus_B25122904.csv", qc_row),
]

for old_name, new_name, row in volume_files:
    df = pd.read_csv(OLD / "tables" / old_name)
    df = df[df["ID"].astype(str) != ID].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TABLES / new_name, index=False)

# ---------- B25122904 connectome QC ----------
qc_src = ROOT / "connectome_QC_summary_autoorient.csv"
if qc_src.exists():
    qc = pd.read_csv(qc_src)
    qc[qc["ID"].astype(str) == ID].to_csv(TABLES / "GLP1_connectome_QC_B25122904.csv", index=False)

# ---------- overlap summary ----------
overlap = pd.DataFrame({
    "ID": meta26["ID"].astype(str),
    "has_metadata": True,
})
overlap["has_regional_FA"] = overlap["ID"].isin(pd.read_csv(TABLES / "GLP1_regional_FA_26_plus_B25122904.csv")["ID"].astype(str))
overlap["has_regional_volume"] = overlap["ID"].isin(pd.read_csv(TABLES / "GLP1_regional_volume_absolute_mm3_26_plus_B25122904.csv")["ID"].astype(str))

old_cardiac = pd.read_csv(OLD / "tables/GLP1_connectome_cardiac_mwm_analysis_table_25.csv")
cardiac_ids = set(old_cardiac["ID"].astype(str))
overlap["has_cardiac_or_mwm_merged_data"] = overlap["ID"].isin(cardiac_ids)

keep = [c for c in ["ID", "TreatmentGroup", "Group", "NameGroup", "Diet", "Sex", "Genotype", "Weight", "Age_Months"] if c in meta26.columns]
overlap = overlap.merge(meta26[keep], on="ID", how="left")
overlap.to_csv(DOCS / "overlap_mouse_level_26_plus_B25122904.csv", index=False)

summary = overlap.groupby(
    [c for c in ["Group", "Genotype", "Sex"] if c in overlap.columns],
    dropna=False
).agg(
    n_mice=("ID", "count"),
    n_with_FA=("has_regional_FA", "sum"),
    n_with_volume=("has_regional_volume", "sum"),
    n_with_cardiac_or_mwm=("has_cardiac_or_mwm_merged_data", "sum"),
).reset_index()
summary.to_csv(DOCS / "overlap_summary_26_plus_B25122904.csv", index=False)

report = f"""# B25122904 plus package update

B25122904 has been added as an additional imaging mouse.

Metadata:
- BadeaID: {new_row.get('BadeaID')}
- Genotype: {new_row.get('Genotype')}
- Sex: {new_row.get('Sex')}
- Diet: {new_row.get('Diet')}
- Group: {new_row.get('Group')}
- Weight: {new_row.get('Weight')}
- Age_Months: {new_row.get('Age_Months')}

Imaging status:
- DWI/T2 preprocessing: complete
- T2-mask tensor metrics: complete
- 100k tractography: complete
- Atlas label auto-orientation and connectome construction: complete
- Regional FA/MD/AD/RD appended to 26-mouse tables
- Regional absolute and normalized volume appended to 26-mouse tables

Cardiac status:
- B25122904 is not present in the current cardiac/MWM merged table.
- It should be treated as imaging + metadata only until cardiac availability is confirmed.

Main 26-mouse files:
- tables/GLP1_mouse_metadata_26_plus_B25122904.csv
- tables/GLP1_regional_FA_26_plus_B25122904.csv
- tables/GLP1_regional_MD_26_plus_B25122904.csv
- tables/GLP1_regional_AD_26_plus_B25122904.csv
- tables/GLP1_regional_RD_26_plus_B25122904.csv
- tables/GLP1_regional_volume_absolute_mm3_26_plus_B25122904.csv
- tables/GLP1_regional_volume_normalized_26_plus_B25122904.csv
- tables/GLP1_regional_volume_bilateral_166_26_plus_B25122904.csv
- tables/GLP1_regional_volume_QC_26_plus_B25122904.csv
- docs/overlap_mouse_level_26_plus_B25122904.csv
- docs/overlap_summary_26_plus_B25122904.csv
"""
(DOCS / "B25122904_plus_update_report.md").write_text(report)

print("Done.")
print("New package:", NEW)
