import os
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"
GDIR = f"{ROOT}/connectomes_global_25"
if not os.path.isdir(GDIR):
    GDIR = f"{ROOT}/connectomes_global"

status = pd.read_csv(f"{ROOT}/cohort_27_file_status_from_sources.csv")
ids = status.loc[status["ready_for_connectome_inputs"], "ID"].astype(str).tolist()

gmap_f = f"{GDIR}/global_roi_mapping.csv"
if not os.path.exists(gmap_f):
    gmap_f = f"{ROOT}/connectomes_global/global_roi_mapping.csv"

gmap = pd.read_csv(gmap_f)
old_labels_global = gmap["old_label"].astype(int).tolist()

metrics = {
    "FA": "fa_T2mask_clipped.nii.gz",
    "MD": "md_T2mask.nii.gz",
    "AD": "ad_T2mask.nii.gz",
    "RD": "rd_T2mask.nii.gz",
}

all_outputs = {}

for metric, suffix in metrics.items():
    rows = []

    for ID in ids:
        pre = f"{ROOT}/preproc_{ID}"
        label_f = f"{pre}/{ID}_label_T2_ants_fixed_in_b0_compact.nii.gz"
        map_f = f"{pre}/{ID}_label_fixed_compact_mapping.csv"

        img_f = f"{pre}/{ID}_{suffix}"
        if metric == "FA" and not os.path.exists(img_f):
            # create clipped FA if missing
            raw_fa = f"{pre}/{ID}_fa_T2mask.nii.gz"
            img_f = raw_fa

        if not os.path.exists(label_f) or not os.path.exists(map_f) or not os.path.exists(img_f):
            print(f"WARNING missing files for {ID} {metric}")
            continue

        lab = nib.load(label_f).get_fdata().astype(int)
        img = nib.load(img_f).get_fdata().astype(float)

        # clip FA if needed
        if metric == "FA":
            img = np.clip(img, 0, 1)

        local_map = pd.read_csv(map_f)
        local_map.columns = [c.strip() for c in local_map.columns]
        if "new_label" not in local_map.columns or "old_label" not in local_map.columns:
            local_map.columns = ["new_label", "old_label"]

        old_to_new = dict(zip(local_map["old_label"].astype(int), local_map["new_label"].astype(int)))

        row = {"ID": ID}

        for old_label in old_labels_global:
            col = f"ROI_{old_label}_{metric}"
            if old_label in old_to_new:
                new_label = old_to_new[old_label]
                mask = lab == new_label
                vals = img[mask]
                vals = vals[np.isfinite(vals)]
                vals = vals[vals != 0]
                row[col] = float(vals.mean()) if vals.size > 0 else np.nan
            else:
                row[col] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    out_f = f"analysis_25/regional_{metric}_25.csv"
    out.to_csv(out_f, index=False)
    all_outputs[metric] = out.shape
    print(out_f, out.shape)

print("Done regional diffusion extraction.")
print(all_outputs)
