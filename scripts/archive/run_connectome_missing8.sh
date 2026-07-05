#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"
LABEL_DIR="/mnt/newStor/paros/paros_WORK/aidan/GLP1/labels"

cd "$ROOT"
mkdir -p logs

while read -r ID; do
  [ -z "$ID" ] && continue

  echo
  echo "========================================"
  echo "Connectome for $ID"
  echo "========================================"

  (
    PRE="${ROOT}/preproc_${ID}"

    T2="${ROOT}/all_niis/${ID}_T2.nii.gz"
    T2MASK=$(find "${ROOT}/masks" -maxdepth 1 -type f \( \
      -name "${ID}_mask_T2.nii.gz" -o \
      -name "${ID}_T2_mask.nii.gz" -o \
      -name "${ID}*mask*T2*.nii.gz" -o \
      -name "${ID}*T2*mask*.nii.gz" \
    \) | head -n 1)

    LABEL=$(find "$LABEL_DIR" -maxdepth 1 -type f \( \
      -name "${ID}*.nii.gz" -o \
      -name "${ID}*.nii" \
    \) | head -n 1)

    B0="${PRE}/${ID}_mean_b0.nii.gz"
    MAT="${PRE}/${ID}_T2_to_b0_dof6.mat"
    TRACKS="${PRE}/${ID}_tracks_100k_T2mask.tck"
    FA="${PRE}/${ID}_fa_T2mask.nii.gz"

    if [ ! -f "$T2" ] || [ ! -f "$T2MASK" ] || [ ! -f "$LABEL" ] || [ ! -f "$B0" ] || [ ! -f "$MAT" ] || [ ! -f "$TRACKS" ] || [ ! -f "$FA" ]; then
      echo "Missing required file for $ID"
      echo "T2=$T2"
      echo "T2MASK=$T2MASK"
      echo "LABEL=$LABEL"
      echo "B0=$B0"
      echo "MAT=$MAT"
      echo "TRACKS=$TRACKS"
      echo "FA=$FA"
      exit 1
    fi

    echo "Step 1: Auto-orient atlas label to native T2"
    python - "$ID" "$LABEL" "$T2" "$T2MASK" "${PRE}/${ID}_label_T2_ants_nativeT2_fixed.nii.gz" << 'PY'
import sys
import itertools
import numpy as np
import nibabel as nib

ID, label_f, t2_f, mask_f, out_f = sys.argv[1:]

lab_img = nib.load(label_f)
t2_img = nib.load(t2_f)
mask_img = nib.load(mask_f)

lab = lab_img.get_fdata()
mask = mask_img.get_fdata() > 0

target_shape = mask.shape

best = None
best_score = -1
best_perm = None
best_flips = None

for perm in itertools.permutations(range(3)):
    arr = np.transpose(lab, perm)
    if arr.shape != target_shape:
        continue

    for flips in itertools.product([False, True], repeat=3):
        test = arr.copy()
        for ax, doflip in enumerate(flips):
            if doflip:
                test = np.flip(test, axis=ax)

        label_vox = test > 0
        if mask.sum() == 0:
            score = 0
        else:
            score = np.logical_and(label_vox, mask).sum() / mask.sum()

        if score > best_score:
            best_score = score
            best = test
            best_perm = perm
            best_flips = flips

if best is None:
    raise RuntimeError(f"{ID}: no orientation candidate matched target shape {target_shape}")

best = best.astype(np.uint16)
hdr = t2_img.header.copy()
hdr.set_data_dtype(np.uint16)
nib.save(nib.Nifti1Image(best, t2_img.affine, hdr), out_f)

print(f"{ID}: best_perm={best_perm}, best_flips={best_flips}, T2_coverage={best_score:.6f}")
PY

    echo "Step 2: Transform fixed label to b0"
    flirt \
      -in "${PRE}/${ID}_label_T2_ants_nativeT2_fixed.nii.gz" \
      -ref "$B0" \
      -applyxfm \
      -init "$MAT" \
      -interp nearestneighbour \
      -out "${PRE}/${ID}_label_T2_ants_fixed_in_b0.nii.gz"

    echo "Step 3: Compact remap label"
    python - "$ID" \
      "${PRE}/${ID}_label_T2_ants_fixed_in_b0.nii.gz" \
      "${PRE}/${ID}_label_T2_ants_fixed_in_b0_compact.nii.gz" \
      "${PRE}/${ID}_label_fixed_compact_mapping.csv" << 'PY'
import sys
import numpy as np
import pandas as pd
import nibabel as nib

ID, in_f, out_f, map_f = sys.argv[1:]

img = nib.load(in_f)
data = img.get_fdata().astype(np.int64)

vals = np.unique(data)
vals = vals[vals > 0]

out = np.zeros(data.shape, dtype=np.uint16)
rows = []

for new_label, old_label in enumerate(vals, start=1):
    out[data == old_label] = new_label
    rows.append({"new_label": new_label, "old_label": int(old_label)})

hdr = img.header.copy()
hdr.set_data_dtype(np.uint16)
nib.save(nib.Nifti1Image(out, img.affine, hdr), out_f)
pd.DataFrame(rows).to_csv(map_f, index=False)

print(f"{ID}: compact labels = {len(vals)}")
PY

    echo "Step 4: Count connectome"
    tck2connectome \
      "$TRACKS" \
      "${PRE}/${ID}_label_T2_ants_fixed_in_b0_compact.nii.gz" \
      "${PRE}/${ID}_connectome_count_100k_fixed.csv" \
      -out_assignments "${PRE}/${ID}_assignments_100k_fixed.csv" \
      -symmetric \
      -zero_diagonal \
      -assignment_radial_search 1 \
      -force

    echo "Step 5: FA-weighted connectome"
    fslmaths "$FA" \
      -thr 0 -uthr 1 \
      "${PRE}/${ID}_fa_T2mask_clipped.nii.gz"

    tcksample \
      "$TRACKS" \
      "${PRE}/${ID}_fa_T2mask_clipped.nii.gz" \
      "${PRE}/${ID}_meanFA_per_streamline_100k_fixed_clipped.txt" \
      -stat_tck mean \
      -force

    tck2connectome \
      "$TRACKS" \
      "${PRE}/${ID}_label_T2_ants_fixed_in_b0_compact.nii.gz" \
      "${PRE}/${ID}_connectome_meanFA_100k_fixed_clipped.csv" \
      -scale_file "${PRE}/${ID}_meanFA_per_streamline_100k_fixed_clipped.txt" \
      -stat_edge mean \
      -symmetric \
      -zero_diagonal \
      -assignment_radial_search 1 \
      -force

    echo "DONE connectome for $ID"
  ) || {
    echo "FAILED connectome for $ID"
    continue
  }

done < ids_missing8_ready_for_connectome.txt
