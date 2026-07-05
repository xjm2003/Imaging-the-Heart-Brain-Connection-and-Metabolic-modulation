#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"
RAW_SRC="/mnt/newStor/paros/paros_MRI/GLP_1/all_niis"
MASK_SRC="/mnt/newStor/paros/paros_WORK/skull_stripping/GLP1_T2s"

cd "$ROOT"

mkdir -p all_niis masks logs

# sync source files into project folders
rsync -av --ignore-existing "$RAW_SRC"/*_DTI.nii.gz all_niis/
rsync -av --ignore-existing "$RAW_SRC"/*_DTI.bvec all_niis/
rsync -av --ignore-existing "$RAW_SRC"/*_DTI.bval all_niis/
rsync -av --ignore-existing "$RAW_SRC"/*_T2.nii.gz all_niis/
rsync -av --ignore-existing "$MASK_SRC"/*mask*.nii.gz masks/

while read -r ID; do
  [ -z "$ID" ] && continue

  echo
  echo "========================================"
  echo "Processing $ID"
  echo "========================================"

  (
    PRE="${ROOT}/preproc_${ID}"
    mkdir -p "$PRE"

    DWI_NII="${ROOT}/all_niis/${ID}_DTI.nii.gz"
    BVEC="${ROOT}/all_niis/${ID}_DTI.bvec"
    BVAL="${ROOT}/all_niis/${ID}_DTI.bval"
    T2="${ROOT}/all_niis/${ID}_T2.nii.gz"

    T2MASK=$(find "${ROOT}/masks" -maxdepth 1 -type f \( \
      -name "${ID}_mask_T2.nii.gz" -o \
      -name "${ID}_T2_mask.nii.gz" -o \
      -name "${ID}*mask*T2*.nii.gz" -o \
      -name "${ID}*T2*mask*.nii.gz" \
    \) | head -n 1)

    if [ ! -f "$DWI_NII" ] || [ ! -f "$BVEC" ] || [ ! -f "$BVAL" ] || [ ! -f "$T2" ] || [ ! -f "$T2MASK" ]; then
      echo "Missing input for $ID"
      echo "DWI=$DWI_NII"
      echo "BVEC=$BVEC"
      echo "BVAL=$BVAL"
      echo "T2=$T2"
      echo "T2MASK=$T2MASK"
      exit 1
    fi

    if [ -f "${PRE}/${ID}_tracks_100k_T2mask.tck" ]; then
      echo "Already has 100k T2mask tracks; skip tractography for $ID"
      exit 0
    fi

    echo "Step 1: Convert DWI to MRtrix mif"
    mrconvert "$DWI_NII" "${PRE}/${ID}_dwi_raw.mif" \
      -fslgrad "$BVEC" "$BVAL" \
      -force

    echo "Step 2: Denoise and degibbs"
    dwidenoise "${PRE}/${ID}_dwi_raw.mif" "${PRE}/${ID}_dwi_denoised.mif" -force
    mrdegibbs "${PRE}/${ID}_dwi_denoised.mif" "${PRE}/${ID}_dwi_degibbs.mif" -force

    echo "Step 3: Initial DWI mask for gradient check"
    dwi2mask "${PRE}/${ID}_dwi_degibbs.mif" "${PRE}/${ID}_dwi_mask.mif" -force

    echo "Step 4: dwigradcheck"
    if dwigradcheck "${PRE}/${ID}_dwi_degibbs.mif" \
      -mask "${PRE}/${ID}_dwi_mask.mif" \
      -export_grad_fsl "${PRE}/${ID}_checked.bvec" "${PRE}/${ID}_checked.bval" \
      -force; then

      mrconvert "${PRE}/${ID}_dwi_degibbs.mif" "${PRE}/${ID}_dwi_checked.mif" \
        -fslgrad "${PRE}/${ID}_checked.bvec" "${PRE}/${ID}_checked.bval" \
        -force
    else
      echo "WARNING: dwigradcheck failed for $ID; continuing with original gradients"
      mrconvert "${PRE}/${ID}_dwi_degibbs.mif" "${PRE}/${ID}_dwi_checked.mif" -force
    fi

    echo "Step 5: Mean b0"
    dwiextract "${PRE}/${ID}_dwi_checked.mif" - -bzero | \
      mrmath - mean "${PRE}/${ID}_mean_b0.mif" -axis 3 -force

    mrconvert "${PRE}/${ID}_mean_b0.mif" "${PRE}/${ID}_mean_b0.nii.gz" -force

    echo "Step 6: Register T2 to b0"
    flirt \
      -in "$T2" \
      -ref "${PRE}/${ID}_mean_b0.nii.gz" \
      -dof 6 \
      -omat "${PRE}/${ID}_T2_to_b0_dof6.mat" \
      -out "${PRE}/${ID}_T2_in_b0_dof6.nii.gz"

    echo "Step 7: Transform T2 mask to b0"
    flirt \
      -in "$T2MASK" \
      -ref "${PRE}/${ID}_mean_b0.nii.gz" \
      -applyxfm \
      -init "${PRE}/${ID}_T2_to_b0_dof6.mat" \
      -interp nearestneighbour \
      -out "${PRE}/${ID}_mask_from_T2_in_b0_raw.nii.gz"

    fslmaths "${PRE}/${ID}_mask_from_T2_in_b0_raw.nii.gz" \
      -thr 0.5 -bin \
      "${PRE}/${ID}_mask_from_T2_in_b0.nii.gz"

    mrconvert "${PRE}/${ID}_mask_from_T2_in_b0.nii.gz" \
      "${PRE}/${ID}_mask_from_T2_in_b0.mif" \
      -datatype bit \
      -force

    echo "Step 8: Tensor metrics with T2 mask"
    dwi2tensor "${PRE}/${ID}_dwi_checked.mif" "${PRE}/${ID}_dt_T2mask.mif" \
      -mask "${PRE}/${ID}_mask_from_T2_in_b0.mif" \
      -force

    tensor2metric "${PRE}/${ID}_dt_T2mask.mif" \
      -fa "${PRE}/${ID}_fa_T2mask.nii.gz" \
      -adc "${PRE}/${ID}_md_T2mask.nii.gz" \
      -ad "${PRE}/${ID}_ad_T2mask.nii.gz" \
      -rd "${PRE}/${ID}_rd_T2mask.nii.gz" \
      -force

    echo "Step 9: CSD response and FOD"
    dwi2response tournier "${PRE}/${ID}_dwi_checked.mif" "${PRE}/${ID}_response_T2mask.txt" \
      -mask "${PRE}/${ID}_mask_from_T2_in_b0.mif" \
      -force

    dwi2fod csd "${PRE}/${ID}_dwi_checked.mif" \
      "${PRE}/${ID}_response_T2mask.txt" \
      "${PRE}/${ID}_wmfod_T2mask.mif" \
      -mask "${PRE}/${ID}_mask_from_T2_in_b0.mif" \
      -force

    echo "Step 10: 100k T2mask tractography"
    tckgen "${PRE}/${ID}_wmfod_T2mask.mif" \
      "${PRE}/${ID}_tracks_100k_T2mask.tck" \
      -algorithm iFOD2 \
      -seed_image "${PRE}/${ID}_mask_from_T2_in_b0.mif" \
      -mask "${PRE}/${ID}_mask_from_T2_in_b0.mif" \
      -select 100000 \
      -cutoff 0.15 \
      -minlength 2 \
      -maxlength 40 \
      -angle 30 \
      -force

    echo "DONE preproc/tractography for $ID"
  ) || {
    echo "FAILED preproc/tractography for $ID"
    continue
  }

done < ids_missing8_ready_for_connectome.txt
