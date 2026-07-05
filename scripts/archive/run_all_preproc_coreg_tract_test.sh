#!/bin/bash
set -euo pipefail

ROOT=/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1
RAW=${ROOT}/all_niis
MASKDIR=${ROOT}/masks
LOGDIR=${ROOT}/logs_batch_preproc

mkdir -p ${LOGDIR}

echo "ROOT=${ROOT}"
echo "RAW=${RAW}"
echo "MASKDIR=${MASKDIR}"
echo "LOGDIR=${LOGDIR}"

for DWI in ${RAW}/*_DTI.nii.gz; do

  ID=$(basename ${DWI} _DTI.nii.gz)

  echo ""
  echo "===================================================="
  echo "Processing ${ID}"
  echo "===================================================="

  PRE=${ROOT}/preproc_${ID}
  COREG=${ROOT}/coreg_${ID}
  LOG=${LOGDIR}/${ID}.log

  mkdir -p ${PRE}
  mkdir -p ${COREG}

  BVEC=${RAW}/${ID}_DTI.bvec
  BVAL=${RAW}/${ID}_DTI.bval
  T2=${RAW}/${ID}_T2.nii.gz

  # Some mask names are inconsistent, so try both patterns.
  T2MASK=""
  if [ -f ${MASKDIR}/${ID}_mask_T2.nii.gz ]; then
    T2MASK=${MASKDIR}/${ID}_mask_T2.nii.gz
  elif [ -f ${MASKDIR}/${ID}_T2_mask.nii.gz ]; then
    T2MASK=${MASKDIR}/${ID}_T2_mask.nii.gz
  else
    echo "WARNING: T2 mask not found for ${ID}. Skipping T2 coreg." | tee -a ${LOG}
  fi

  {
    echo "Subject: ${ID}"
    echo "DWI: ${DWI}"
    echo "BVEC: ${BVEC}"
    echo "BVAL: ${BVAL}"
    echo "T2: ${T2}"
    echo "T2MASK: ${T2MASK}"
    echo "PRE: ${PRE}"
    echo "COREG: ${COREG}"

    echo ""
    echo "Step 0: Check required DWI files"

    for f in ${DWI} ${BVEC} ${BVAL}; do
      if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
      fi
    done

    echo ""
    echo "Step 0.1: DWI and bval/bvec summary"

    fslinfo ${DWI}

    echo "bval count:"
    awk '{print NF}' ${BVAL}

    echo "bvec count:"
    awk '{print NF}' ${BVEC}

    python3 - <<PY
import numpy as np
b = np.loadtxt("${BVAL}").reshape(-1)
print("n volumes:", b.size)
print("unique rounded b values:", np.unique(np.round(b, -1)))
print("n low-b < 50:", np.sum(b < 50))
print("n DWI >= 50:", np.sum(b >= 50))
print("low-b indices 0-based:", np.where(b < 50)[0])
print("low-b indices 1-based:", np.where(b < 50)[0] + 1)
print("mean nonzero b:", np.mean(b[b >= 50]) if np.any(b >= 50) else np.nan)
PY

    echo ""
    echo "Step 1: Convert raw DWI to MRtrix .mif"

    mrconvert \
      ${DWI} \
      ${PRE}/${ID}_dwi_raw.mif \
      -fslgrad ${BVEC} ${BVAL} \
      -force

    echo ""
    echo "Step 2: MP-PCA denoising"

    dwidenoise \
      ${PRE}/${ID}_dwi_raw.mif \
      ${PRE}/${ID}_dwi_denoised.mif \
      -noise ${PRE}/${ID}_noise.mif \
      -force

    echo ""
    echo "Step 3: Gibbs ringing correction"

    mrdegibbs \
      ${PRE}/${ID}_dwi_denoised.mif \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      -force

    echo ""
    echo "Step 4: Generate mean low-b image"

    LOWB_IDX=$(python3 - <<PY
import numpy as np
b = np.loadtxt("${BVAL}").reshape(-1)
idx = np.where(b < 50)[0]
if idx.size == 0:
    idx = np.argsort(b)[:4]
print(",".join(map(str, idx.tolist())))
PY
)

    echo "Using low-b volume indices: ${LOWB_IDX}"

    mrconvert \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      ${PRE}/${ID}_lowb_vols.mif \
      -coord 3 ${LOWB_IDX} \
      -force

    mrmath \
      ${PRE}/${ID}_lowb_vols.mif \
      mean \
      ${PRE}/${ID}_mean_b0.mif \
      -axis 3 \
      -force

    mrconvert \
      ${PRE}/${ID}_mean_b0.mif \
      ${PRE}/${ID}_mean_b0.nii.gz \
      -force

    echo ""
    echo "Step 5: Generate DWI mask using dwi2mask"

    dwi2mask \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      ${PRE}/${ID}_mask.mif \
      -force

    mrconvert \
      ${PRE}/${ID}_mask.mif \
      ${PRE}/${ID}_mask.nii.gz \
      -force

    echo ""
    echo "Step 6: Tensor fitting and tensor metrics using dwi2mask"

    dwi2tensor \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      ${PRE}/${ID}_dt.mif \
      -mask ${PRE}/${ID}_mask.mif \
      -force

    tensor2metric \
      ${PRE}/${ID}_dt.mif \
      -fa ${PRE}/${ID}_fa.mif \
      -adc ${PRE}/${ID}_md.mif \
      -ad ${PRE}/${ID}_ad.mif \
      -rd ${PRE}/${ID}_rd.mif \
      -force

    mrconvert ${PRE}/${ID}_fa.mif ${PRE}/${ID}_fa.nii.gz -force
    mrconvert ${PRE}/${ID}_md.mif ${PRE}/${ID}_md.nii.gz -force
    mrconvert ${PRE}/${ID}_ad.mif ${PRE}/${ID}_ad.nii.gz -force
    mrconvert ${PRE}/${ID}_rd.mif ${PRE}/${ID}_rd.nii.gz -force

    echo ""
    echo "Step 7: FOD and 100k test tractography using dwi2mask"

    dwi2response tournier \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      ${PRE}/${ID}_wm_response.txt \
      -mask ${PRE}/${ID}_mask.mif \
      -force

    dwi2fod csd \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      ${PRE}/${ID}_wm_response.txt \
      ${PRE}/${ID}_wmfod.mif \
      -mask ${PRE}/${ID}_mask.mif \
      -force

    tckgen \
      ${PRE}/${ID}_wmfod.mif \
      ${PRE}/${ID}_tracks_100k.tck \
      -algorithm iFOD2 \
      -seed_image ${PRE}/${ID}_mask.mif \
      -mask ${PRE}/${ID}_mask.mif \
      -select 100000 \
      -cutoff 0.1 \
      -minlength 0.5 \
      -maxlength 80 \
      -force

    echo ""
    echo "Step 8: T2 to b0 coregistration if T2 and T2 mask exist"

    if [ -f "${T2}" ] && [ -n "${T2MASK}" ] && [ -f "${T2MASK}" ]; then

      echo "Skull-strip T2"

      fslmaths \
        ${T2} \
        -mas ${T2MASK} \
        ${COREG}/${ID}_T2_brain.nii.gz

      echo "Register T2 brain to mean b0"

      flirt \
        -in ${COREG}/${ID}_T2_brain.nii.gz \
        -ref ${PRE}/${ID}_mean_b0.nii.gz \
        -out ${COREG}/${ID}_T2_brain_in_b0.nii.gz \
        -omat ${COREG}/${ID}_T2_to_b0.mat \
        -dof 6 \
        -cost mutualinfo

      echo "Transform T2 mask to b0 space"

      flirt \
        -in ${T2MASK} \
        -ref ${PRE}/${ID}_mean_b0.nii.gz \
        -out ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        -applyxfm \
        -init ${COREG}/${ID}_T2_to_b0.mat \
        -interp nearestneighbour

      mrconvert \
        ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        ${COREG}/${ID}_T2_mask_in_b0.mif \
        -force

      echo ""
      echo "Step 9: Recompute tensor/FOD/tracks using T2 mask in b0"

      mrconvert \
        ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      mrconvert \
        ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        ${PRE}/${ID}_mask_from_T2_in_b0.nii.gz \
        -force

      dwi2tensor \
        ${PRE}/${ID}_dwi_denoised_degibbs.mif \
        ${PRE}/${ID}_dt_T2mask.mif \
        -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      tensor2metric \
        ${PRE}/${ID}_dt_T2mask.mif \
        -fa ${PRE}/${ID}_fa_T2mask.mif \
        -adc ${PRE}/${ID}_md_T2mask.mif \
        -ad ${PRE}/${ID}_ad_T2mask.mif \
        -rd ${PRE}/${ID}_rd_T2mask.mif \
        -force

      mrconvert ${PRE}/${ID}_fa_T2mask.mif ${PRE}/${ID}_fa_T2mask.nii.gz -force
      mrconvert ${PRE}/${ID}_md_T2mask.mif ${PRE}/${ID}_md_T2mask.nii.gz -force
      mrconvert ${PRE}/${ID}_ad_T2mask.mif ${PRE}/${ID}_ad_T2mask.nii.gz -force
      mrconvert ${PRE}/${ID}_rd_T2mask.mif ${PRE}/${ID}_rd_T2mask.nii.gz -force

      dwi2response tournier \
        ${PRE}/${ID}_dwi_denoised_degibbs.mif \
        ${PRE}/${ID}_wm_response_T2mask.txt \
        -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      dwi2fod csd \
        ${PRE}/${ID}_dwi_denoised_degibbs.mif \
        ${PRE}/${ID}_wm_response_T2mask.txt \
        ${PRE}/${ID}_wmfod_T2mask.mif \
        -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      tckgen \
        ${PRE}/${ID}_wmfod_T2mask.mif \
        ${PRE}/${ID}_tracks_100k_T2mask.tck \
        -algorithm iFOD2 \
        -seed_image ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -select 100000 \
        -cutoff 0.1 \
        -minlength 0.5 \
        -maxlength 80 \
        -force

    else
      echo "WARNING: T2 or T2 mask missing for ${ID}; skipped T2 coreg and T2mask tractography."
    fi

    echo ""
    echo "Finished ${ID}"

  } 2>&1 | tee ${LOG}

done

echo ""
echo "All subjects finished."
echo "Logs are in: ${LOGDIR}"
