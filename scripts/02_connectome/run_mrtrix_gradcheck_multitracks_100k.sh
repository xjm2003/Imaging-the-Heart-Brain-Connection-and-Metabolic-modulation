0~#!/bin/bash
set -euo pipefail

ROOT=/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1
RAW=${ROOT}/all_niis
MASKDIR=${ROOT}/masks
LOGDIR=${ROOT}/logs_batch_preproc_fullT2_dof6

# Tractography outputs to generate for each subject.
# This will create 10k, 20k, 50k, and 100k .tck files for both dwi2mask and T2mask workflows.
TRACK_COUNTS=(10000 20000 50000 100000)

# Number of streamlines used internally by dwigradcheck for each gradient-orientation test.
# Increase to 5000 or 10000 if results are unstable, but it will run slower.
GRADCHECK_NUMBER=1000

mkdir -p ${LOGDIR}

echo "ROOT=${ROOT}"
echo "RAW=${RAW}"
echo "MASKDIR=${MASKDIR}"
echo "LOGDIR=${LOGDIR}"
echo "TRACK_COUNTS=${TRACK_COUNTS[*]}"
echo "GRADCHECK_NUMBER=${GRADCHECK_NUMBER}"

for DWI in ${RAW}/*_DTI.nii.gz; do

  ID=$(basename ${DWI} _DTI.nii.gz)

  echo ""
  echo "===================================================="
  echo "Processing ${ID}"
  echo "===================================================="

  PRE=${ROOT}/preproc_${ID}
  COREG=${ROOT}/coreg_${ID}
  REG=${COREG}/reg_tests
  LOG=${LOGDIR}/${ID}.log

  mkdir -p ${PRE}
  mkdir -p ${COREG}
  mkdir -p ${REG}

  BVEC=${RAW}/${ID}_DTI.bvec
  BVAL=${RAW}/${ID}_DTI.bval
  T2=${RAW}/${ID}_T2.nii.gz

  BVEC_GRADCHECK=${PRE}/${ID}_DTI_gradcheck.bvec
  BVAL_GRADCHECK=${PRE}/${ID}_DTI_gradcheck.bval
  DWI_GRADCHECK=${PRE}/${ID}_dwi_denoised_degibbs_gradcheck.mif

  T2MASK=""
  if [ -f ${MASKDIR}/${ID}_mask_T2.nii.gz ]; then
    T2MASK=${MASKDIR}/${ID}_mask_T2.nii.gz
  elif [ -f ${MASKDIR}/${ID}_T2_mask.nii.gz ]; then
    T2MASK=${MASKDIR}/${ID}_T2_mask.nii.gz
  else
    echo "WARNING: T2 mask not found for ${ID}. T2-mask-based steps will be skipped." | tee -a ${LOG}
  fi

  {
    echo "Subject: ${ID}"
    echo "DWI: ${DWI}"
    echo "BVEC: ${BVEC}"
    echo "BVAL: ${BVAL}"
    echo "BVEC_GRADCHECK: ${BVEC_GRADCHECK}"
    echo "BVAL_GRADCHECK: ${BVAL_GRADCHECK}"
    echo "DWI_GRADCHECK: ${DWI_GRADCHECK}"
    echo "T2: ${T2}"
    echo "T2MASK: ${T2MASK}"
    echo "PRE: ${PRE}"
    echo "COREG: ${COREG}"
    echo "REG: ${REG}"

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
    echo "Step 3.1: Check and correct diffusion gradient orientation with dwigradcheck"

    dwigradcheck \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      -number ${GRADCHECK_NUMBER} \
      -export_grad_fsl ${BVEC_GRADCHECK} ${BVAL_GRADCHECK} \
      -force

    echo "Re-import corrected gradient table into preprocessed DWI .mif"

    mrconvert \
      ${PRE}/${ID}_dwi_denoised_degibbs.mif \
      ${DWI_GRADCHECK} \
      -fslgrad ${BVEC_GRADCHECK} ${BVAL_GRADCHECK} \
      -force

    DWI_PROC=${DWI_GRADCHECK}
    BVEC_PROC=${BVEC_GRADCHECK}
    BVAL_PROC=${BVAL_GRADCHECK}

    echo "Corrected gradient summary from MRtrix header:"
    mrinfo ${DWI_PROC} -shell_bvalues -shell_sizes

    echo ""
    echo "Step 4: Generate mean low-b / pseudo-b0 image"

    LOWB_IDX=$(python3 - <<PY
import numpy as np
b = np.loadtxt("${BVAL_PROC}").reshape(-1)
idx = np.where(b < 50)[0]
if idx.size == 0:
    idx = np.argsort(b)[:4]
print(",".join(map(str, idx.tolist())))
PY
)

    echo "Using low-b volume indices: ${LOWB_IDX}"

    mrconvert \
      ${DWI_PROC} \
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

    B0=${PRE}/${ID}_mean_b0.nii.gz

    echo ""
    echo "Step 5: Generate DWI mask using dwi2mask for QC comparison"

    dwi2mask \
      ${DWI_PROC} \
      ${PRE}/${ID}_mask.mif \
      -force

    mrconvert \
      ${PRE}/${ID}_mask.mif \
      ${PRE}/${ID}_mask.nii.gz \
      -force

    echo ""
    echo "Step 6: Tensor metrics using dwi2mask, QC-only version"

    dwi2tensor \
      ${DWI_PROC} \
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
    echo "Step 7: FOD and multi-count tractography using dwi2mask, QC-only version"

    dwi2response tournier \
      ${DWI_PROC} \
      ${PRE}/${ID}_wm_response.txt \
      -mask ${PRE}/${ID}_mask.mif \
      -force

    dwi2fod csd \
      ${DWI_PROC} \
      ${PRE}/${ID}_wm_response.txt \
      ${PRE}/${ID}_wmfod.mif \
      -mask ${PRE}/${ID}_mask.mif \
      -force

    for NTRACKS in "${TRACK_COUNTS[@]}"; do
      LABEL=$((NTRACKS / 1000))k
      echo "Generating ${LABEL} tracks using dwi2mask: ${PRE}/${ID}_tracks_${LABEL}.tck"

      tckgen \
        ${PRE}/${ID}_wmfod.mif \
        ${PRE}/${ID}_tracks_${LABEL}.tck \
        -algorithm iFOD2 \
        -seed_image ${PRE}/${ID}_mask.mif \
        -mask ${PRE}/${ID}_mask.mif \
        -select ${NTRACKS} \
        -cutoff 0.1 \
        -minlength 0.5 \
        -maxlength 80 \
        -force

      tckinfo ${PRE}/${ID}_tracks_${LABEL}.tck
    done

    echo ""
    echo "Step 8: T2 to b0 registration using FULL T2, 6 DOF"

    if [ -f "${T2}" ] && [ -n "${T2MASK}" ] && [ -f "${T2MASK}" ]; then

      echo "Step 8.1: Create skull-stripped T2 only for QC, not for registration"

      fslmaths \
        ${T2} \
        -mas ${T2MASK} \
        ${COREG}/${ID}_T2_brain_QC.nii.gz

      echo "Step 8.2: Register FULL T2 to mean b0, 6 DOF"

      flirt \
        -in ${T2} \
        -ref ${B0} \
        -out ${COREG}/${ID}_T2_full_in_b0.nii.gz \
        -omat ${COREG}/${ID}_T2_to_b0.mat \
        -dof 6 \
        -cost mutualinfo \
        -searchrx -180 180 \
        -searchry -180 180 \
        -searchrz -180 180

      cp ${COREG}/${ID}_T2_full_in_b0.nii.gz \
         ${COREG}/${ID}_T2_brain_in_b0.nii.gz

      cp ${COREG}/${ID}_T2_full_in_b0.nii.gz \
         ${REG}/${ID}_T2_full_in_b0_dof6.nii.gz

      cp ${COREG}/${ID}_T2_to_b0.mat \
         ${REG}/${ID}_T2_full_to_b0_dof6.mat

      echo "Step 8.3: Apply full-T2-to-b0 transform to manual T2 mask"

      flirt \
        -in ${T2MASK} \
        -ref ${B0} \
        -out ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        -applyxfm \
        -init ${COREG}/${ID}_T2_to_b0.mat \
        -interp nearestneighbour

      cp ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
         ${REG}/${ID}_T2mask_full_dof6_in_b0.nii.gz

      mrconvert \
        ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        ${COREG}/${ID}_T2_mask_in_b0.mif \
        -force

      echo ""
      echo "Step 9: Generate MRtrix mask from manual T2 mask in b0"

      mrconvert \
        ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      mrconvert \
        ${COREG}/${ID}_T2_mask_in_b0.nii.gz \
        ${PRE}/${ID}_mask_from_T2_in_b0.nii.gz \
        -force

      echo ""
      echo "Step 10: Recompute tensor metrics using manual T2 mask in b0"

      dwi2tensor \
        ${DWI_PROC} \
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

      echo ""
      echo "Step 11: FOD and multi-count tractography using manual T2 mask in b0"

      dwi2response tournier \
        ${DWI_PROC} \
        ${PRE}/${ID}_wm_response_T2mask.txt \
        -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      dwi2fod csd \
        ${DWI_PROC} \
        ${PRE}/${ID}_wm_response_T2mask.txt \
        ${PRE}/${ID}_wmfod_T2mask.mif \
        -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
        -force

      for NTRACKS in "${TRACK_COUNTS[@]}"; do
        LABEL=$((NTRACKS / 1000))k
        echo "Generating ${LABEL} tracks using T2mask: ${PRE}/${ID}_tracks_${LABEL}_T2mask.tck"

        tckgen \
          ${PRE}/${ID}_wmfod_T2mask.mif \
          ${PRE}/${ID}_tracks_${LABEL}_T2mask.tck \
          -algorithm iFOD2 \
          -seed_image ${PRE}/${ID}_mask_from_T2_in_b0.mif \
          -mask ${PRE}/${ID}_mask_from_T2_in_b0.mif \
          -select ${NTRACKS} \
          -cutoff 0.1 \
          -minlength 0.5 \
          -maxlength 80 \
          -force

        tckinfo ${PRE}/${ID}_tracks_${LABEL}_T2mask.tck
      done

    else
      echo "WARNING: T2 or manual T2 mask missing for ${ID}; skipped T2-to-b0 and T2mask tractography."
    fi

    echo ""
    echo "Finished ${ID}"

    echo ""
    echo "Primary QC files:"
    echo "  ${PRE}/${ID}_mean_b0.nii.gz"
    echo "  ${COREG}/${ID}_T2_full_in_b0.nii.gz"
    echo "  ${COREG}/${ID}_T2_mask_in_b0.nii.gz"
    echo "  ${PRE}/${ID}_mask_from_T2_in_b0.nii.gz"
    echo "  ${PRE}/${ID}_fa_T2mask.nii.gz"
    echo "  ${PRE}/${ID}_DTI_gradcheck.bvec"
    echo "  ${PRE}/${ID}_DTI_gradcheck.bval"
    echo "  ${PRE}/${ID}_tracks_10k_T2mask.tck"
    echo "  ${PRE}/${ID}_tracks_20k_T2mask.tck"
    echo "  ${PRE}/${ID}_tracks_50k_T2mask.tck"
    echo "  ${PRE}/${ID}_tracks_100k_T2mask.tck"

  } 2>&1 | tee ${LOG}

done

echo ""
echo "All subjects finished."
echo "Logs are in: ${LOGDIR}"1~
