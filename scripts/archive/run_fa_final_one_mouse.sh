#!/bin/bash
set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1

ID=${1:?Usage: ./run_fa_final_one_mouse.sh MOUSE_ID}

DWI=all_niis/${ID}_DTI.nii.gz
BVAL=all_niis/${ID}_DTI.bval
BVEC=all_niis/${ID}_DTI.bvec
T2=all_niis/${ID}_T2.nii.gz
MASK_T2=masks/${ID}_mask_T2.nii.gz

OUT=final_fa_${ID}
mkdir -p ${OUT}

# Cleaned gradient files
CLEAN_BVAL=${OUT}/${ID}_DTI_clean.bval
CLEAN_BVEC=${OUT}/${ID}_DTI_clean.bvec

# QC / registration files
RAW_VOL0=${OUT}/${ID}_raw_DWI_vol0.nii.gz
RAW_LOWB_MEAN=${OUT}/${ID}_raw_lowb_mean.nii.gz
MEAN_B0=${OUT}/${ID}_mean_b0.nii.gz
T2_IN_B0=${OUT}/${ID}_T2_in_b0.nii.gz
T2_TO_B0=${OUT}/${ID}_T2_to_b0.mat
MASK_B0=${OUT}/${ID}_mask_in_b0.nii.gz

# Eddy files
INDEX=${OUT}/index.txt
ACQP=${OUT}/acqparams.txt
EDDY_PREFIX=${OUT}/${ID}_eddy
EDDY_DWI=${OUT}/${ID}_eddy.nii.gz
EDDY_BVEC=${OUT}/${ID}_eddy.eddy_rotated_bvecs
EDDY_BVEC_CLEAN=${OUT}/${ID}_eddy.eddy_rotated_bvecs_clean

# Tensor outputs
DTIFIT_PREFIX=${OUT}/${ID}_dtifit_clean
FA=${DTIFIT_PREFIX}_FA.nii.gz
MD=${DTIFIT_PREFIX}_MD.nii.gz
L1=${DTIFIT_PREFIX}_L1.nii.gz
L2=${DTIFIT_PREFIX}_L2.nii.gz
L3=${DTIFIT_PREFIX}_L3.nii.gz
AD=${OUT}/${ID}_dtifit_clean_AD.nii.gz
RD=${OUT}/${ID}_dtifit_clean_RD.nii.gz

# QC masks
FA_GT1=${OUT}/${ID}_FA_gt1_mask.nii.gz
L1_NEG=${OUT}/${ID}_L1_negative_mask.nii.gz
L2_NEG=${OUT}/${ID}_L2_negative_mask.nii.gz
L3_NEG=${OUT}/${ID}_L3_negative_mask.nii.gz
ANY_NEG=${OUT}/${ID}_any_negative_eigen_mask.nii.gz
INVALID=${OUT}/${ID}_invalid_tensor_mask.nii.gz
VALID=${OUT}/${ID}_valid_tensor_mask.nii.gz
FA_CLIPPED=${OUT}/${ID}_dtifit_clean_FA_clipped.nii.gz
FA_VALIDONLY=${OUT}/${ID}_dtifit_clean_FA_validonly.nii.gz

LOG=${OUT}/${ID}_FA_QC_summary.txt

echo "========================================"
echo "Running final FA pipeline for ${ID}"
echo "Output folder: ${OUT}"
echo "========================================"

echo ""
echo "Step 0: Check input files"
for f in ${DWI} ${BVAL} ${BVEC} ${T2} ${MASK_T2}; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing file $f"
    exit 1
  fi
  ls -lh "$f"
done

echo ""
echo "Step 1: Clean bval and bvec"
python - <<EOF
import numpy as np

bval_path = "${BVAL}"
bvec_path = "${BVEC}"
clean_bval_path = "${CLEAN_BVAL}"
clean_bvec_path = "${CLEAN_BVEC}"

bvals = np.loadtxt(bval_path)
bvecs = np.loadtxt(bvec_path)

if bvecs.shape[0] != 3:
    raise ValueError(f"bvec should have 3 rows, got {bvecs.shape}")

if bvecs.shape[1] != bvals.size:
    raise ValueError(
        f"bvec columns {bvecs.shape[1]} != bval entries {bvals.size}"
    )

lowb = bvals < 50

bvals_clean = bvals.copy()
bvals_clean[lowb] = 0

bvecs_clean = bvecs.copy()
bvecs_clean[:, lowb] = 0

np.savetxt(clean_bval_path, bvals_clean[None, :], fmt="%.6g")
np.savetxt(clean_bvec_path, bvecs_clean, fmt="%.10g")

print("Number of volumes:", bvals.size)
print("Low-b / b0 volumes:", int(lowb.sum()))
EOF

echo "Clean bval:"
cat ${CLEAN_BVAL}

echo ""
echo "Step 2: Generate raw DWI QC images and mean b0"

# Save first raw DWI volume for QC
fslroi ${DWI} ${RAW_VOL0} 0 1

# Generate mean low-b / b0 image in native DWI space
dwiextract ${DWI} \
  -fslgrad ${CLEAN_BVEC} ${CLEAN_BVAL} \
  -bzero - | \
mrmath - mean ${MEAN_B0} -axis 3 -force

# Keep an explicit QC copy
cp ${MEAN_B0} ${RAW_LOWB_MEAN}

echo "Raw DWI volume 0:"
ls -lh ${RAW_VOL0}
fslstats ${RAW_VOL0} -R

echo "Mean low-b / b0:"
ls -lh ${MEAN_B0}
fslstats ${MEAN_B0} -R

echo ""
echo "Step 3: Register T2 to mean b0"
flirt \
  -in ${T2} \
  -ref ${MEAN_B0} \
  -out ${T2_IN_B0} \
  -omat ${T2_TO_B0}

echo "T2 registration outputs:"
ls -lh ${T2_IN_B0}
ls -lh ${T2_TO_B0}

echo ""
echo "Step 4: Transform T2 mask to b0 space"
flirt \
  -in ${MASK_T2} \
  -ref ${MEAN_B0} \
  -applyxfm \
  -init ${T2_TO_B0} \
  -out ${MASK_B0} \
  -interp nearestneighbour

echo "Mask voxel count:"
fslstats ${MASK_B0} -V

echo ""
echo "Step 5: Create eddy index and acqparams"
NVOL=$(fslval ${DWI} dim4)

indx=""
for ((i=1; i<=${NVOL}; i++)); do
  indx="${indx} 1"
done
echo ${indx} > ${INDEX}

# Default / temporary acquisition parameter.
# For final production, confirm phase-encoding direction and readout time
# from the Bruker .method file.
echo "0 1 0 0.05" > ${ACQP}

echo "index.txt:"
cat ${INDEX}

echo "acqparams.txt:"
cat ${ACQP}

echo ""
echo "Step 6: Run FSL eddy"

if command -v eddy >/dev/null 2>&1; then
  EDDY_CMD=eddy
else
  echo "ERROR: eddy not found"
  exit 1
fi

${EDDY_CMD} \
  --imain=${DWI} \
  --mask=${MASK_B0} \
  --acqp=${ACQP} \
  --index=${INDEX} \
  --bvecs=${CLEAN_BVEC} \
  --bvals=${CLEAN_BVAL} \
  --out=${EDDY_PREFIX} \
  --repol || \
${EDDY_CMD} \
  --imain=${DWI} \
  --mask=${MASK_B0} \
  --acqp=${ACQP} \
  --index=${INDEX} \
  --bvecs=${CLEAN_BVEC} \
  --bvals=${CLEAN_BVAL} \
  --out=${EDDY_PREFIX}

echo "Eddy outputs:"
ls -lh ${EDDY_DWI}
ls -lh ${EDDY_BVEC}

echo ""
echo "Step 7: Clean eddy rotated bvec"
python - <<EOF
import numpy as np

bval_path = "${CLEAN_BVAL}"
bvec_path = "${EDDY_BVEC}"
out_path = "${EDDY_BVEC_CLEAN}"

bvals = np.loadtxt(bval_path)
bvecs = np.loadtxt(bvec_path)

if bvecs.shape[0] != 3:
    raise ValueError(f"eddy rotated bvec should have 3 rows, got {bvecs.shape}")

if bvecs.shape[1] != bvals.size:
    raise ValueError(
        f"eddy rotated bvec columns {bvecs.shape[1]} != bval entries {bvals.size}"
    )

bvecs = np.nan_to_num(bvecs, nan=0.0, posinf=0.0, neginf=0.0)
bvecs[:, bvals < 50] = 0

np.savetxt(out_path, bvecs, fmt="%.10g")

print("Cleaned eddy rotated bvec written.")
print("bvec shape:", bvecs.shape)
print("bval entries:", bvals.size)
EOF

echo "Check cleaned rotated bvec:"
awk 'NR==1 {print "bvec columns:", NF} END {print "bvec rows:", NR}' ${EDDY_BVEC_CLEAN}
awk '{print "bval entries:", NF}' ${CLEAN_BVAL}
fslval ${EDDY_DWI} dim4

echo ""
echo "Step 8: Tensor fitting with dtifit OLS"
dtifit \
  -k ${EDDY_DWI} \
  -o ${DTIFIT_PREFIX} \
  -m ${MASK_B0} \
  -r ${EDDY_BVEC_CLEAN} \
  -b ${CLEAN_BVAL}

echo ""
echo "Step 8b: Generate AD and RD maps"

# AD = axial diffusivity = L1
cp ${L1} ${AD}

# RD = radial diffusivity = (L2 + L3) / 2
fslmaths ${L2} \
  -add ${L3} \
  -div 2 \
  ${RD}

echo "AD output:"
ls -lh ${AD}

echo "RD output:"
ls -lh ${RD}

echo ""
echo "Step 9: Generate QC masks"

# FA > 1
fslmaths ${FA} -thr 1 -bin ${FA_GT1}

# Negative eigenvalue masks
fslmaths ${L1} -mul -1 -thr 0.000001 -bin ${L1_NEG}
fslmaths ${L2} -mul -1 -thr 0.000001 -bin ${L2_NEG}
fslmaths ${L3} -mul -1 -thr 0.000001 -bin ${L3_NEG}

# Any negative eigenvalue
fslmaths ${L1_NEG} \
  -add ${L2_NEG} \
  -add ${L3_NEG} \
  -bin ${ANY_NEG}

# Invalid tensor = FA > 1 or any negative eigenvalue
fslmaths ${FA_GT1} \
  -add ${ANY_NEG} \
  -bin ${INVALID}

# Valid tensor mask
fslmaths ${MASK_B0} \
  -sub ${INVALID} \
  -thr 0.5 \
  -bin ${VALID}

# Clipped FA for downstream statistics
fslmaths ${FA} \
  -thr 0 -uthr 1 \
  ${FA_CLIPPED}

# FA masked by valid tensor mask
fslmaths ${FA} \
  -mas ${VALID} \
  ${FA_VALIDONLY}

echo ""
echo "Step 10: QC summary"

MASK_VOX=$(fslstats ${MASK_B0} -V | awk '{print $1}')
FA_GT1_VOX=$(fslstats ${FA_GT1} -k ${MASK_B0} -V | awk '{print $1}')
L1_NEG_VOX=$(fslstats ${L1_NEG} -k ${MASK_B0} -V | awk '{print $1}')
L2_NEG_VOX=$(fslstats ${L2_NEG} -k ${MASK_B0} -V | awk '{print $1}')
L3_NEG_VOX=$(fslstats ${L3_NEG} -k ${MASK_B0} -V | awk '{print $1}')
INVALID_VOX=$(fslstats ${INVALID} -k ${MASK_B0} -V | awk '{print $1}')
VALID_VOX=$(fslstats ${VALID} -V | awk '{print $1}')

FA_MEAN_FULL=$(fslstats ${FA} -k ${MASK_B0} -M)
FA_MEAN_VALID=$(fslstats ${FA} -k ${VALID} -M)
FA_CLIPPED_MEAN_VALID=$(fslstats ${FA_CLIPPED} -k ${VALID} -M)
MD_MEAN_VALID=$(fslstats ${MD} -k ${VALID} -M)
AD_MEAN_VALID=$(fslstats ${AD} -k ${VALID} -M)
RD_MEAN_VALID=$(fslstats ${RD} -k ${VALID} -M)

python - <<EOF | tee ${LOG}
mask_vox = int("${MASK_VOX}")
fa_gt1 = int("${FA_GT1_VOX}")
l1_neg = int("${L1_NEG_VOX}")
l2_neg = int("${L2_NEG_VOX}")
l3_neg = int("${L3_NEG_VOX}")
invalid = int("${INVALID_VOX}")
valid = int("${VALID_VOX}")

def pct(x):
    return 100 * x / mask_vox if mask_vox > 0 else float("nan")

print("Mouse ID: ${ID}")
print("Mask voxels:", mask_vox)
print("Valid tensor voxels:", valid, f"({pct(valid):.2f}%)")
print("Invalid tensor voxels:", invalid, f"({pct(invalid):.2f}%)")
print("FA > 1 voxels:", fa_gt1, f"({pct(fa_gt1):.2f}%)")
print("Negative L1 voxels:", l1_neg, f"({pct(l1_neg):.2f}%)")
print("Negative L2 voxels:", l2_neg, f"({pct(l2_neg):.2f}%)")
print("Negative L3 voxels:", l3_neg, f"({pct(l3_neg):.2f}%)")
print("")
print("Mean FA in full mask: ${FA_MEAN_FULL}")
print("Mean FA in valid tensor mask: ${FA_MEAN_VALID}")
print("Mean clipped FA in valid tensor mask: ${FA_CLIPPED_MEAN_VALID}")
print("Mean MD in valid tensor mask: ${MD_MEAN_VALID}")
print("Mean AD in valid tensor mask: ${AD_MEAN_VALID}")
print("Mean RD in valid tensor mask: ${RD_MEAN_VALID}")
EOF

echo ""
echo "FA range:"
fslstats ${FA} -R

echo "Clipped FA range:"
fslstats ${FA_CLIPPED} -R

echo ""
echo "DONE"
echo "Main outputs:"
echo "Raw FA:        ${FA}"
echo "Clipped FA:    ${FA_CLIPPED}"
echo "MD:            ${MD}"
echo "AD:            ${AD}"
echo "RD:            ${RD}"
echo "Valid mask:    ${VALID}"
echo "Invalid mask:  ${INVALID}"
echo "Raw vol0 QC:   ${RAW_VOL0}"
echo "Mean b0 QC:    ${MEAN_B0}"
echo "QC summary:    ${LOG}"

echo ""
echo "QC visualization command:"
echo "fsleyes ${MEAN_B0} ${RAW_VOL0} ${MASK_B0} ${FA} ${INVALID}"
