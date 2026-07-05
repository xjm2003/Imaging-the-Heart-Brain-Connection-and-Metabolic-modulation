#!/bin/bash
set -uo pipefail

ROOT=/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1
LABELDIR=/mnt/newStor/paros/paros_WORK/aidan/GLP1/labels
IDLIST=${ROOT}/ids_connectome_available.txt
LOGDIR=${ROOT}/logs_connectome_autoorient
QC=${ROOT}/connectome_QC_summary_autoorient.csv

mkdir -p ${LOGDIR}

echo "ID,best_perm,best_flips,T2_coverage,B0_coverage,label_outside_b0,N_ROI,total_streamlines,both_unassigned,one_unassigned,same_node,between_node,between_fraction,isolated_nodes,nonzero_edges,total_matrix_sum,mean_FA,max_FA" > ${QC}

process_id() (
  set -euo pipefail

  ID=$1

  echo ""
  echo "===================================================="
  echo "Processing ${ID}"
  echo "===================================================="

  PRE=${ROOT}/preproc_${ID}
  COREG=${ROOT}/coreg_${ID}
  REG=${COREG}/reg_tests

  LABEL=${LABELDIR}/${ID}_label_T2_ants.nii.gz

  if [ -f ${ROOT}/GLP1_T2s/${ID}_T2.nii.gz ]; then
    T2=${ROOT}/GLP1_T2s/${ID}_T2.nii.gz
  elif [ -f ${ROOT}/all_niis/${ID}_T2.nii.gz ]; then
    T2=${ROOT}/all_niis/${ID}_T2.nii.gz
  else
    T2=""
  fi

  if [ -f ${ROOT}/GLP1_T2s/${ID}_T2_mask.nii.gz ]; then
    T2MASK=${ROOT}/GLP1_T2s/${ID}_T2_mask.nii.gz
  elif [ -f ${ROOT}/GLP1_T2s/${ID}_mask_T2.nii.gz ]; then
    T2MASK=${ROOT}/GLP1_T2s/${ID}_mask_T2.nii.gz
  elif [ -f ${ROOT}/masks/${ID}_T2_mask.nii.gz ]; then
    T2MASK=${ROOT}/masks/${ID}_T2_mask.nii.gz
  elif [ -f ${ROOT}/masks/${ID}_mask_T2.nii.gz ]; then
    T2MASK=${ROOT}/masks/${ID}_mask_T2.nii.gz
  else
    T2MASK=""
  fi

  B0=${PRE}/${ID}_mean_b0.nii.gz
  B0MASK=${PRE}/${ID}_mask_from_T2_in_b0.nii.gz
  TRACKS=${PRE}/${ID}_tracks_100k_T2mask.tck
  FA=${PRE}/${ID}_fa_T2mask.nii.gz
  MAT=${REG}/${ID}_T2_full_to_b0_dof6.mat

  LABEL_NATIVE_FIXED=${PRE}/${ID}_label_T2_ants_nativeT2_fixed.nii.gz
  ORIENT_INFO=${PRE}/${ID}_label_autoorient_info.csv

  LABEL_B0=${PRE}/${ID}_label_T2_ants_fixed_in_b0.nii.gz
  LABEL_COMPACT=${PRE}/${ID}_label_T2_ants_fixed_in_b0_compact.nii.gz
  MAPFILE=${PRE}/${ID}_label_fixed_compact_mapping.csv
  COMPACT_INFO=${PRE}/${ID}_label_compact_info.csv

  COUNT=${PRE}/${ID}_connectome_count_100k_fixed.csv
  ASSIGN=${PRE}/${ID}_assignments_100k_fixed.csv

  FA_CLIP=${PRE}/${ID}_fa_T2mask_clipped.nii.gz
  MEANFA=${PRE}/${ID}_meanFA_per_streamline_100k_fixed_clipped.txt
  FACONN=${PRE}/${ID}_connectome_meanFA_100k_fixed_clipped.csv

  QC_VALUES=${PRE}/${ID}_connectome_qc_values.csv

  echo "Checking required files..."

  MISSING=0
  for f in ${LABEL} ${B0} ${B0MASK} ${TRACKS} ${FA} ${MAT}; do
    if [ ! -f "$f" ]; then
      echo "ERROR: missing $f"
      MISSING=1
    fi
  done

  if [ -z "${T2}" ] || [ ! -f "${T2}" ]; then
    echo "ERROR: missing T2 for ${ID}"
    MISSING=1
  fi

  if [ -z "${T2MASK}" ] || [ ! -f "${T2MASK}" ]; then
    echo "ERROR: missing T2 mask for ${ID}"
    MISSING=1
  fi

  if [ ${MISSING} -ne 0 ]; then
    echo "ERROR: required files missing for ${ID}; skip."
    exit 1
  fi

  echo "LABEL=${LABEL}"
  echo "T2=${T2}"
  echo "T2MASK=${T2MASK}"
  echo "B0=${B0}"
  echo "B0MASK=${B0MASK}"
  echo "TRACKS=${TRACKS}"
  echo "FA=${FA}"
  echo "MAT=${MAT}"

  echo ""
  echo "Step 1: Auto-orient atlas label into native T2 space"

  python3 - "${LABEL}" "${T2}" "${T2MASK}" "${LABEL_NATIVE_FIXED}" "${ORIENT_INFO}" << 'PY'
import sys
import itertools
import nibabel as nib
import numpy as np

label_f, t2_f, mask_f, out_f, info_f = sys.argv[1:]

label_img = nib.load(label_f)
ref_img = nib.load(t2_f)
mask_img = nib.load(mask_f)

L0 = np.asanyarray(label_img.dataobj)
M = np.asanyarray(mask_img.dataobj) > 0

if M.shape != ref_img.shape[:3]:
    raise RuntimeError(f"T2 mask shape {M.shape} does not match T2 shape {ref_img.shape[:3]}")

best = None
best_data = None

for perm in itertools.permutations([0, 1, 2]):
    Lp = np.transpose(L0, perm)

    if Lp.shape != M.shape:
        continue

    for flips in itertools.product([False, True], repeat=3):
        L = Lp.copy()
        for ax, doflip in enumerate(flips):
            if doflip:
                L = np.flip(L, axis=ax)

        label_mask = L > 0
        inter = int(np.sum(label_mask & M))
        label_vox = int(label_mask.sum())
        outside = int(np.sum(label_mask & ~M))
        coverage = inter / max(int(M.sum()), 1)
        outside_frac = outside / max(label_vox, 1)
        score = coverage - outside_frac

        result = {
            "perm": perm,
            "flips": flips,
            "coverage": coverage,
            "outside": outside,
            "label_voxels": label_vox,
            "score": score,
        }

        if best is None or result["score"] > best["score"]:
            best = result
            best_data = L

if best is None:
    raise RuntimeError("No valid permutation/flip found. Label and T2 mask shapes may be incompatible.")

out = np.rint(best_data).astype(np.uint16)
out_img = nib.Nifti1Image(out, ref_img.affine, ref_img.header)
out_img.set_data_dtype(np.uint16)
nib.save(out_img, out_f)

perm_str = "".join(str(x) for x in best["perm"])
flip_str = "".join("1" if x else "0" for x in best["flips"])

with open(info_f, "w") as f:
    f.write("key,value\n")
    f.write(f"best_perm,{perm_str}\n")
    f.write(f"best_flips,{flip_str}\n")
    f.write(f"T2_coverage,{best['coverage']}\n")
    f.write(f"label_outside_T2,{best['outside']}\n")
    f.write(f"label_voxels,{best['label_voxels']}\n")

print("Saved:", out_f)
print("Best perm:", best["perm"])
print("Best flips:", best["flips"])
print("T2 coverage:", best["coverage"])
print("Label outside T2:", best["outside"])
PY

  BEST_PERM=$(awk -F, '$1=="best_perm"{print $2}' ${ORIENT_INFO})
  BEST_FLIPS=$(awk -F, '$1=="best_flips"{print $2}' ${ORIENT_INFO})
  T2_COV=$(awk -F, '$1=="T2_coverage"{print $2}' ${ORIENT_INFO})

  echo ""
  echo "Step 2: Apply T2-to-b0 transform to fixed label"

  flirt \
    -in ${LABEL_NATIVE_FIXED} \
    -ref ${B0} \
    -applyxfm \
    -init ${MAT} \
    -interp nearestneighbour \
    -out ${LABEL_B0}

  echo ""
  echo "Step 3: Compact ROI labels in b0 space"

  python3 - "${LABEL_B0}" "${LABEL_COMPACT}" "${MAPFILE}" "${B0MASK}" "${COMPACT_INFO}" << 'PY'
import sys
import csv
import nibabel as nib
import numpy as np

infile, outfile, mapfile, maskfile, infofile = sys.argv[1:]

img = nib.load(infile)
data = np.rint(np.asanyarray(img.dataobj)).astype(np.int64)

mask = np.asanyarray(nib.load(maskfile).dataobj) > 0
if mask.shape != data.shape:
    raise RuntimeError(f"B0 mask shape {mask.shape} does not match label shape {data.shape}")

labels = np.unique(data)
labels = labels[labels > 0]

mapping = {old: new for new, old in enumerate(labels, start=1)}

out = np.zeros(data.shape, dtype=np.uint16)
for old, new in mapping.items():
    out[data == old] = new

out_img = nib.Nifti1Image(out, img.affine, img.header)
out_img.set_data_dtype(np.uint16)
nib.save(out_img, outfile)

with open(mapfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["new_label", "old_label"])
    for old, new in mapping.items():
        writer.writerow([new, old])

label_mask = out > 0
inter = int(np.sum(label_mask & mask))
coverage = inter / max(int(mask.sum()), 1)
outside = int(np.sum(label_mask & ~mask))

with open(infofile, "w") as f:
    f.write("key,value\n")
    f.write(f"N_ROI,{len(labels)}\n")
    f.write(f"B0_coverage,{coverage}\n")
    f.write(f"label_outside_b0,{outside}\n")

print("Saved:", outfile)
print("Saved:", mapfile)
print("N compact ROIs:", len(labels))
print("B0 coverage:", coverage)
print("Label outside B0:", outside)
PY

  N_ROI=$(awk -F, '$1=="N_ROI"{print $2}' ${COMPACT_INFO})
  B0_COV=$(awk -F, '$1=="B0_coverage"{print $2}' ${COMPACT_INFO})
  LABEL_OUTSIDE_B0=$(awk -F, '$1=="label_outside_b0"{print $2}' ${COMPACT_INFO})

  echo ""
  echo "Step 4: Count connectome"

  tck2connectome \
    ${TRACKS} \
    ${LABEL_COMPACT} \
    ${COUNT} \
    -symmetric \
    -zero_diagonal \
    -assignment_radial_search 1 \
    -out_assignments ${ASSIGN} \
    -force

  echo ""
  echo "Step 5: Clipped FA connectome"

  fslmaths \
    ${FA} \
    -thr 0 -uthr 1 \
    ${FA_CLIP}

  tcksample \
    ${TRACKS} \
    ${FA_CLIP} \
    ${MEANFA} \
    -stat_tck mean \
    -force

  tck2connectome \
    ${TRACKS} \
    ${LABEL_COMPACT} \
    ${FACONN} \
    -scale_file ${MEANFA} \
    -stat_edge mean \
    -symmetric \
    -zero_diagonal \
    -assignment_radial_search 1 \
    -force

  echo ""
  echo "Step 6: QC summary"

  python3 - "${ASSIGN}" "${COUNT}" "${FACONN}" "${QC_VALUES}" << 'PY'
import sys
import numpy as np

assign_f, count_f, fa_f, out_f = sys.argv[1:]

A = np.loadtxt(assign_f, comments="#", dtype=int)
if A.ndim == 1:
    A = A.reshape(1, -1)

a = A[:, 0]
b = A[:, 1]

total = int(len(A))
both_unassigned = int(np.sum((a == 0) & (b == 0)))
one_unassigned = int(np.sum(((a == 0) & (b != 0)) | ((a != 0) & (b == 0))))
same_node = int(np.sum((a == b) & (a != 0)))
between_node = int(np.sum((a != 0) & (b != 0) & (a != b)))
between_fraction = between_node / max(total, 1)

M = np.loadtxt(count_f, delimiter=",")
if M.ndim == 0:
    M = M.reshape(1, 1)

degree = np.sum(M > 0, axis=0)
isolated_nodes = int(np.sum(degree == 0))
nonzero_edges = int(np.count_nonzero(np.triu(M, k=1)))
total_matrix_sum = float(np.sum(M))

F = np.loadtxt(fa_f, delimiter=",")
if F.ndim == 0:
    F = F.reshape(1, 1)

valid = F[(F > 0) & np.isfinite(F)]
mean_FA = float(np.mean(valid)) if valid.size else 0.0
max_FA = float(np.max(valid)) if valid.size else 0.0

with open(out_f, "w") as f:
    f.write("key,value\n")
    f.write(f"total_streamlines,{total}\n")
    f.write(f"both_unassigned,{both_unassigned}\n")
    f.write(f"one_unassigned,{one_unassigned}\n")
    f.write(f"same_node,{same_node}\n")
    f.write(f"between_node,{between_node}\n")
    f.write(f"between_fraction,{between_fraction}\n")
    f.write(f"isolated_nodes,{isolated_nodes}\n")
    f.write(f"nonzero_edges,{nonzero_edges}\n")
    f.write(f"total_matrix_sum,{total_matrix_sum}\n")
    f.write(f"mean_FA,{mean_FA}\n")
    f.write(f"max_FA,{max_FA}\n")

print("Total streamlines:", total)
print("Between-node streamlines:", between_node)
print("Between-node fraction:", between_fraction)
print("Isolated nodes:", isolated_nodes)
print("Nonzero undirected edges:", nonzero_edges)
print("Mean FA:", mean_FA)
print("Max FA:", max_FA)
PY

  TOTAL=$(awk -F, '$1=="total_streamlines"{print $2}' ${QC_VALUES})
  BOTH_UNASSIGNED=$(awk -F, '$1=="both_unassigned"{print $2}' ${QC_VALUES})
  ONE_UNASSIGNED=$(awk -F, '$1=="one_unassigned"{print $2}' ${QC_VALUES})
  SAME_NODE=$(awk -F, '$1=="same_node"{print $2}' ${QC_VALUES})
  BETWEEN_NODE=$(awk -F, '$1=="between_node"{print $2}' ${QC_VALUES})
  BETWEEN_FRAC=$(awk -F, '$1=="between_fraction"{print $2}' ${QC_VALUES})
  ISOLATED=$(awk -F, '$1=="isolated_nodes"{print $2}' ${QC_VALUES})
  NONZERO_EDGES=$(awk -F, '$1=="nonzero_edges"{print $2}' ${QC_VALUES})
  TOTAL_MATRIX_SUM=$(awk -F, '$1=="total_matrix_sum"{print $2}' ${QC_VALUES})
  MEAN_FA=$(awk -F, '$1=="mean_FA"{print $2}' ${QC_VALUES})
  MAX_FA=$(awk -F, '$1=="max_FA"{print $2}' ${QC_VALUES})

  echo "${ID},${BEST_PERM},${BEST_FLIPS},${T2_COV},${B0_COV},${LABEL_OUTSIDE_B0},${N_ROI},${TOTAL},${BOTH_UNASSIGNED},${ONE_UNASSIGNED},${SAME_NODE},${BETWEEN_NODE},${BETWEEN_FRAC},${ISOLATED},${NONZERO_EDGES},${TOTAL_MATRIX_SUM},${MEAN_FA},${MAX_FA}" >> ${QC}

  echo ""
  echo "Finished ${ID}"
  echo "Count connectome: ${COUNT}"
  echo "FA connectome: ${FACONN}"
)

if [ ! -f "${IDLIST}" ]; then
  echo "ERROR: ID list not found: ${IDLIST}"
  exit 1
fi

while read ID; do
  [ -z "${ID}" ] && continue

  LOG=${LOGDIR}/${ID}.log

  if process_id "${ID}" 2>&1 | tee "${LOG}"; then
    echo "SUCCESS: ${ID}"
  else
    echo "FAILED: ${ID}. Check log: ${LOG}"
  fi

done < ${IDLIST}

echo ""
echo "All requested IDs finished."
echo "QC summary:"
echo "${QC}"
echo "Logs:"
echo "${LOGDIR}"
