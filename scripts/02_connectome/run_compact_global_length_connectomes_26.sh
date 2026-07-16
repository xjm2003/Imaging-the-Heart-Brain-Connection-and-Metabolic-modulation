#!/usr/bin/env bash
set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1

OUT=length_connectomes_100k_T2mask
mkdir -p "$OUT"

for d in preproc_B*; do
    [ -d "$d" ] || continue

    ID=${d#preproc_}
    ATLAS="${d}/${ID}_label_T2_ants_fixed_in_b0_compact.nii.gz"

    if [ -f "${d}/${ID}_tracks_100k_T2mask.tck" ]; then
        TCK="${d}/${ID}_tracks_100k_T2mask.tck"
        TAG="100k_T2mask"
    elif [ -f "${d}/${ID}_tracks_100k.tck" ]; then
        TCK="${d}/${ID}_tracks_100k.tck"
        TAG="100k"
    else
        echo "SKIP ${ID}: no 100k tract file"
        continue
    fi

    if [ ! -f "$ATLAS" ]; then
        echo "SKIP ${ID}: no compact b0 atlas"
        continue
    fi

    mkdir -p "${OUT}/${ID}"

    echo "===== ${ID} ====="

    tck2connectome "$TCK" "$ATLAS" "${OUT}/${ID}/${ID}_connectome_mean_length_${TAG}.csv" \
        -scale_length \
        -stat_edge mean \
        -symmetric \
        -zero_diagonal \
        -out_assignments "${OUT}/${ID}/${ID}_assignments_length_${TAG}.csv"

    tck2connectome "$TCK" "$ATLAS" "${OUT}/${ID}/${ID}_connectome_total_length_${TAG}.csv" \
        -scale_length \
        -stat_edge sum \
        -symmetric \
        -zero_diagonal
done
