#!/usr/bin/env bash
set -euo pipefail

OUT=length_connectomes_noncompact_100k_T2mask
mkdir -p "$OUT"

for d in preproc_B*; do
    [ -d "$d" ] || continue

    ID=${d#preproc_}
    ATLAS="${d}/${ID}_label_T2_ants_fixed_in_b0.nii.gz"

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
        echo "SKIP ${ID}: no non-compact b0 atlas"
        continue
    fi

    mkdir -p "${OUT}/${ID}"

    echo "===== ${ID} ====="

    tck2connectome "$TCK" "$ATLAS" "${OUT}/${ID}/${ID}_connectome_mean_length_noncompact_${TAG}.csv" \
        -scale_length \
        -stat_edge mean \
        -symmetric \
        -zero_diagonal

done
