import os
import numpy as np
import pandas as pd

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"
STATUS = os.path.join(ROOT, "cohort_27_file_status_from_sources.csv")
OUTDIR = os.path.join(ROOT, "connectomes_global_25")

os.makedirs(OUTDIR, exist_ok=True)

status = pd.read_csv(STATUS)
ids = status.loc[status["ready_for_connectome_inputs"], "ID"].astype(str).tolist()

print("Ready IDs:", len(ids))
for ID in ids:
    print(" ", ID)

def read_mapping(map_f):
    df = pd.read_csv(map_f)
    df.columns = [c.strip() for c in df.columns]

    lower = {c.lower(): c for c in df.columns}

    if "new_label" in lower and "old_label" in lower:
        new_col = lower["new_label"]
        old_col = lower["old_label"]
    else:
        # fallback: assume first column = new/compact label, second = old atlas label
        new_col = df.columns[0]
        old_col = df.columns[1]

    out = df[[new_col, old_col]].copy()
    out.columns = ["new_label", "old_label"]
    out["new_label"] = out["new_label"].astype(int)
    out["old_label"] = out["old_label"].astype(int)
    return out

all_old = set()
subject_maps = {}

for ID in ids:
    map_f = os.path.join(ROOT, f"preproc_{ID}", f"{ID}_label_fixed_compact_mapping.csv")
    if not os.path.exists(map_f):
        raise FileNotFoundError(f"Missing mapping: {map_f}")

    mp = read_mapping(map_f)
    subject_maps[ID] = mp
    all_old.update(mp["old_label"].tolist())

old_labels = sorted(all_old)
old_to_global = {old: i for i, old in enumerate(old_labels)}

global_map = pd.DataFrame({
    "global_index_1based": np.arange(1, len(old_labels) + 1),
    "old_label": old_labels
})
global_map.to_csv(os.path.join(OUTDIR, "global_roi_mapping.csv"), index=False)

G = len(old_labels)
print("Global ROI number:", G)

for ID in ids:
    pre = os.path.join(ROOT, f"preproc_{ID}")
    mp = subject_maps[ID]

    count_f = os.path.join(pre, f"{ID}_connectome_count_100k_fixed.csv")
    fa_f = os.path.join(pre, f"{ID}_connectome_meanFA_100k_fixed_clipped.csv")

    if not os.path.exists(count_f):
        raise FileNotFoundError(f"Missing count connectome: {count_f}")
    if not os.path.exists(fa_f):
        raise FileNotFoundError(f"Missing FA connectome: {fa_f}")

    C = np.loadtxt(count_f, delimiter=",")
    F = np.loadtxt(fa_f, delimiter=",")

    Cg = np.zeros((G, G), dtype=float)
    Fg = np.zeros((G, G), dtype=float)

    for _, r1 in mp.iterrows():
        i_local = int(r1["new_label"]) - 1
        i_global = old_to_global[int(r1["old_label"])]

        for _, r2 in mp.iterrows():
            j_local = int(r2["new_label"]) - 1
            j_global = old_to_global[int(r2["old_label"])]

            if i_local < C.shape[0] and j_local < C.shape[1]:
                Cg[i_global, j_global] = C[i_local, j_local]
            if i_local < F.shape[0] and j_local < F.shape[1]:
                Fg[i_global, j_global] = F[i_local, j_local]

    np.savetxt(os.path.join(OUTDIR, f"{ID}_count_global.csv"), Cg, delimiter=",", fmt="%.6g")
    np.savetxt(os.path.join(OUTDIR, f"{ID}_meanFA_global.csv"), Fg, delimiter=",", fmt="%.6g")

    print(f"{ID}: saved global matrices {Cg.shape}")

print("\nSaved global connectomes to:", OUTDIR)
