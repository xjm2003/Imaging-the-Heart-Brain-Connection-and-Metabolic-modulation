import os
import numpy as np
import pandas as pd

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"
GDIR = f"{ROOT}/connectomes_global_25"
if not os.path.isdir(GDIR):
    GDIR = f"{ROOT}/connectomes_global"

status = pd.read_csv(f"{ROOT}/cohort_27_file_status_from_sources.csv")
ids = status.loc[status["ready_for_connectome_inputs"], "ID"].astype(str).tolist()

rows = []

for ID in ids:
    cfile = f"{GDIR}/{ID}_count_global.csv"
    ffile = f"{GDIR}/{ID}_meanFA_global.csv"

    row = {"ID": ID}

    row["has_count"] = os.path.exists(cfile)
    row["has_FA"] = os.path.exists(ffile)

    if row["has_count"] and row["has_FA"]:
        C = np.loadtxt(cfile, delimiter=",")
        F = np.loadtxt(ffile, delimiter=",")

        row["C_shape"] = str(C.shape)
        row["F_shape"] = str(F.shape)
        row["C_symmetric"] = np.allclose(C, C.T, equal_nan=True)
        row["F_symmetric"] = np.allclose(F, F.T, equal_nan=True)
        row["C_diag_sum"] = float(np.trace(C))
        row["F_diag_sum"] = float(np.trace(F))
        row["C_nan"] = int(np.isnan(C).sum())
        row["F_nan"] = int(np.isnan(F).sum())
        row["C_total"] = float(np.nansum(C))
        row["C_nonzero_edges"] = int(np.count_nonzero(np.triu(C, k=1)))
        row["F_nonzero_edges"] = int(np.count_nonzero(np.triu(F, k=1)))
        row["F_mean_nonzero"] = float(F[F > 0].mean()) if np.any(F > 0) else np.nan
        row["F_max"] = float(np.nanmax(F))
        row["pass_basic_QC"] = (
            row["C_symmetric"]
            and row["F_symmetric"]
            and row["C_diag_sum"] == 0
            and row["F_diag_sum"] == 0
            and row["C_nan"] == 0
            and row["F_nan"] == 0
            and row["F_max"] <= 1.0
        )
    else:
        row["C_shape"] = ""
        row["F_shape"] = ""
        row["C_symmetric"] = False
        row["F_symmetric"] = False
        row["C_diag_sum"] = np.nan
        row["F_diag_sum"] = np.nan
        row["C_nan"] = np.nan
        row["F_nan"] = np.nan
        row["C_total"] = np.nan
        row["C_nonzero_edges"] = np.nan
        row["F_nonzero_edges"] = np.nan
        row["F_mean_nonzero"] = np.nan
        row["F_max"] = np.nan
        row["pass_basic_QC"] = False

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("analysis_25/connectomes_global_QC_25.csv", index=False)

print(df.to_string(index=False))
print("\nSummary:")
print("Expected ready IDs:", len(ids))
print("Count matrices present:", int(df["has_count"].sum()))
print("FA matrices present:", int(df["has_FA"].sum()))
print("Passed basic QC:", int(df["pass_basic_QC"].sum()))
print("\nFailed / missing:")
print(df[~df["pass_basic_QC"]][["ID", "has_count", "has_FA", "C_shape", "F_shape", "F_max"]].to_string(index=False))
