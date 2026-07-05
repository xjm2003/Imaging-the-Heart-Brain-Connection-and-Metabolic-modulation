import pandas as pd
import numpy as np

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"

status = pd.read_csv(f"{ROOT}/cohort_27_file_status_from_sources.csv")
ids = status.loc[status["ready_for_connectome_inputs"], "ID"].astype(str).tolist()

mp = pd.read_csv(f"{ROOT}/metadata_sources/subject_mapping.csv", dtype=str).fillna("")
meta = pd.read_csv(f"{ROOT}/metadata_sources/metadata_appended.csv", dtype=str).fillna("")

for df in [mp, meta]:
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Ensure known manual mapping
manual_rows = pd.DataFrame([
    {"ARunno": "B26010901", "BadeaID": "250530_21"}
])
mp = mp[mp["ARunno"] != "B26010901"].copy()
mp = pd.concat([mp, manual_rows], ignore_index=True)

# Drop duplicated mapping rows
mp = mp.drop_duplicates("ARunno", keep="first")

# Try to recover mapping from metadata if subject_mapping misses new IDs
meta_mapping_candidates = []
for col in ["ARunno", "Second_ARunno", "SAMBA Brunno"]:
    if col in meta.columns:
        tmp = meta[[col, "BadeaID"]].copy()
        tmp = tmp.rename(columns={col: "ARunno"})
        tmp = tmp[(tmp["ARunno"] != "") & (tmp["BadeaID"] != "")]
        meta_mapping_candidates.append(tmp)

if meta_mapping_candidates:
    meta_mp = pd.concat(meta_mapping_candidates, ignore_index=True)
    meta_mp = meta_mp.drop_duplicates("ARunno", keep="first")
    mp = pd.concat([mp, meta_mp], ignore_index=True)
    mp = mp.drop_duplicates("ARunno", keep="first")

# Deduplicate metadata by BadeaID
meta = meta.drop_duplicates("BadeaID", keep="first")

base = pd.DataFrame({"ID": ids})
base["ARunno"] = base["ID"]

df = base.merge(mp[["ARunno", "BadeaID"]], on="ARunno", how="left")
df = df.merge(meta, on="BadeaID", how="left", suffixes=("", "_meta"))

def infer_group(row):
    text = " ".join([
        str(row.get("Group", "")),
        str(row.get("NameGroup", "")),
        str(row.get("Diet", "")),
    ]).upper()
    if "GLP" in text:
        return "HFD_GLP1"
    if "HFD" in text:
        return "HFD"
    if "CONTROL" in text:
        return "Control"
    return "UNKNOWN"

df["TreatmentGroup"] = df.apply(infer_group, axis=1)

if "MappingNote" not in df.columns:
    df["MappingNote"] = ""
df["MappingNote"] = df["MappingNote"].fillna("")
df.loc[df["ID"].eq("B26010901"), "MappingNote"] = "manual_inferred_B26010901_to_250530_21"

priority = [
    "ID", "ARunno", "BadeaID",
    "TreatmentGroup", "Group", "NameGroup", "Diet",
    "Sex", "Genotype", "Weight",
    "Age_Months", "Age_Imaging", "Age_Years",
    "Cardiac_ID", "Cohort_ID", "MappingNote"
]

cols = [c for c in priority if c in df.columns]
cols += [c for c in df.columns if c not in cols]

out = df[cols].copy()
out.to_csv("analysis_25/metadata_connectome_25_clean.csv", index=False)

print(out[[c for c in priority if c in out.columns]].to_string(index=False))

print("\nN rows:", len(out))
print("Missing BadeaID:")
print(out[out["BadeaID"].fillna("").eq("")]["ID"].to_string(index=False))
print("\nTreatmentGroup counts:")
print(out["TreatmentGroup"].value_counts(dropna=False))
