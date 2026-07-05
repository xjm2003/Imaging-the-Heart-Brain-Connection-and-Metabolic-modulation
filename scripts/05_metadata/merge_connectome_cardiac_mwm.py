import pandas as pd
import re
from collections import Counter

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"

def make_unique(cols):
    counts = Counter()
    out = []
    for c in cols:
        c = str(c).strip()
        counts[c] += 1
        out.append(c if counts[c] == 1 else f"{c}.{counts[c]}")
    return out

def norm_id(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]", "", x)
    return x

meta = pd.read_csv(f"{ROOT}/analysis_25/metadata_connectome_25_clean.csv", dtype=str).fillna("")
feat = pd.read_csv(f"{ROOT}/analysis_25/connectome_subject_summary_25.csv", dtype=str).fillna("")
card = pd.read_csv(f"{ROOT}/metadata_sources/CardiacMetrics_MWMResults_GLP1Mice_merged.csv", dtype=str).fillna("")

for df in [meta, feat, card]:
    df.columns = make_unique(df.columns)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

meta["BadeaID_norm"] = meta["BadeaID"].map(norm_id)
card["Cardiac_ID_norm"] = card["ID"].map(norm_id)

card = card.drop_duplicates("Cardiac_ID_norm", keep="first").copy()

cardiac_vars = [
    "Mass",
    "Age",
    "Diastolic_LV_Volume",
    "Systolic_LV_Volume",
    "Heart_Rate",
    "Stroke_Volume",
    "Ejection_Fraction",
    "Cardiac_Output",
    "Diastolic_RV",
    "Systolic_RV",
    "Diastolic_LA",
    "Systolic_LA",
    "Diastolic_RA",
    "Systolic_RA",
    "Diastolic_Myo",
    "Systolic_Myo",
    "RV_Stroke_Volume",
    "MYO_MASS",
]

mwm_vars = [
    "Day1_Totaldistance",
    "Day2_Totaldistance",
    "Day3_Totaldistance",
    "Day4_Totaldistance",
    "Day5_Totaldistance",
    "Day1_SW",
    "Day2_SW",
    "Day3_SW",
    "Day4_SW",
    "Day5_SW",
    "Distance Probe Day 5",
    "Distance Probe Day 8",
    "SW_Distance_Probe_Day_5",
    "SW_Distance_Probe_Day_8",
]

weight_series_vars = [c for c in card.columns if c.startswith("AnimalWeight")]

keep = ["Cardiac_ID_norm", "ID", "Animal Code"] \
     + [c for c in cardiac_vars if c in card.columns] \
     + [c for c in mwm_vars if c in card.columns] \
     + weight_series_vars

card_sub = card[keep].copy()

rename = {}
for c in card_sub.columns:
    if c == "Cardiac_ID_norm":
        continue
    elif c in ["ID", "Animal Code"]:
        rename[c] = "cardiac_table_" + c.replace(" ", "_")
    elif c == "Age":
        rename[c] = "Age_at_cardiac_measurement"
    elif c in cardiac_vars:
        rename[c] = "cardiac_" + c
    elif c in mwm_vars:
        rename[c] = "mwm_" + c.replace(" ", "_")
    elif c in weight_series_vars:
        rename[c] = "longitudinal_" + c

card_sub = card_sub.rename(columns=rename)

df = meta.merge(feat, on="ID", how="left")
df = df.merge(card_sub, left_on="BadeaID_norm", right_on="Cardiac_ID_norm", how="left")

df.to_csv("analysis_25/analysis_table_connectome_cardiac_mwm_25.csv", index=False)

show_cols = [
    "ID", "BadeaID", "cardiac_table_ID", "cardiac_table_Animal_Code",
    "TreatmentGroup", "Sex", "Genotype", "Weight", "Age_Months",
    "count_density", "mean_FA_edge",
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LA",
    "cardiac_Ejection_Fraction",
    "cardiac_Cardiac_Output",
    "cardiac_Stroke_Volume",
    "cardiac_Heart_Rate",
    "Age_at_cardiac_measurement"
]
show_cols = [c for c in show_cols if c in df.columns]

print(df[show_cols].to_string(index=False))
print("\nN rows:", len(df))
print("Duplicated IDs:", df[df["ID"].duplicated()]["ID"].tolist())

if "cardiac_Ejection_Fraction" in df.columns:
    missing = df[df["cardiac_Ejection_Fraction"].astype(str).str.strip().eq("")]
    print("\nMissing cardiac rows:", len(missing))
    if len(missing):
        print(missing[["ID", "BadeaID"]].to_string(index=False))
