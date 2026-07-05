import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("analysis_table_connectome_cardiac_mwm_17.csv")

connectome_cols = [
    "count_total",
    "count_density",
    "max_count_node_strength",
    "mean_FA_edge",
    "max_FA_edge",
    "mean_FA_node_strength",
    "max_FA_node_strength"
]

cardiac_cols = [
    "cardiac_Mass",
    "cardiac_Age",
    "cardiac_Diastolic_LV_Volume",
    "cardiac_Systolic_LV_Volume",
    "cardiac_Heart_Rate",
    "cardiac_Stroke_Volume",
    "cardiac_Ejection_Fraction",
    "cardiac_Cardiac_Output",
    "cardiac_Diastolic_RV",
    "cardiac_Systolic_RV",
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LA",
    "cardiac_Diastolic_RA",
    "cardiac_Systolic_RA",
    "cardiac_Diastolic_Myo",
    "cardiac_Systolic_Myo",
    "cardiac_RV_Stroke_Volume",
    "cardiac_MYO_MASS",
]

cardiac_cols = [c for c in cardiac_cols if c in df.columns]

rows = []
for xcol in connectome_cols:
    for ycol in cardiac_cols:
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        ok = x.notna() & y.notna()

        if ok.sum() >= 8 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
            rho, p = spearmanr(x[ok], y[ok])
            rows.append({
                "connectome_metric": xcol,
                "cardiac_metric": ycol,
                "n": int(ok.sum()),
                "spearman_rho": rho,
                "p_value": p
            })

res = pd.DataFrame(rows).sort_values("p_value")

# Benjamini-Hochberg FDR
m = len(res)
if m > 0:
    res["rank"] = range(1, m + 1)
    res["q_value_BH"] = (res["p_value"] * m / res["rank"]).clip(upper=1)
    res["q_value_BH"] = res["q_value_BH"][::-1].cummin()[::-1]

res.to_csv("connectome_cardiac_correlations_17.csv", index=False)

print(res.head(40).to_string(index=False))
