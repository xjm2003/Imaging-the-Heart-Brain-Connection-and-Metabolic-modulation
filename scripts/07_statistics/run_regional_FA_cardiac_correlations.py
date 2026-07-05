import pandas as pd
from scipy.stats import spearmanr

main = pd.read_csv("analysis_25/analysis_table_connectome_cardiac_mwm_25.csv")
fa = pd.read_csv("analysis_25/regional_FA_25.csv")

df = main.merge(fa, on="ID", how="inner")

roi_cols = [c for c in df.columns if c.startswith("ROI_") and c.endswith("_FA")]

cardiac_cols = [
    "cardiac_Diastolic_LA",
    "cardiac_Systolic_LA",
    "cardiac_Ejection_Fraction",
    "cardiac_Cardiac_Output",
    "cardiac_Stroke_Volume",
    "cardiac_Heart_Rate",
    "cardiac_MYO_MASS",
]
cardiac_cols = [c for c in cardiac_cols if c in df.columns]

rows = []

for xcol in roi_cols:
    for ycol in cardiac_cols:
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        ok = x.notna() & y.notna()

        if ok.sum() >= 8 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
            rho, p = spearmanr(x[ok], y[ok])
            rows.append({
                "regional_metric": xcol,
                "cardiac_metric": ycol,
                "n": int(ok.sum()),
                "spearman_rho": rho,
                "p_value": p
            })

res = pd.DataFrame(rows).sort_values("p_value")

m = len(res)
if m > 0:
    res["rank"] = range(1, m + 1)
    res["q_value_BH"] = (res["p_value"] * m / res["rank"]).clip(upper=1)
    res["q_value_BH"] = res["q_value_BH"][::-1].cummin()[::-1]

res.to_csv("analysis_25/regional_FA_cardiac_correlations_25.csv", index=False)

print(res.head(50).to_string(index=False))
