import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from numpy.linalg import lstsq

df = pd.read_csv("analysis_25/analysis_table_connectome_cardiac_mwm_25.csv")

connectome_cols = [
    "count_total",
    "count_density",
    "max_count_node_strength",
    "sd_count_node_strength",
    "mean_FA_edge",
    "median_FA_edge",
    "max_FA_edge",
    "mean_FA_node_strength",
    "max_FA_node_strength",
    "sd_FA_node_strength",
]

cardiac_cols = [
    "cardiac_Mass",
    "Age_at_cardiac_measurement",
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

connectome_cols = [c for c in connectome_cols if c in df.columns]
cardiac_cols = [c for c in cardiac_cols if c in df.columns]

def residualize_rank(y, cov):
    y = pd.Series(y).rank().to_numpy(dtype=float)
    X = pd.DataFrame(cov).rank().to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = lstsq(X, y, rcond=None)
    return y - X @ beta

rows = []

for xcol in connectome_cols:
    for ycol in cardiac_cols:
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")

        covars = []
        if ycol != "Age_at_cardiac_measurement" and "Age_at_cardiac_measurement" in df.columns:
            covars.append("Age_at_cardiac_measurement")
        if ycol != "cardiac_Mass" and "cardiac_Mass" in df.columns:
            covars.append("cardiac_Mass")

        cov = df[covars].apply(pd.to_numeric, errors="coerce") if covars else pd.DataFrame(index=df.index)

        ok = x.notna() & y.notna()
        for c in covars:
            ok &= cov[c].notna()

        if ok.sum() >= 8 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
            if covars:
                rx = residualize_rank(x[ok], cov.loc[ok, covars])
                ry = residualize_rank(y[ok], cov.loc[ok, covars])
                rho, p = spearmanr(rx, ry)
            else:
                rho, p = spearmanr(x[ok], y[ok])

            rows.append({
                "connectome_metric": xcol,
                "cardiac_metric": ycol,
                "n": int(ok.sum()),
                "adjusted_for": "+".join(covars) if covars else "none",
                "partial_spearman_rho": rho,
                "p_value": p
            })

res = pd.DataFrame(rows).sort_values("p_value")

m = len(res)
if m > 0:
    res["rank"] = range(1, m + 1)
    res["q_value_BH"] = (res["p_value"] * m / res["rank"]).clip(upper=1)
    res["q_value_BH"] = res["q_value_BH"][::-1].cummin()[::-1]

res.to_csv("analysis_25/connectome_cardiac_partial_correlations_25.csv", index=False)

print(res.head(50).to_string(index=False))
