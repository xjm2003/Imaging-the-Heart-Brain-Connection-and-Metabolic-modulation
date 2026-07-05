import os
import re
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from numpy.linalg import lstsq

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"
OUTDIR = "analysis_25/genotype_node_level_clean"
os.makedirs(OUTDIR, exist_ok=True)

MAIN_F = "analysis_25/analysis_table_connectome_cardiac_mwm_25.csv"
COUNT_F = "analysis_25/node_strength_count_25.csv"
FA_F = "analysis_25/node_strength_FA_25.csv"

main = pd.read_csv(MAIN_F)
count = pd.read_csv(COUNT_F)
fa = pd.read_csv(FA_F)

meta_cols = ["ID", "BadeaID", "Genotype", "Sex", "Age_Months", "Weight"]
meta_cols = [c for c in meta_cols if c in main.columns]
meta = main[meta_cols].copy()

# Clean genotype / covariates
meta["Genotype"] = meta["Genotype"].astype(str).str.strip()
meta.loc[meta["Genotype"].isin(["nan", "NaN", "None", "", "NA", "<NA>"]), "Genotype"] = pd.NA

if "Sex" in meta.columns:
    meta["Sex"] = meta["Sex"].astype(str).str.strip()
    meta.loc[meta["Sex"].isin(["nan", "NaN", "None", "", "NA", "<NA>"]), "Sex"] = pd.NA

for c in ["Age_Months", "Weight"]:
    if c in meta.columns:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

# Exclude missing genotype for formal genotype analysis
meta_clean = meta.dropna(subset=["Genotype"]).copy()

print("Formal genotype sample:")
print(meta_clean[["ID", "Genotype", "Sex", "Age_Months", "Weight"]].to_string(index=False))
print("\nGenotype counts:")
print(meta_clean["Genotype"].value_counts(dropna=False))
print("\nGenotype by sex:")
print(pd.crosstab(meta_clean["Genotype"], meta_clean["Sex"]))

meta_clean.to_csv(f"{OUTDIR}/genotype_metadata_clean_23.csv", index=False)

count_df = meta_clean.merge(count, on="ID", how="inner")
fa_df = meta_clean.merge(fa, on="ID", how="inner")

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out

def extract_roi_label(col):
    m = re.match(r"ROI_(.+?)_(count_strength|FA_strength)$", col)
    return m.group(1) if m else col

def residualize(y, cov_df):
    y = pd.Series(y).astype(float)
    X = cov_df.copy()

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = pd.get_dummies(X, drop_first=True)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    ok = y.notna()
    for c in X.columns:
        ok &= X[c].notna()

    out = pd.Series(np.nan, index=y.index, dtype=float)

    if ok.sum() < 8:
        return out

    Xmat = X.loc[ok].to_numpy(dtype=float)
    yvec = y.loc[ok].to_numpy(dtype=float)

    Xmat = np.column_stack([np.ones(Xmat.shape[0]), Xmat])
    beta, *_ = lstsq(Xmat, yvec, rcond=None)

    resid = yvec - Xmat @ beta
    out.loc[ok] = resid
    return out

def kruskal_effect_size(H, n, k):
    # epsilon-squared approximation
    if pd.isna(H) or n <= k:
        return np.nan
    return (H - k + 1) / (n - k)

def run_analysis(df, suffix, prefix):
    node_cols = [c for c in df.columns if c.startswith("ROI_") and c.endswith(suffix)]

    covars = []
    for c in ["Age_Months", "Weight", "Sex"]:
        if c in df.columns:
            covars.append(c)

    unadj_rows = []
    adj_rows = []
    pair_rows = []

    for col in node_cols:
        tmp = df[["ID", "Genotype", col] + covars].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=["Genotype", col])

        groups = []
        genotypes = []

        for g, sub in tmp.groupby("Genotype"):
            vals = sub[col].dropna().astype(float).values
            if len(vals) >= 2:
                groups.append(vals)
                genotypes.append(g)

        if len(groups) >= 2:
            H, p = kruskal(*groups)
        else:
            H, p = np.nan, np.nan

        row = {
            "node_metric": col,
            "ROI_old_label": extract_roi_label(col),
            "analysis": "unadjusted",
            "test": "Kruskal_Wallis",
            "n": int(len(tmp)),
            "n_genotypes": int(len(groups)),
            "genotypes": ";".join(genotypes),
            "H_stat": H,
            "epsilon_squared": kruskal_effect_size(H, len(tmp), len(groups)),
            "p_value": p,
        }

        for g in sorted(tmp["Genotype"].dropna().unique()):
            vals = tmp.loc[tmp["Genotype"] == g, col].dropna().astype(float)
            row[f"n_{g}"] = int(len(vals))
            row[f"mean_{g}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"median_{g}"] = float(vals.median()) if len(vals) else np.nan

        unadj_rows.append(row)

        # Pairwise comparisons
        unique_g = sorted(tmp["Genotype"].dropna().unique())
        for i in range(len(unique_g)):
            for j in range(i + 1, len(unique_g)):
                g1, g2 = unique_g[i], unique_g[j]
                v1 = tmp.loc[tmp["Genotype"] == g1, col].dropna().astype(float)
                v2 = tmp.loc[tmp["Genotype"] == g2, col].dropna().astype(float)

                if len(v1) >= 2 and len(v2) >= 2:
                    U, pp = mannwhitneyu(v1, v2, alternative="two-sided")
                    pair_rows.append({
                        "node_metric": col,
                        "ROI_old_label": extract_roi_label(col),
                        "group1": g1,
                        "group2": g2,
                        "n1": int(len(v1)),
                        "n2": int(len(v2)),
                        "median1": float(v1.median()),
                        "median2": float(v2.median()),
                        "median_diff_group1_minus_group2": float(v1.median() - v2.median()),
                        "mannwhitney_U": float(U),
                        "p_value": float(pp),
                    })

        # Adjusted analysis: residualize node strength on age + weight + sex
        if covars:
            tmp2 = df[["ID", "Genotype", col] + covars].copy()
            tmp2[col] = pd.to_numeric(tmp2[col], errors="coerce")

            tmp2["resid"] = residualize(tmp2[col], tmp2[covars])
            tmp2 = tmp2.dropna(subset=["Genotype", "resid"])

            groups2 = []
            genotypes2 = []

            for g, sub in tmp2.groupby("Genotype"):
                vals = sub["resid"].dropna().astype(float).values
                if len(vals) >= 2:
                    groups2.append(vals)
                    genotypes2.append(g)

            if len(groups2) >= 2:
                H2, p2 = kruskal(*groups2)
            else:
                H2, p2 = np.nan, np.nan

            row2 = {
                "node_metric": col,
                "ROI_old_label": extract_roi_label(col),
                "analysis": "adjusted",
                "test": "Kruskal_Wallis_on_residuals",
                "adjusted_for": "+".join(covars),
                "n": int(len(tmp2)),
                "n_genotypes": int(len(groups2)),
                "genotypes": ";".join(genotypes2),
                "H_stat": H2,
                "epsilon_squared": kruskal_effect_size(H2, len(tmp2), len(groups2)),
                "p_value": p2,
            }

            for g in sorted(tmp2["Genotype"].dropna().unique()):
                vals = tmp2.loc[tmp2["Genotype"] == g, "resid"].dropna().astype(float)
                row2[f"n_{g}"] = int(len(vals))
                row2[f"mean_resid_{g}"] = float(vals.mean()) if len(vals) else np.nan
                row2[f"median_resid_{g}"] = float(vals.median()) if len(vals) else np.nan

            adj_rows.append(row2)

    unadj = pd.DataFrame(unadj_rows).sort_values("p_value")
    adj = pd.DataFrame(adj_rows).sort_values("p_value")
    pair = pd.DataFrame(pair_rows).sort_values("p_value")

    if len(unadj):
        unadj["q_value_BH"] = bh_fdr(unadj["p_value"].fillna(1).values)
    if len(adj):
        adj["q_value_BH"] = bh_fdr(adj["p_value"].fillna(1).values)
    if len(pair):
        pair["q_value_BH"] = bh_fdr(pair["p_value"].fillna(1).values)

    unadj.to_csv(f"{OUTDIR}/{prefix}_node_strength_genotype_unadjusted_clean.csv", index=False)
    adj.to_csv(f"{OUTDIR}/{prefix}_node_strength_genotype_adjusted_clean.csv", index=False)
    pair.to_csv(f"{OUTDIR}/{prefix}_node_strength_genotype_pairwise_clean.csv", index=False)

    print(f"\n=== {prefix} unadjusted top 20 ===")
    print(unadj.head(20).to_string(index=False))

    print(f"\n=== {prefix} adjusted top 20 ===")
    print(adj.head(20).to_string(index=False))

    print(f"\n=== {prefix} FDR < 0.05 unadjusted ===")
    sig = unadj[unadj["q_value_BH"] < 0.05]
    print(sig.to_string(index=False) if len(sig) else "none")

    print(f"\n=== {prefix} FDR < 0.05 adjusted ===")
    sig = adj[adj["q_value_BH"] < 0.05]
    print(sig.to_string(index=False) if len(sig) else "none")

run_analysis(count_df, "_count_strength", "count")
run_analysis(fa_df, "_FA_strength", "FA")

print("\nSaved clean genotype node-level results to:", OUTDIR)
