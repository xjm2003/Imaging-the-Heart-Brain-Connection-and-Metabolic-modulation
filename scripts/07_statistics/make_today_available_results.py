from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

BASE = Path("analysis_ready_regional_FA_volume_20260624")
TABLES = BASE / "tables"
DOCS = BASE / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(TABLES / "GLP1_mouse_metadata_25.csv")
fa = pd.read_csv(TABLES / "GLP1_regional_FA_25.csv")
vol_norm = pd.read_csv(TABLES / "GLP1_regional_volume_normalized_25.csv")
vol_abs = pd.read_csv(TABLES / "GLP1_regional_volume_absolute_mm3_25.csv")

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return q
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q

def choose_age_col(df):
    for c in ["Age_Months", "Age_Imaging", "Age"]:
        if c in df.columns and df[c].notna().sum() > 0:
            return c
    return None

def glp1_filter(df):
    cols = [c for c in ["TreatmentGroup", "Group", "NameGroup"] if c in df.columns]
    if not cols:
        return pd.Series(True, index=df.index)
    text = df[cols].astype(str).agg(" ".join, axis=1)
    mask = text.str.contains("GLP", case=False, na=False)
    return mask if mask.any() else pd.Series(True, index=df.index)

def fit_group_effect(df, outcome, effect_var, covariates):
    d = df[[outcome, effect_var] + covariates].copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna(subset=[outcome, effect_var])
    d[effect_var] = d[effect_var].astype(str)

    levels = sorted(d[effect_var].dropna().unique())
    if len(levels) < 2:
        return []
    ref = levels[0]

    X_parts = [pd.Series(1.0, index=d.index, name="Intercept")]
    terms = ["Intercept"]

    for lev in levels[1:]:
        name = f"{effect_var}:{lev}_vs_{ref}"
        X_parts.append((d[effect_var] == lev).astype(float).rename(name))
        terms.append(name)

    used_covs = []
    for c in covariates:
        if c not in d.columns:
            continue
        s = d[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) < 2:
            continue

        if pd.api.types.is_numeric_dtype(s) or c in ["Weight", "Age_Months", "Age_Imaging", "total_brain_label_volume_mm3"]:
            vals = pd.to_numeric(s, errors="coerce")
            if vals.notna().sum() == 0 or vals.nunique(dropna=True) < 2:
                continue
            X_parts.append(vals.rename(c))
            terms.append(c)
            used_covs.append(c)
        else:
            cats = sorted(s.dropna().astype(str).unique())
            if len(cats) < 2:
                continue
            for cat in cats[1:]:
                name = f"{c}:{cat}_vs_{cats[0]}"
                X_parts.append((s.astype(str) == cat).astype(float).rename(name))
                terms.append(name)
            used_covs.append(c)

    model = pd.concat([d[[outcome, effect_var]], *X_parts], axis=1).dropna()
    y = model[outcome].astype(float).to_numpy()
    X = model[terms].astype(float).to_numpy()

    n, p = X.shape
    if n <= p + 1:
        return []

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df_resid = n - p
    sigma2 = float((resid @ resid) / df_resid)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    counts = model[effect_var].value_counts().to_dict()
    rows = []
    for i, term in enumerate(terms):
        if not term.startswith(effect_var + ":"):
            continue
        lev = term.replace(effect_var + ":", "").split("_vs_")[0]
        tval = beta[i] / se[i] if se[i] > 0 else np.nan
        pval = 2 * stats.t.sf(abs(tval), df_resid) if stats is not None and np.isfinite(tval) else np.nan
        rows.append({
            "outcome": outcome,
            "effect_variable": effect_var,
            "reference_level": ref,
            "comparison_level": lev,
            "n_total_model": n,
            "n_reference": counts.get(ref, 0),
            "n_comparison": counts.get(lev, 0),
            "effect_estimate": beta[i],
            "standard_error": se[i],
            "t_statistic": tval,
            "p_value": pval,
            "model_df_resid": df_resid,
            "covariates_used": ";".join(used_covs) if used_covs else "none"
        })
    return rows

def run_sex_stats(measure_df, cols, out_name, model_label, extra_covariates=None):
    d = meta.merge(measure_df, on="ID", how="inner")
    d = d.loc[glp1_filter(d)].copy()

    age_col = choose_age_col(d)
    covs = []
    if "Genotype" in d.columns:
        covs.append("Genotype")
    if age_col:
        covs.append(age_col)
    if "Weight" in d.columns:
        covs.append("Weight")
    if extra_covariates:
        covs.extend([c for c in extra_covariates if c in d.columns])

    rows = []
    for y in cols:
        rows.extend(fit_group_effect(d, y, "Sex", covs))

    res = pd.DataFrame(rows)
    if len(res):
        res.insert(0, "model", model_label)
        res["FDR_q_value"] = bh_fdr(res["p_value"].values)
    else:
        res = pd.DataFrame({"note": [f"No valid sex-effect model could be fit for {model_label}."]})

    res.to_csv(DOCS / out_name, index=False)

fa_cols = [c for c in fa.columns if re.match(r"ROI_\d+_FA$", c)]
vol_norm_cols = [c for c in vol_norm.columns if re.match(r"ROI_\d+_volume_fraction$", c)]
vol_abs_cols = [c for c in vol_abs.columns if re.match(r"ROI_\d+_volume_mm3$", c)]

run_sex_stats(
    fa,
    fa_cols,
    "sex_effect_regional_FA_GLP1.csv",
    "Regional FA ~ Sex + Genotype + Age + Weight within GLP-1 cohort"
)

run_sex_stats(
    vol_norm,
    vol_norm_cols,
    "sex_effect_regional_volume_normalized_GLP1.csv",
    "Normalized regional volume ~ Sex + Genotype + Age + Weight within GLP-1 cohort"
)

run_sex_stats(
    vol_abs,
    vol_abs_cols,
    "sex_effect_regional_volume_absolute_TBV_adjusted_GLP1.csv",
    "Absolute regional volume ~ Sex + total brain label volume + Genotype + Age + Weight within GLP-1 cohort",
    extra_covariates=["total_brain_label_volume_mm3"]
)

# Summarize available genotype + sex results
summary_lines = []
summary_lines.append("# Results currently available for today")
summary_lines.append("")
summary_lines.append("## 1. Genotype and sex effects in the GLP-1 cohort")
summary_lines.append("")
summary_lines.append("Current GLP-1 regional tables include 25 mice with 332-ROI regional FA and regional volume measurements.")
summary_lines.append("")

for c in ["Genotype", "Sex", "TreatmentGroup", "Group", "NameGroup"]:
    if c in meta.columns:
        summary_lines.append(f"### {c} counts")
        summary_lines.append("")
        summary_lines.append(meta[c].value_counts(dropna=False).to_string())
        summary_lines.append("")

files_to_summarize = [
    ("Genotype effect: regional FA", DOCS / "genotype_comparison_regional_FA_GLP1.csv"),
    ("Genotype effect: normalized regional volume", DOCS / "genotype_comparison_regional_volume_normalized_GLP1.csv"),
    ("Genotype effect: absolute regional volume, TBV-adjusted", DOCS / "genotype_comparison_regional_volume_absolute_TBV_adjusted_GLP1.csv"),
    ("Sex effect: regional FA", DOCS / "sex_effect_regional_FA_GLP1.csv"),
    ("Sex effect: normalized regional volume", DOCS / "sex_effect_regional_volume_normalized_GLP1.csv"),
    ("Sex effect: absolute regional volume, TBV-adjusted", DOCS / "sex_effect_regional_volume_absolute_TBV_adjusted_GLP1.csv"),
]

for title, path in files_to_summarize:
    summary_lines.append(f"### {title}")
    summary_lines.append("")
    if not path.exists():
        summary_lines.append("Not available yet.")
        summary_lines.append("")
        continue

    df = pd.read_csv(path)
    if "p_value" not in df.columns:
        summary_lines.append(df.to_string(index=False))
        summary_lines.append("")
        continue

    df = df.sort_values("p_value")
    n_fdr_005 = int((df["FDR_q_value"] < 0.05).sum()) if "FDR_q_value" in df.columns else 0
    n_fdr_010 = int((df["FDR_q_value"] < 0.10).sum()) if "FDR_q_value" in df.columns else 0

    summary_lines.append(f"Number of tests: {len(df)}")
    summary_lines.append(f"FDR < 0.05: {n_fdr_005}")
    summary_lines.append(f"FDR < 0.10: {n_fdr_010}")
    summary_lines.append("")
    summary_lines.append("Top available results by uncorrected p-value:")
    summary_lines.append("")
    show_cols = [c for c in [
        "outcome", "reference_genotype", "comparison_genotype",
        "reference_level", "comparison_level",
        "n_total_model", "n_reference", "n_comparison",
        "effect_estimate", "standard_error", "p_value", "FDR_q_value",
        "covariates_used"
    ] if c in df.columns]
    summary_lines.append(df[show_cols].head(10).to_string(index=False))
    summary_lines.append("")

summary_lines.append("## 2. GLP-1 treatment effects and interaction analyses in the larger cohort")
summary_lines.append("")
summary_lines.append("Current folder contains the GLP-1 working dataset and regional FA/volume tables for 25 mice.")
summary_lines.append("The larger imaging cohort with treatment/control groups is not yet represented as an analysis-ready regional FA/volume table in this folder.")
summary_lines.append("Therefore, GLP-1 treatment effects and treatment-by-sex or treatment-by-genotype interaction models cannot yet be estimated from the current analysis-ready package.")
summary_lines.append("")
summary_lines.append("Current action item: request or locate the larger-cohort regional volume/FA table with treatment group, genotype, sex, age, weight, and matched mouse IDs. Once available, the planned models are:")
summary_lines.append("")
summary_lines.append("- Regional FA ~ Treatment + Sex + Genotype + Age + Weight")
summary_lines.append("- Regional FA ~ Treatment * Sex + Genotype + Age + Weight")
summary_lines.append("- Regional FA ~ Treatment * Genotype + Sex + Age + Weight")
summary_lines.append("- Normalized regional volume ~ Treatment + Sex + Genotype + Age + Weight")
summary_lines.append("- Absolute regional volume ~ Treatment + total brain volume + Sex + Genotype + Age + Weight")
summary_lines.append("")
summary_lines.append("For now, only the GLP-1-cohort genotype and sex effects are available.")

(DOCS / "today_available_results.md").write_text("\n".join(summary_lines))

email = """Dear Alexandra,

For today, I have added the currently available results for the GLP-1 cohort in the analysis-ready folder:

/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1/analysis_ready_regional_FA_volume_20260624

The new summary file is:

docs/today_available_results.md

I included the currently available genotype-effect and sex-effect analyses within the GLP-1 cohort for regional FA, normalized regional volume, and absolute regional volume. The corresponding result tables are also in the docs folder.

For the larger-cohort GLP-1 treatment effects and interaction analyses, I do not yet see an analysis-ready larger-cohort regional FA/volume table in this folder. Therefore, those treatment and treatment-interaction models cannot yet be estimated from the current package. I noted this in the summary and listed the planned models to run once the larger-cohort table is available.

Best,
Jimin
"""
(DOCS / "email_today_results.txt").write_text(email)

print("Wrote:")
print(DOCS / "sex_effect_regional_FA_GLP1.csv")
print(DOCS / "sex_effect_regional_volume_normalized_GLP1.csv")
print(DOCS / "sex_effect_regional_volume_absolute_TBV_adjusted_GLP1.csv")
print(DOCS / "today_available_results.md")
print(DOCS / "email_today_results.txt")
