import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

df = pd.read_csv("analysis_25/analysis_table_connectome_cardiac_mwm_25.csv")
res = pd.read_csv("analysis_25/connectome_cardiac_correlations_25.csv")

outdir = "figures_connectome_cardiac_25"
os.makedirs(outdir, exist_ok=True)

top = res.head(10)

for _, row in top.iterrows():
    xcol = row["connectome_metric"]
    ycol = row["cardiac_metric"]

    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    ok = x.notna() & y.notna()

    rho, p = spearmanr(x[ok], y[ok])

    plt.figure(figsize=(5.2, 4.2))
    plt.scatter(x[ok], y[ok])

    for _, r in df[ok].iterrows():
        plt.text(
            pd.to_numeric(r[xcol], errors="coerce"),
            pd.to_numeric(r[ycol], errors="coerce"),
            str(r["ID"]),
            fontsize=6
        )

    q = row.get("q_value_BH", None)
    if pd.notna(q):
        title = f"{xcol} vs {ycol}\nrho={rho:.3f}, p={p:.3g}, q={q:.3g}"
    else:
        title = f"{xcol} vs {ycol}\nrho={rho:.3f}, p={p:.3g}"

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()

    safe = f"{xcol}_vs_{ycol}".replace("/", "_").replace(" ", "_")
    plt.savefig(f"{outdir}/{safe}.png", dpi=300)
    plt.close()

print("Saved figures to", outdir)
