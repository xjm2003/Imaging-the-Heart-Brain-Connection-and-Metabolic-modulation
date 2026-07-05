import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1"

# Prefer 25-mouse global directory if it exists
if os.path.isdir(os.path.join(ROOT, "connectomes_global_25")):
    BASE = os.path.join(ROOT, "connectomes_global_25")
else:
    BASE = os.path.join(ROOT, "connectomes_global")

outdir = os.path.join(ROOT, "connectome_figures")
os.makedirs(outdir, exist_ok=True)

ids = [
    "B25122902","B25122903","B25123000","B25123002","B25123003",
    "B25123005","B25123008","B25123009",
    "B26010600","B26010602","B26010604","B26010605","B26010606",
    "B26010607","B26010608","B26010800",
    "B26010801","B26010802","B26010803","B26010804",
    "B26010807","B26010808","B26010900","B26010901","B26010902"
]

count_list = []
fa_list = []
missing_count = []
missing_fa = []

for ID in ids:
    count_candidates = [
        os.path.join(BASE, f"{ID}_count_global.csv"),
        os.path.join(BASE, ID, f"{ID}_connectome_count_global_332.csv"),
        os.path.join(BASE, ID, f"{ID}_count_global.csv"),
    ]

    fa_candidates = [
        os.path.join(BASE, f"{ID}_meanFA_global.csv"),
        os.path.join(BASE, ID, f"{ID}_connectome_meanFA_global_332.csv"),
        os.path.join(BASE, ID, f"{ID}_meanFA_global.csv"),
    ]

    count_file = next((f for f in count_candidates if os.path.exists(f)), None)
    fa_file = next((f for f in fa_candidates if os.path.exists(f)), None)

    if count_file is not None:
        count_list.append(np.loadtxt(count_file, delimiter=","))
    else:
        missing_count.append(ID)

    if fa_file is not None:
        fa_list.append(np.loadtxt(fa_file, delimiter=","))
    else:
        missing_fa.append(ID)

print("Using connectome directory:", BASE)
print("Loaded count matrices:", len(count_list))
print("Loaded FA matrices:", len(fa_list))

if missing_count:
    print("Missing count matrices:", missing_count)

if missing_fa:
    print("Missing FA matrices:", missing_fa)

if len(count_list) == 0:
    raise RuntimeError("No count matrices were found.")

if len(fa_list) == 0:
    raise RuntimeError("No FA matrices were found.")

Cmean = np.mean(count_list, axis=0)
Fmean = np.mean(fa_list, axis=0)

def plot_matrix(M, title, outfile, transform=None, cmap="viridis", vmax_percentile=None):
    X = M.copy()

    if transform == "log1p":
        X = np.log1p(X)

    vmin = None
    vmax = None
    if vmax_percentile is not None:
        vals = X[np.isfinite(X)]
        vals = vals[vals > 0]
        if vals.size > 0:
            vmax = np.percentile(vals, vmax_percentile)
            vmin = 0

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        X,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    plt.title(title)
    plt.xlabel("ROI index")
    plt.ylabel("ROI index")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

plot_matrix(
    Cmean,
    f"Mean count connectome across {len(count_list)} mice (log1p)",
    os.path.join(outdir, f"mean_count_connectome_{len(count_list)}_log1p.png"),
    transform="log1p",
    cmap="hot"
)

plot_matrix(
    Cmean,
    f"Mean count connectome across {len(count_list)} mice (log1p, 99th pct)",
    os.path.join(outdir, f"mean_count_connectome_{len(count_list)}_log1p_p99.png"),
    transform="log1p",
    cmap="hot",
    vmax_percentile=99
)

plot_matrix(
    Fmean,
    f"Mean FA connectome across {len(fa_list)} mice",
    os.path.join(outdir, f"mean_FA_connectome_{len(fa_list)}.png"),
    transform=None,
    cmap="viridis"
)

plot_matrix(
    Fmean,
    f"Mean FA connectome across {len(fa_list)} mice (99th pct)",
    os.path.join(outdir, f"mean_FA_connectome_{len(fa_list)}_p99.png"),
    transform=None,
    cmap="viridis",
    vmax_percentile=99
)

print("Saved group mean figures to:", outdir)
