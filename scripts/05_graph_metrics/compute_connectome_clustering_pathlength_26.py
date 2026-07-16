import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse.csgraph import shortest_path, connected_components

base = Path("/mnt/newStor/paros/paros_WORK/jimin/MRI_GLP_1")
tables = base / "analysis_ready_regional_FA_volume_20260624_plus_B25122904/tables"

# Use existing 26-mouse feature table to define cohort
brain_file = tables / "GLP1_brain_length_volume_features_26_plus_B25122904.csv"
brain = pd.read_csv(brain_file)
mouse_ids = sorted(brain["mouse_id"].astype(str).tolist())

length_root = base / "length_connectomes_100k_T2mask"

outdir = tables / "graph_metrics_clustering_pathlength"
outdir.mkdir(parents=True, exist_ok=True)

def read_matrix_csv(f):
    return pd.read_csv(f, header=None).values.astype(float)

def clean_matrix(mat):
    mat = np.asarray(mat, dtype=float)
    mat[~np.isfinite(mat)] = 0
    mat[mat < 0] = 0
    np.fill_diagonal(mat, 0)
    return mat

def largest_component_indices(A):
    A_bin = (A > 0).astype(int)
    n_components, labels = connected_components(A_bin, directed=False, return_labels=True)

    if n_components == 0:
        return np.arange(A.shape[0])

    sizes = np.bincount(labels)
    largest = np.argmax(sizes)
    return np.where(labels == largest)[0]

def binary_clustering_and_transitivity(A):
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0)

    k = A.sum(axis=1)
    A3_diag = np.diag(A @ A @ A)

    denom = k * (k - 1)
    ci = np.full(A.shape[0], np.nan)
    valid = denom > 0
    ci[valid] = A3_diag[valid] / denom[valid]

    avg_clustering = np.nanmean(ci)

    # global transitivity = closed triples / connected triples
    transitivity = np.nansum(A3_diag) / np.nansum(denom) if np.nansum(denom) > 0 else np.nan

    return avg_clustering, transitivity

def weighted_clustering_onnela(W):
    """
    Weighted clustering for connection-strength matrices.
    Appropriate for count matrix, not for physical length distance matrix.
    Onnela-style normalized weighted clustering.
    """
    W = clean_matrix(W)
    if W.max() <= 0:
        return np.nan

    A = (W > 0).astype(float)
    k = A.sum(axis=1)

    Wn = W / W.max()
    S = np.power(Wn, 1/3)
    cyc = np.diag(S @ S @ S)

    denom = k * (k - 1)
    ci = np.full(W.shape[0], np.nan)
    valid = denom > 0
    ci[valid] = cyc[valid] / denom[valid]

    return np.nanmean(ci)

def path_metrics_from_distance(D):
    """
    D is a distance matrix with 0 for absent edges and positive distance for present edges.
    Computes metrics on the largest connected component.
    """
    D = clean_matrix(D)
    A = D > 0

    if A.sum() == 0:
        return {
            "n_nodes": D.shape[0],
            "lcc_n_nodes": 0,
            "lcc_fraction": np.nan,
            "characteristic_path_length": np.nan,
            "global_efficiency": np.nan,
        }

    idx = largest_component_indices(A.astype(float))
    Dlcc = D[np.ix_(idx, idx)]

    dist = shortest_path(Dlcc, directed=False, unweighted=False)
    n = dist.shape[0]

    finite = np.isfinite(dist)
    np.fill_diagonal(finite, False)

    if finite.sum() == 0:
        char_path = np.nan
        eff = np.nan
    else:
        dvals = dist[finite]
        char_path = float(np.mean(dvals))
        eff = float(np.mean(1 / dvals))

    return {
        "n_nodes": D.shape[0],
        "lcc_n_nodes": len(idx),
        "lcc_fraction": len(idx) / D.shape[0],
        "characteristic_path_length": char_path,
        "global_efficiency": eff,
    }

def binary_path_metrics(A):
    A = (clean_matrix(A) > 0).astype(float)
    return path_metrics_from_distance(A)

def find_count_matrix(mouse_id):
    d = base / f"preproc_{mouse_id}"

    candidates = [
        d / f"{mouse_id}_connectome_count_100k_fixed.csv",
        d / f"{mouse_id}_connectome_count_100k_T2mask.csv",
        d / f"{mouse_id}_connectome_count_100k.csv",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = sorted(d.glob("*connectome*count*100k*.csv"))
    hits = [h for h in hits if "assignment" not in h.name.lower()]
    return hits[0] if hits else None

def find_length_mean_matrix(mouse_id):
    d = length_root / mouse_id

    hits = sorted(d.glob(f"{mouse_id}_connectome_mean_length_*.csv"))
    return hits[0] if hits else None

rows = []

for mouse_id in mouse_ids:
    row = {"mouse_id": mouse_id}

    # --------------------------
    # Count connectome metrics
    # --------------------------
    count_file = find_count_matrix(mouse_id)
    row["count_graph_file"] = str(count_file) if count_file else ""

    if count_file and count_file.exists():
        C = clean_matrix(read_matrix_csv(count_file))

        # Some count matrices may not be perfectly symmetric
        C = (C + C.T) / 2
        A = (C > 0).astype(float)

        row["count_graph_n_nodes"] = C.shape[0]
        row["count_graph_nonzero_edges"] = int(np.triu(A, 1).sum())
        row["count_graph_density"] = row["count_graph_nonzero_edges"] / (C.shape[0] * (C.shape[0] - 1) / 2)

        row["count_binary_clustering"], row["count_binary_transitivity"] = binary_clustering_and_transitivity(A)
        row["count_weighted_clustering"] = weighted_clustering_onnela(C)

        # Binary path length: each existing edge distance = 1
        bp = binary_path_metrics(A)
        row["count_binary_lcc_n_nodes"] = bp["lcc_n_nodes"]
        row["count_binary_lcc_fraction"] = bp["lcc_fraction"]
        row["count_binary_characteristic_path_length"] = bp["characteristic_path_length"]
        row["count_binary_global_efficiency"] = bp["global_efficiency"]

        # Weighted path length: stronger count = shorter graph distance
        D_inv = np.zeros_like(C)
        mask = C > 0
        D_inv[mask] = 1 / C[mask]

        wp = path_metrics_from_distance(D_inv)
        row["count_invweight_lcc_n_nodes"] = wp["lcc_n_nodes"]
        row["count_invweight_lcc_fraction"] = wp["lcc_fraction"]
        row["count_invweight_characteristic_path_length"] = wp["characteristic_path_length"]
        row["count_invweight_global_efficiency"] = wp["global_efficiency"]

    else:
        print(f"WARNING: no count matrix found for {mouse_id}")

    # --------------------------
    # Length connectome metrics
    # --------------------------
    length_file = find_length_mean_matrix(mouse_id)
    row["length_graph_file"] = str(length_file) if length_file else ""

    if length_file and length_file.exists():
        L = clean_matrix(read_matrix_csv(length_file))
        L = (L + L.T) / 2
        A = (L > 0).astype(float)

        row["length_graph_n_nodes"] = L.shape[0]
        row["length_graph_nonzero_edges"] = int(np.triu(A, 1).sum())
        row["length_graph_density"] = row["length_graph_nonzero_edges"] / (L.shape[0] * (L.shape[0] - 1) / 2)

        row["length_binary_clustering"], row["length_binary_transitivity"] = binary_clustering_and_transitivity(A)

        bp = binary_path_metrics(A)
        row["length_binary_lcc_n_nodes"] = bp["lcc_n_nodes"]
        row["length_binary_lcc_fraction"] = bp["lcc_fraction"]
        row["length_binary_characteristic_path_length"] = bp["characteristic_path_length"]
        row["length_binary_global_efficiency"] = bp["global_efficiency"]

        # Physical path length: edge distance = mean streamline length
        pp = path_metrics_from_distance(L)
        row["length_physical_lcc_n_nodes"] = pp["lcc_n_nodes"]
        row["length_physical_lcc_fraction"] = pp["lcc_fraction"]
        row["length_physical_characteristic_path_length"] = pp["characteristic_path_length"]
        row["length_physical_global_efficiency"] = pp["global_efficiency"]

    else:
        print(f"WARNING: no length mean matrix found for {mouse_id}")

    rows.append(row)

out = pd.DataFrame(rows)

out_file = outdir / "GLP1_connectome_graph_metrics_clustering_pathlength_26.csv"
out.to_csv(out_file, index=False)

print("Saved:", out_file)
print("Shape:", out.shape)

show_cols = [
    "mouse_id",
    "count_graph_n_nodes",
    "count_graph_density",
    "count_binary_clustering",
    "count_binary_characteristic_path_length",
    "count_invweight_characteristic_path_length",
    "length_graph_n_nodes",
    "length_graph_density",
    "length_binary_clustering",
    "length_binary_characteristic_path_length",
    "length_physical_characteristic_path_length",
]

show_cols = [c for c in show_cols if c in out.columns]
print(out[show_cols].to_string(index=False))
