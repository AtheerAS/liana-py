import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, diags
from anndata import AnnData
from joblib import Parallel, delayed

# --- Placeholder for li.mt._constants.DefaultValues ---
class DefaultValues:
    lr_sep = '^'
V = DefaultValues()

# =================================================================
# 1. Helper Function: Splits complex names
# =================================================================

def _split_complex(name: str, complex_sep: str = "_"):
    """Helper for splitting complex names."""
    toks = name.split(complex_sep)
    if len(toks) > 1:
        return toks[0], complex_sep.join(toks[1:])
    else:
        return name, name

# =================================================================
# 2. Worker Function: Computes score for a single permutation
# =================================================================

def _run_single_permutation(
    shuffled_labels: np.ndarray,
    lrdata_X,
    interaction_names: np.ndarray,
    celltype_names: list,
    xy_sep: str,
) -> dict:
    """Run a single permutation using precomputed shuffled labels."""
    cats = pd.Categorical(shuffled_labels, categories=celltype_names, ordered=False)
    col_idx = cats.codes

    n_cells = col_idx.size
    n_types = len(celltype_names)
    row_idx = np.arange(n_cells)

    # Build sparse one-hot and normalize columns
    t_sparse = csr_matrix(
        (np.ones(n_cells, dtype=np.float64), (row_idx, col_idx)),
        shape=(n_cells, n_types)
    )
    col_sums = np.asarray(t_sparse.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    t_sparse = t_sparse @ diags(1.0 / col_sums)

    # Ensure X is sparse CSR
    X = lrdata_X if issparse(lrdata_X) else csr_matrix(lrdata_X)

    # Aggregation
    result = t_sparse.T @ X
    values = (result.toarray() if issparse(result) else np.asarray(result)).ravel()

    # Build keys with vectorized string ops
    interaction_names_tiled = np.tile(interaction_names.astype(str), n_types)
    celltype_names_repeated = np.repeat(
        np.asarray(celltype_names, dtype=str),
        interaction_names.shape[0]
    )
    keys = np.char.add(np.char.add(interaction_names_tiled, xy_sep), celltype_names_repeated)

    return dict(zip(keys.tolist(), values.tolist()))

# =================================================================
# 3. Main Function: Computes score and p-value
# =================================================================

def compute_global_score(
    lrdata: AnnData,
    groupby: str,
    xy_sep: str = V.lr_sep,
    complex_sep: str = "_",
    n_perms: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> None:
    """
    Computes global score and calculates permutation test p-values.

    Args:
        lrdata (AnnData): The annotated data matrix output from inflow score.
        groupby (str): The grouping column (cell type) in `lrdata.obs`.
        xy_sep (str, optional): Separator for names. Defaults to `V.lr_sep` ('^').
        complex_sep (str, optional): Separator for splitting complex names. Defaults to "_".
        n_perms (int, optional): Number of permutations for p-value calculation. Defaults to 500.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all processors).

    Returns:
        None: The result with 'lr_mean' and 'pval' is stored in `lrdata.uns["global_score"]`.
    """

    if groupby not in lrdata.obs.columns:
        raise KeyError(
            f"`groupby`='{groupby}' not found in lrdata.obs. "
            "Use the same grouping column used to build lrdata."
        )

    rng_main = np.random.default_rng(seed)
    original_groupby_labels = lrdata.obs[groupby].copy()

    # --- Part A: Observed score (sparse) ---

    # Fixed column order for cell types
    celltypes = pd.get_dummies(lrdata.obs[groupby])
    celltype_names = list(celltypes.columns)

    # Map original labels to fixed indices
    cats_obs = pd.Categorical(lrdata.obs[groupby].values, categories=celltype_names, ordered=False)
    col_idx_obs = cats_obs.codes

    n_cells = col_idx_obs.size
    n_types = len(celltype_names)
    row_idx = np.arange(n_cells)

    # Sparse one-hot and normalize columns
    t_sparse = csr_matrix(
        (np.ones(n_cells, dtype=np.float64), (row_idx, col_idx_obs)),
        shape=(n_cells, n_types)
    )
    col_sums = np.asarray(t_sparse.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    t_sparse = t_sparse @ diags(1.0 / col_sums)

    # Ensure X is sparse CSR
    X_raw = lrdata.X
    X = X_raw if issparse(X_raw) else csr_matrix(X_raw)

    # Aggregation
    result = t_sparse.T @ X
    values = (result.toarray() if issparse(result) else np.asarray(result)).ravel()

    # Names
    interaction_names = lrdata.var.index.astype(str).values

    # Build full names "L^R^Source^Target"
    full_names = np.char.add(
        np.char.add(
            np.tile(interaction_names.astype(str), n_types),
            xy_sep
        ),
        np.repeat(np.asarray(celltype_names, dtype=str), interaction_names.shape[0])
    )

    # Build observed df
    parts = [n.split(xy_sep) for n in full_names.tolist()]
    df = pd.DataFrame(parts, columns=["source", "ligand", "receptor", "target"])
    df["lr_mean"] = values
    df["pval"] = 0.0

    # Complex parsing
    lig_primary, lig_complex = zip(*df["ligand"].map(lambda x: _split_complex(x, complex_sep)))
    rec_primary, rec_complex = zip(*df["receptor"].map(lambda x: _split_complex(x, complex_sep)))
    df["ligand"] = lig_primary
    df["ligand_complex"] = lig_complex
    df["receptor"] = rec_primary
    df["receptor_complex"] = rec_complex

    observed_df = df.copy()

    # Prepare for permutation test
    interaction_keys_for_init = full_names.tolist()
    observed_score_map = dict(zip(interaction_keys_for_init, observed_df["lr_mean"].values))
    perm_matrix = {key: [] for key in interaction_keys_for_init}

    # --- Part B: Permutation test ---

    # Precompute all permutations
    permuted_labels_list = [
        rng_main.permutation(original_groupby_labels.values)
        for _ in range(n_perms)
    ]

    if verbose:
        print(f"Running {n_perms} permutations in parallel...")

    joblib_verbose = 5 if verbose else 0

    # Run in parallel
    permuted_scores_list = Parallel(n_jobs=n_jobs, verbose=joblib_verbose)(
        delayed(_run_single_permutation)(
            shuffled_labels=shuffled_labels,
            lrdata_X=X,
            interaction_names=interaction_names,
            celltype_names=celltype_names,
            xy_sep=xy_sep,
        ) for shuffled_labels in permuted_labels_list
    )

    # Aggregate
    for scores_dict in permuted_scores_list:
        for key, score in scores_dict.items():
            perm_matrix[key].append(score)

    # Compute p-values
    n = float(n_perms + 1)
    pvals = []
    for key in interaction_keys_for_init:
        real_score = observed_score_map.get(key)
        perm_scores = np.asarray(perm_matrix[key], dtype=np.float64)
        pval = (np.sum(perm_scores >= real_score) + 1.0) / n
        pvals.append(pval)

    observed_df["pval"] = pvals
    observed_df = observed_df[[
        "ligand", "ligand_complex", "receptor", "receptor_complex",
        "source", "target", "lr_mean", "pval"
    ]]

    # Save result
    lrdata.uns["global_score"] = observed_df