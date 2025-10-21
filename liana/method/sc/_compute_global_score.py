import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
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
    original_groupby_labels: pd.Series,
    lrdata_X: np.ndarray,
    lrdata_var_index: np.ndarray,
    xy_sep: str,
    seed: int,
) -> dict:
    """Worker function to compute global scores for a single permutation."""
    rng = np.random.default_rng(seed)
    
    # 1. Shuffle the labels
    shuffled_labels = rng.permutation(original_groupby_labels.values)

    # 2. Recreate the one-hot normalized matrix (t) from shuffled labels
    celltypes = pd.get_dummies(shuffled_labels)
    
    ct = celltypes.values.astype(float) # (n_cells, k)
    col_sums = ct.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    t = ct / col_sums # Normalized matrix (n_cells, k)
    
    # 3. Aggregated values calculation (t.T @ X)
    values = (t.T @ lrdata_X).ravel()
    
    # 4. Prepare keys for output (k*p total interactions)
    celltype_names = list(celltypes.columns.values)
    interaction_names = lrdata_var_index

    interaction_names_tiled = np.tile(interaction_names, len(celltype_names))
    celltype_names_repeated = np.repeat(celltype_names, interaction_names.shape[0])
    
    # Key format: "source^ligand^receptor" + xy_sep + "target"
    keys = (
        interaction_names_tiled +
        xy_sep +
        celltype_names_repeated
    )
    
    return dict(zip(keys, values))


# =================================================================
# 3. Main Function: Computes score and p-value
# =================================================================

def compute_global_score(
    lrdata: AnnData,
    groupby: str,
    xy_sep: str = V.lr_sep,
    complex_sep: str = "_",
    n_perms: int = 1000,
    seed: int = 42,
    n_jobs: int = -1,
) -> None:
    """
    Computes global score and calculates permutation test p-values.

    Args:
        lrdata (AnnData): The annotated data matrix output from inflow score.
        groupby (str): The grouping column (cell type) in `lrdata.obs`.
        xy_sep (str, optional): Separator for names. Defaults to `V.lr_sep` ('^').
        complex_sep (str, optional): Separator for splitting complex names. Defaults to "_".
        n_perms (int, optional): Number of permutations for p-value calculation. Defaults to 1000.
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
    
    # --- Part A: Compute Observed Score ---

    # One-hot and normalize per cell type (using original labels)
    celltypes = pd.get_dummies(lrdata.obs[groupby])
    t = celltypes.values.astype(float) # (n_cells, k)
    col_sums = t.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    t = t / col_sums # (n_cells, k), Normalized

    # Aggregated values
    X_raw = lrdata.X
    X = X_raw.toarray() if issparse(X_raw) else np.asarray(X_raw)
    values = (t.T @ X).ravel()

    # Names
    celltype_names = list(celltypes.columns.values)
    interaction_names = lrdata.var.index.astype(str).values 

    # Expand to include target
    full_names = (
        np.tile(interaction_names, len(celltype_names)) +
        xy_sep +
        np.repeat(celltype_names, interaction_names.shape[0])
    )

    # Build the observed DataFrame
    parts = [n.split(xy_sep) for n in full_names]
    # We assume the index structure is L-R-Source
    df = pd.DataFrame(parts, columns=["source", "ligand", "receptor", "target"]) 
    df["lr_mean"] = values
    df["pval"] = 0.0 # Initialize pval column

    # Complex parsing
    lig_primary, lig_complex = zip(*df["ligand"].map(lambda x: _split_complex(x, complex_sep)))
    rec_primary, rec_complex = zip(*df["receptor"].map(lambda x: _split_complex(x, complex_sep)))

    df["ligand"] = lig_primary
    df["ligand_complex"] = lig_complex
    df["receptor"] = rec_primary
    df["receptor_complex"] = rec_complex
    
    # Store the observed scores and necessary metadata for p-value calculation
    observed_df = df.copy()
    
    # Map the full key to the observed lr_mean score for quick lookup
    interaction_keys_for_init = full_names # full_names = L-R-Source^Target
    observed_score_map = dict(zip(interaction_keys_for_init, observed_df["lr_mean"].values))

    # --- Part B: Permutation Test  ---

    # Initialize perm_matrix
    perm_matrix = {key: [] for key in interaction_keys_for_init}
    
    # Generate unique seeds
    seeds = rng_main.integers(0, 2**32 - 1, size=n_perms)
    
    print(f"Running {n_perms} permutations...")
    
    # Parallel Execution
    permuted_scores_list = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single_permutation)(
            original_groupby_labels,
            X, # Use pre-computed dense X
            interaction_names, # This is the raw L-R-Source names
            xy_sep,
            s
        ) for s in seeds
    )

    # Aggregation
    for scores_dict in permuted_scores_list:
        for key, score in scores_dict.items():
            perm_matrix[key].append(score)

    # Compute p-values
    pvals = []
    
    for key in interaction_keys_for_init:
        real_score = observed_score_map.get(key)
        perm_scores = np.array(perm_matrix[key])
        
        # P-value calculation with (N+1) smoothing
        pval = (np.sum(perm_scores >= real_score) + 1) / (n_perms + 1)
        pvals.append(pval)
        
    # Assign the calculated p-values back to the observed_df
    observed_df["pval"] = pvals
    
    # Reorder columns (final save)
    observed_df = observed_df[["ligand", "ligand_complex", "receptor", "receptor_complex",
                              "source", "target", "lr_mean", "pval"]]

    # Final Cleanup and Saving
    lrdata.obs[groupby] = original_groupby_labels
    lrdata.uns["global_score"] = observed_df