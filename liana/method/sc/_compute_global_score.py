import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from anndata import AnnData
from liana._constants import DefaultValues as V

def compute_global_score(
    lrdata: AnnData,
    groupby: str,
    xy_sep: str = V.lr_sep,
    complex_sep: str = "_",
) -> None:
    
    """
    Computes global score by aggregating ligand-receptor interactions by cell type.

    Args:
        lrdata (AnnData): The annotated data matrix output from inflow score.
        groupby (str): The grouping column (cell type) in `lrdata.obs`, should be the same one used in inflow function.
        xy_sep (str, optional): Separator for names. Defaults to `V.lr_sep`.
        complex_sep (str, optional): Separator for splitting complex names. Defaults to "_".
    Raises:
        KeyError: If the `groupby` column is not the same as the one used in inflow function.

    Returns:
        None: The result is stored in `lrdata.uns["global_score"]`.
    """
        
    if groupby not in lrdata.obs.columns:
        raise KeyError(
            f"`groupby`='{groupby}' not found in lrdata.obs. "
            "Use the same grouping column used to build lrdata."
        )

    # One-hot and normalize per cell type
    celltypes = pd.get_dummies(lrdata.obs[groupby])
    ct = csr_matrix(celltypes.astype(int).values)  # (n_cells, k)
    t = ct.toarray().astype(float)                 # (n_cells, k)
    col_sums = t.sum(axis=0, keepdims=True)        # (1, k)
    col_sums[col_sums == 0] = 1.0
    t = t / col_sums                               # (n_cells, k)

    # Aggregated values per target cell type
    X_raw = lrdata.X
    if hasattr(X_raw, "toarray"):                
        X = X_raw.toarray()
    else:
        X = np.asarray(X_raw)
    values = (t.T @ X)                             # (k, p)
    values = values.ravel()

    # Names
    celltype_names = list(celltypes.columns.values)   # targets (k)
    interaction_names = lrdata.var.index.astype(str).values  # "source^ligand^receptor" (p)

    # Expand to include target
    full_names = (
        np.tile(interaction_names, len(celltype_names)) +
        xy_sep +
        np.repeat(celltype_names, interaction_names.shape[0])
    )

    # Split "source^ligand^receptor^target"
    parts = [n.split(xy_sep) for n in full_names]
    df = pd.DataFrame(parts, columns=["source", "ligand", "receptor", "target"])
    df["lr_mean"] = values

    # Complex parsing helpers
    def _split_complex(name: str):
        toks = name.split(complex_sep)
        if len(toks) > 1:
            return toks[0], complex_sep.join(toks[1:])
        else:
            return name, name

    lig_primary, lig_complex = zip(*df["ligand"].map(_split_complex))
    rec_primary, rec_complex = zip(*df["receptor"].map(_split_complex))

    df["ligand"] = lig_primary
    df["ligand_complex"] = lig_complex
    df["receptor"] = rec_primary
    df["receptor_complex"] = rec_complex

    # Reorder columns
    df = df[["ligand", "ligand_complex", "receptor", "receptor_complex",
             "source", "target", "lr_mean"]]

    # Save the result into adata.uns["global_score"]
    lrdata.uns["global_score"] = df