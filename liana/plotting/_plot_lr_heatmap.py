import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from anndata import AnnData # Assuming 'lrdata' is an anndata object, common in this context

def lr_heatmap(
    ligand: str,
    receptor: str,
    lrdata: AnnData,
    groupby: str,
    sep: str = "^",
    clip: bool = True,
    filter_min_mean: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = "Receiver cell type",
    ylabel: str = "Sender cell type",
    cmap: str = "mako_r",
    figsize: Tuple[float, float] = (10, 8),
    **kwargs: Any
):
    """
    Generates a heatmap of ligand-receptor (LR) interaction strengths.

    The heatmap visualizes the mean LR interaction signal from each 'sender'
    cell type to each 'receiver' cell type, based on the `lrdata` object.

    :param ligand: Name of the ligand (e.g., "COL1A1").
    :param receptor: Name of the receptor (e.g., "DDR1").
    :param lrdata: AnnData object; output from inflow function.
    :param groupby: Key in `lrdata.obs` to use for defining 'receiver' cell types.
    :param sep: Separator used in `.var_names` to delineate sender, ligand, and receptor.
                Defaults to "^".
    :param clip: If True, clips the heatmap values to the [5th, 95th] percentiles
                 to stabilize the color scale. Defaults to True.
    :param filter_min_mean: Minimum mean LR signal required for a sender or receiver
                            cell type to be included in the heatmap. Groups below
                            this threshold are filtered out. Defaults to None (no filtering).
    :param title: Custom title for the plot. If None, a default title is generated.
    :param xlabel: Label for the x-axis (receiver cell types).
    :param ylabel: Label for the y-axis (sender cell types).
    :param cmap: Matplotlib colormap name for the heatmap. Defaults to "mako_r".
    :param figsize: Tuple defining the figure size (width, height) in inches.
    :param kwargs: Additional keyword arguments passed to `seaborn.heatmap`.

    :raises ValueError: If the LR pair is not found, the `groupby` key is missing,
                        or the heatmap is empty after filtering.
    """
    # 1. Input Validation and Data Subsetting
    if not isinstance(lrdata, AnnData):
        raise TypeError("lrdata must be an AnnData object.")
    if groupby not in lrdata.obs.columns:
        raise ValueError(f"'{groupby}' not found in lrdata.obs. Available columns are: {list(lrdata.obs.columns)}")

    pair = f"{ligand}{sep}{receptor}"
    mask = lrdata.var_names.str.endswith(pair)
    if not mask.any():
        raise ValueError(f"❌ No LR features found for: {pair} (e.g., 'Sender{sep}{pair}'). Check ligand/receptor names or 'sep'.")

    # Use a view or copy for safety and performance
    sub = lrdata[:, mask].copy()

    # 2. Robust Parsing of Variable Names
    # Expects format: 'sender^ligand^receptor'
    try:
        parts = sub.var_names.to_series().str.rsplit(sep, n=2, expand=True)
        if parts.shape[1] != 3:
             # Fallback/Error if the split didn't yield 3 parts
            raise ValueError(f"Could not parse var_names using separator '{sep}'. Expected format: 'sender{sep}ligand{sep}receptor'.")

        parts.columns = ["sender", "ligand", "receptor"]
        # Assign the parsed sender labels to `sub.var`
        sub.var["sender"] = parts["sender"].values
    except Exception as e:
        raise ValueError(f"Error during var_names parsing: {e}")

    # 3. Prepare Data for Heatmap
    receiver = sub.obs[groupby].astype(str) # Convert to string for consistent grouping

    # Extract expression data
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else sub.X
    df = pd.DataFrame(X, index=sub.obs_names, columns=sub.var["sender"])
    df["receiver"] = receiver.values

    # Calculate mean inflow (mean signal for each sender→receiver interaction)
    # The result is (sender rows) x (receiver columns)
    heatmap_df = df.groupby("receiver").mean().T

    # 4. Optional Filtering of Low-Mean Groups
    if filter_min_mean is not None:
        try:
            filter_min_mean = float(filter_min_mean)
        except ValueError:
            raise TypeError("filter_min_mean must be a number (float or int).")

        # Keep groups (rows/cols) where the average signal is above the threshold
        keep_rows = heatmap_df.mean(axis=1) >= filter_min_mean
        keep_cols = heatmap_df.mean(axis=0) >= filter_min_mean
        heatmap_df = heatmap_df.loc[keep_rows, keep_cols]

        if heatmap_df.empty:
            raise ValueError("⚠️ After filtering, the heatmap is empty. Try lowering **filter_min_mean**.")

    # 5. Ordering for Visual Clarity
    mat = heatmap_df.copy()

    # Order rows (senders) and columns (receivers) by their total signal sum
    row_order = mat.sum(axis=1).sort_values(ascending=False).index
    col_order = mat.sum(axis=0).sort_values(ascending=False).index
    mat = mat.loc[row_order, col_order]

    # 6. Clipping (Color Scale Stabilization)
    vmin = vmax = None
    if clip:
        # Calculate 5th and 95th percentiles from non-NaN values
        valid_data = mat.values[~np.isnan(mat.values)]
        if valid_data.size > 0:
            vmin = np.nanquantile(valid_data, 0.05)
            vmax = np.nanquantile(valid_data, 0.95)
        else:
            print("Warning: Data is empty or all NaN, skipping clipping.")
            clip = False # Disable clipping if data is invalid

    # 7. Plotting
    plt.figure(figsize=figsize)
    
    # Set default seaborn heatmap aesthetics for a clean look
    default_kwargs = {
        'linewidths': 0.4, 
        'linecolor': "white",
        'square': True,
        'cbar_kws': {'shrink': 0.8} # Shrink the colorbar slightly
    }
    
    # Merge default kwargs with user-provided kwargs, prioritizing user input
    plot_kwargs = {**default_kwargs, **kwargs}

    sns.heatmap(
        mat, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        **plot_kwargs
    )
    
    # 8. Final Touches
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title or f"Sender→Receiver Average Inflow Score for {ligand}-{receptor}", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right') # Improve readability of receiver names
    plt.yticks(rotation=0) # Keep sender names horizontal
    plt.tight_layout()
    plt.show()
    
    return mat