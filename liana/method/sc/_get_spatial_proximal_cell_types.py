import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def cell_type_spatial_proximity(
    cell_types,
    coordinates,
    ratio=None,
    interaction_range=250,
    min_cells_in_proximity=10
):
    """
    Compute 1-nearest neighbors between all cell types and
    determine whether cell-type pairs are spatially interacting.

    For each cell of type A, finds its nearest neighbor of type B (for all A–B pairs),
    then flags each pair as 'interacting' if the total number of close A–B pairs
    (within `interaction_range`) is >= `min_cells_in_proximity`.

    Parameters
    ----------
    cell_types : array-like of shape (n_cells,)
        Cell type labels (e.g. adata.obs["cell_type"]).
    coordinates : array-like of shape (n_cells, n_dims)
        Spatial coordinates of each cell (e.g. adata.obsm["spatial"]).
        Units are assumed to be µm unless `ratio` is provided.
    ratio : float, optional
        Conversion factor from pixels to micrometers (µm per pixel).
    interaction_range : float, default=250
        Maximum distance (µm) for considering two cells as interacting.
    min_cells_in_proximity : int, default=10
        Minimum number of close A–B cell pairs required to define
        an interaction between two cell types.

    Returns
    -------
    pairwise_df : pandas.DataFrame
        Columns:
        ["cell_type_1", "cell_index", "cell_type_2",
         "nearest_neighbor_index", "distance_um", "interacting"]
    """
    cell_types = np.asarray(cell_types)
    coordinates = np.asarray(coordinates, dtype=float)
    n_cells = coordinates.shape[0]

    # --- Handle unit conversion ---
    if ratio is not None:
        if ratio <= 0:
            raise ValueError("`ratio` must be positive (µm per pixel).")
        print(f"Converting coordinates from pixels to µm using ratio = {ratio:.3f}")
        coordinates = coordinates * ratio
    else:
        print("Assuming coordinates are already in micrometers (µm).")

    unique_types = np.unique(cell_types)
    results = []

    # --- Compute all pairwise nearest neighbors ---
    for type_a in unique_types:
        idx_a = np.where(cell_types == type_a)[0]
        coords_a = coordinates[idx_a]

        for type_b in unique_types:
            idx_b = np.where(cell_types == type_b)[0]
            coords_b = coordinates[idx_b]

            if len(idx_a) == 0 or len(idx_b) == 0:
                continue

            if type_a == type_b and len(idx_b) == 1:
                continue

            k = 2 if type_a == type_b else 1
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
            nn.fit(coords_b)
            distances, indices = nn.kneighbors(coords_a)

            if type_a == type_b:
                distances = distances[:, 1]
                indices = indices[:, 1]
            else:
                distances = distances[:, 0]
                indices = indices[:, 0]

            nn_indices_global = idx_b[indices]

            df_tmp = pd.DataFrame({
                "cell_type_1": type_a,
                "cell_index": idx_a,
                "cell_type_2": type_b,
                "nearest_neighbor_index": nn_indices_global,
                "distance_um": distances,
            })
            results.append(df_tmp)

    pairwise_df = pd.concat(results, ignore_index=True)

    # --- Summarize interaction counts per type pair ---
    df_filtered = pairwise_df[pairwise_df["distance_um"] <= interaction_range]
    counts = (
        df_filtered.groupby(["cell_type_1", "cell_type_2"])
        .size()
        .reset_index(name="n_interactions")
    )
    counts["interacting"] = (counts["n_interactions"] >= min_cells_in_proximity).astype(int)

    # --- Merge interaction info back to per-cell dataframe ---
    pairwise_df = pairwise_df.merge(
        counts[["cell_type_1", "cell_type_2", "interacting"]],
        on=["cell_type_1", "cell_type_2"],
        how="left"
    )
    pairwise_df["interacting"] = pairwise_df["interacting"].fillna(0).astype(int)

    return pairwise_df