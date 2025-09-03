import re
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import scanpy as sc
import muon as mu



def _sanitize_prefix(raw: str) -> str:
    """
    Turn any non-alphanumeric character into '_', collapse repeats, and trim edges.Necessary for MOFA-Flex to accept view name.
    """
    s = re.sub(r"\W+", "_", raw, flags=re.UNICODE)  # non-word -> _
    s = re.sub(r"_+", "_", s).strip("_")
    # Avoid empty string (e.g., if raw was only punctuation)
    return s if s else "unnamed"


def _dedupe(names: pd.Index) -> Dict[str, str]:
    """
    If multiple raw prefixes sanitize to the same value, add numeric suffixes to ensure uniqueness.
    Returns a mapping {raw_prefix: unique_sanitized_prefix}.
    """
    taken = {}
    out = {}
    for raw in names:
        base = _sanitize_prefix(raw)
        candidate = base
        k = 2
        while candidate in taken and taken[candidate] != raw:
            candidate = f"{base}_{k}"
            k += 1
        taken[candidate] = raw
        out[raw] = candidate
    return out



def lrdata_to_mudata(
    lrdata_path: Path | str,
    output_path: Path | str,
    output_name: str = "mudata.h5mu",
    *,
    filter_views: bool = True,
    min_genes: int = 5,
    min_cells: int = 5,
    strip_prefix: bool = True,
    verbose: bool = True,
) -> mu.MuData:
    """
    Convert an AnnData (lrdata) into a MuData with recipient cell type name as view

    Steps:
      1) Load h5ad
      2) Extract prefixes (text before the first '^') and sanitize them (symbols -> '_')
      3) Build per-prefix views (cell type)
      4) Assemble MuData
      5) Optionally filter each modality and print before/after stats
      6) Save .h5mu to the specified output path

    Parameters
    ----------
    lrdata_path : Path | str
        Path to the input .h5ad file (e.g., ".../lrdata.h5ad").
    output_path : Path | str
        Directory to write the resulting .h5mu file.
    output_name : str
        Filename for the .h5mu (default: "mudata.h5mu").
    filter_views : bool
        If True, run sc.pp.filter_cells(min_genes) and sc.pp.filter_genes(min_cells) per modality.
    min_genes : int
        Minimum genes per cell when filtering.
    min_cells : int
        Minimum cells per gene when filtering.
    strip_prefix : bool
        If True, remove the "<raw_prefix>^" from each view’s var_names.
    verbose : bool
        If True, print progress and summary stats.

    Returns
    -------
    mu.MuData
        The assembled MuData object (also saved to disk).
    """
    lrdata_path = Path(lrdata_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading lrdata from: {lrdata_path}")
    lrdata = sc.read_h5ad(lrdata_path)

    # 2) Extract raw prefixes found before the first '^'
    var_names = pd.Index(lrdata.var_names)
    extracted = var_names.to_series().str.extract(r"^(.*?)\^")[0]
    raw_prefixes = extracted.dropna().unique()

    if verbose:
        total_vars = len(var_names)
        matched_vars = extracted.notna().sum()
        print(f"Total variables: {total_vars}")
        print(f"Variables with a prefix pattern '<prefix>^': {matched_vars}")
        if matched_vars < total_vars:
            print(f"⚠️  {total_vars - matched_vars} variables have no '^' prefix and will be ignored.")

    if len(raw_prefixes) == 0:
        raise ValueError("No prefixes found (no variables matched the pattern '^(.*?)\\^').")

    # Sanitize + de-duplicate prefix names
    raw_to_clean = _dedupe(pd.Index(raw_prefixes))

    # 3) Build views
    views = {}
    for raw_prefix in raw_prefixes:
        clean_prefix = raw_to_clean[raw_prefix]

        # variables that start with "<raw_prefix>^"
        matching_vars = var_names[var_names.str.startswith(f"{raw_prefix}^")]
        if len(matching_vars) == 0:
            if verbose:
                print(f"Skipping prefix '{raw_prefix}' (clean: '{clean_prefix}') — no matching variables.")
            continue

        view_data = lrdata[:, matching_vars].copy()

        if strip_prefix:
            # Remove ONLY the leading "<raw_prefix>^"
            pattern = rf"^{re.escape(raw_prefix)}\^"
            view_data.var_names = view_data.var_names.to_series().str.replace(pattern, "", regex=True).values

        views[clean_prefix] = view_data

        if verbose:
            print(f"Built view '{clean_prefix}' from raw prefix '{raw_prefix}': "
                  f"{view_data.n_obs} cells × {view_data.n_vars} features")

    if len(views) == 0:
        raise ValueError("All detected prefixes produced empty views; nothing to assemble.")

    # 4) Create MuData
    mdata = mu.MuData(views)

    # 5) Optional filtering with before/after stats
    if filter_views:
        for mod, adata in mdata.mod.items():
            n_cells_before, n_vars_before = adata.n_obs, adata.n_vars
            if verbose:
                print(f"\nProcessing modality: {mod}")
                print(f"Before filtering: {n_cells_before} cells × {n_vars_before} genes")

            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)

            if verbose:
                print(f"After filtering:  {adata.n_obs} cells × {adata.n_vars} genes")
                print(f"Removed cells:    {n_cells_before - adata.n_obs}")
                print(f"Removed Interactions:    {n_vars_before - adata.n_vars}")

    # 6) Save
    out_file = output_path / output_name
    if verbose:
        print(f"\nWriting MuData to: {out_file}")
    mdata.write(out_file)

    return mdata