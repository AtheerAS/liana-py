import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import warnings
from typing import List, Tuple, Optional, Union

# Note: AnnData type hints are omitted here for simplicity but would typically use 'anndata.AnnData'

def plot_spatial_inflow_score_single_interaction(
    adata,
    spatial_key: str,
    interaction: str,
    obs_key: str,
    labels: List[str],
    figsize: Tuple[int, int] = (10, 8),
    point_size: int = 3,
    background_alpha: float = 0.2,
    scatter_alpha: float = 0.8,
    cmaps: Optional[List[str]] = None,
    colorbar_layout: Optional[str] = 'side',  # 'side', 'bottom', or None
    max_cbars_per_col: int = 3,  # wrap after this many colorbars per column (side layout)
    auto_bottom_threshold: int = 6,  # if len(labels) >= this, default to bottom layout
    normalize: bool = True,  # Normalize score to 0-1 range
    percentile_scaling: Optional[Tuple[int, int]] = None,  # Tuple (low, high) e.g., (5, 95) to clip outliers
    vmin: Optional[float] = None,  # Minimum value for color scale (overrides normalize)
    vmax: Optional[float] = None,  # Maximum value for color scale (overrides normalize)
    shared_colorscale: bool = False,  # Use same scale for all cell types
    min_expression: Optional[float] = None,  # Filter cells below this threshold
    show_counts: bool = True,  # Show cell counts in colorbar labels
    title: Optional[str] = 'auto',  # 'auto' for auto-generate, None to hide, or custom string
    bbox: Optional[Tuple[float, float, float, float]] = None,  # (xmin, xmax, ymin, ymax) to zoom
    save_path: Optional[str] = None,  # Path to save figure
    dpi: int = 300  # DPI for saved figure
):
    """
    Plot spatial inflow score of a interaction infloe score across multiple cell types.
    
    Parameters
    ----------
    adata : AnnData
        lrdata object, the output of inflow function.
    spatial_key : str
        Key in adata.obsm containing spatial coordinates
    interaction : str
        interaction name to visualize
    obs_key : str
        Key in adata.obs containing cell type annotations
    labels : list of str
        List of cell type labels to compare
    figsize : tuple, optional
        Figure size (width, height). Default: (10, 8)
    point_size : int, optional
        Size of scatter points. Default: 3
    background_alpha : float, optional
        Alpha transparency for background cells. Default: 0.2
    scatter_alpha : float, optional
        Alpha transparency for cell type-specific points. Default: 0.8
    cmaps : list of str, optional
        List of colormap names (one per cell type). If None, uses defaults.
    colorbar_layout : str or None, optional
        Layout for colorbars: 'side' (vertical stack on right), 'bottom' (horizontal), 
        or None (no colorbars). Default: 'side'
    normalize : bool, optional
        If True, normalize score values to 0-1 range for each cell type.
        Makes different cell types directly comparable. Default: True
    percentile_scaling : tuple or None, optional
        (low_percentile, high_percentile) to clip outliers before scaling.
        E.g., (5, 95) clips to 5th-95th percentile. Applied before normalization.
        Default: None (no clipping)
    vmin : float, optional
        Minimum value for color scale. Overrides normalize. Default: None
    vmax : float, optional
        Maximum value for color scale. Overrides normalize. Default: None
    shared_colorscale : bool, optional
        If True, all cell types use the same color scale range. Default: False
    min_expression : float, optional
        Only show cells with inflow score >= this threshold. Applied to raw values
        before normalization. Default: None (show all cells)
    show_counts : bool, optional
        Show number of cells in colorbar labels. Default: True
    title : str or None, optional
        - 'auto': Auto-generate title from interaction name and cell types
        - None: No title
        - str: Custom title text
        Default: 'auto'
    bbox : tuple, optional
        (xmin, xmax, ymin, ymax) to zoom into specific spatial region. Default: None
    save_path : str, optional
        Path to save the figure (e.g., 'figure.png', 'figure.pdf'). Default: None
    dpi : int, optional
        Resolution for saved figure. Default: 300
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    """
    
    # Validate inputs
    if len(labels) == 0:
        raise ValueError("labels must contain at least one cell type name")
    
    if colorbar_layout not in ['side', 'bottom', None]:
        raise ValueError("colorbar_layout must be 'side', 'bottom', or None")
    
    # Default colormaps if not provided
    if cmaps is None:
        default_cmaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 
                         'YlOrBr', 'PuRd', 'BuGn', 'GnBu', 'OrRd']
        cmaps = default_cmaps[:len(labels)]
        if len(labels) > len(default_cmaps):
            cmaps = [default_cmaps[i % len(default_cmaps)] for i in range(len(labels))]
    elif len(cmaps) != len(labels):
        raise ValueError(f"cmaps must contain {len(labels)} colormap names (one per cell type)")
    
    # --- DATA PREPARATION ---
    
    # Extract spatial coordinates for all cells
    coords = adata.obsm[spatial_key]
    
    # Apply bounding box filter if specified (for background plotting)
    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        bbox_mask = (
            (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
            (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        )
        coords_background = coords[bbox_mask, :]
    else:
        coords_background = coords
    
    # Prepare data for each cell type
    cell_type_data = []
    all_expressions_raw = []  # For shared colorscale (before normalization)
    
    for label in labels:
        mask = adata.obs[obs_key].eq(label)
        
        if not mask.any():
            warnings.warn(f"No cells found for cell type: {label}")
            continue
        
        coords_type = coords[mask.values, :]
        
        # Get inflow score values for the interaction, preferring raw data if available
        if adata.raw is not None and interaction in adata.raw.var_names:
            expr_type = adata.raw[mask][:, interaction].X
        else:
            # Check if interaction is in main adata.var_names
            if interaction not in adata.var_names:
                warnings.warn(f"Interaction '{interaction}' not found in adata.raw or adata.X. Skipping cell type {label}.")
                continue
            expr_type = adata[mask][:, interaction].X
        
        # Convert sparse matrix data to 1D arrays if needed
        expr_type = expr_type.toarray().ravel() if not isinstance(expr_type, np.ndarray) else expr_type
        
        # Apply bounding box filter
        if bbox is not None:
            # Re-apply or ensure filtering is done on the subset
            bbox_mask_type = (
                (coords_type[:, 0] >= xmin) & (coords_type[:, 0] <= xmax) &
                (coords_type[:, 1] >= ymin) & (coords_type[:, 1] <= ymax)
            )
            coords_type = coords_type[bbox_mask_type, :]
            expr_type = expr_type[bbox_mask_type]
        
        # Apply expression threshold filter (before normalization)
        if min_expression is not None:
            expr_mask = expr_type >= min_expression
            coords_type = coords_type[expr_mask, :]
            expr_type = expr_type[expr_mask]
        
        if len(expr_type) == 0:
            warnings.warn(f"No cells for {label} after filtering")
            continue
        
        # Store raw scores for shared scaling
        all_expressions_raw.extend(expr_type)
        
        cell_type_data.append({
            'label': label,
            'coords': coords_type,
            'expression_raw': expr_type.copy(),
            'count': len(expr_type)
        })
    
    if len(cell_type_data) == 0:
        raise ValueError("No cells found after filtering for any specified label.")
    
    # --- SCALING AND NORMALIZATION ---
    
    n_cbars = len(cell_type_data)

    # Auto-switch to bottom layout for many labels
    if colorbar_layout == 'side' and n_cbars >= auto_bottom_threshold:
        warnings.warn(
            f"{n_cbars} colorbars requested; switching layout to 'bottom' "
            f"(threshold={auto_bottom_threshold}). Set auto_bottom_threshold higher to keep side layout."
        )
        colorbar_layout = 'bottom'

    # Determine scaling limits
    if shared_colorscale:
        # Use all expressions to determine shared limits
        all_expressions_raw_arr = np.array(all_expressions_raw)
        
        if percentile_scaling is not None:
            low_p, high_p = percentile_scaling
            scale_min = np.percentile(all_expressions_raw_arr, low_p)
            scale_max = np.percentile(all_expressions_raw_arr, high_p)
        else:
            scale_min = np.min(all_expressions_raw_arr)
            scale_max = np.max(all_expressions_raw_arr)
        
        # Apply scaling to each cell type
        for data in cell_type_data:
            expr = data['expression_raw'].copy()
            
            # Clip to percentile range if specified
            if percentile_scaling is not None:
                expr = np.clip(expr, scale_min, scale_max)
            
            # Normalize to 0-1 if requested (and vmin/vmax not specified)
            if normalize and vmin is None and vmax is None:
                if scale_max - scale_min > 0:
                    expr = (expr - scale_min) / (scale_max - scale_min)
                else:
                    expr = np.zeros_like(expr)
            
            data['expression'] = expr
        
        # Set color limits
        if vmin is not None or vmax is not None:
            color_limits = {}
            if vmin is not None:
                color_limits['vmin'] = vmin
            if vmax is not None:
                color_limits['vmax'] = vmax
        elif normalize:
            color_limits = {'vmin': 0, 'vmax': 1}
        else:
            # If not normalized and shared, the raw scale min/max become the vmin/vmax
            color_limits = {'vmin': scale_min, 'vmax': scale_max}
    
    else:
        # Individual scaling per cell type
        color_limits = {} # Only set if vmin/vmax are manually provided
        
        for data in cell_type_data:
            expr = data['expression_raw'].copy()
            
            # Apply percentile clipping
            if percentile_scaling is not None:
                low_p, high_p = percentile_scaling
                scale_min = np.percentile(expr, low_p)
                scale_max = np.percentile(expr, high_p)
                expr = np.clip(expr, scale_min, scale_max)
            else:
                scale_min = np.min(expr)
                scale_max = np.max(expr)
            
            # Normalize to 0-1 if requested (and vmin/vmax not specified)
            if normalize and vmin is None and vmax is None:
                if scale_max - scale_min > 0:
                    expr = (expr - scale_min) / (scale_max - scale_min)
                else:
                    expr = np.zeros_like(expr)
            
            data['expression'] = expr
        
        # Set color limits (only if vmin/vmax are globally provided or normalization is used)
        if vmin is not None or vmax is not None:
            if vmin is not None:
                color_limits['vmin'] = vmin
            if vmax is not None:
                color_limits['vmax'] = vmax
        elif normalize:
            # If normalized, the individual data is already 0-1, so set limits for plotting
            color_limits = {'vmin': 0, 'vmax': 1}
        # If not normalized and not shared, each scatter uses its own automatic vmin/vmax

    # --- VISUALIZATION ---
    
    if colorbar_layout == 'bottom':
        # Leave bottom strip for horizontal colorbars
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.24, 0.8, 0.72])  # [left, bottom, width, height]

    elif colorbar_layout == 'side':
        # Build a right-hand panel using GridSpec and wrap colorbars over columns

        n_cbars_effective = len(cell_type_data)
        
        # Calculate rows and columns for colorbar grid
        if n_cbars_effective <= max_cbars_per_col:
            # Prefer ~square layout for small n
            cbar_cols = int(np.ceil(np.sqrt(n_cbars_effective)))
            cbar_rows = int(np.ceil(n_cbars_effective / cbar_cols))
        else:
            # For large n, wrap after max_cbars_per_col
            cbar_cols = int(np.ceil(n_cbars_effective / max_cbars_per_col))
            cbar_rows = max_cbars_per_col # The number of rows is fixed by max_cbars_per_col
            # Re-calculating cbar_cols based on the fixed cbar_rows
            cbar_cols = int(np.ceil(n_cbars_effective / cbar_rows))

        # Build figure with constrained layout to avoid label clipping
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        gs = GridSpec(
            nrows=max(cbar_rows, 1),
            ncols=1 + cbar_cols,            # 1 big column for main plot + cbar columns
            width_ratios=[30] + [1]*cbar_cols,  # 30:1 ratio for main plot vs cbar
            figure=fig
        )
        ax = fig.add_subplot(gs[:, 0])
        side_cbar_config = dict(gs=gs, cbar_cols=cbar_cols, cbar_rows=cbar_rows)
    else:
        # No colorbars (or None): simple figure
        fig, ax = plt.subplots(figsize=figsize)

    
    # 1. Plot background (all cells in light grey for spatial context)
    ax.scatter(coords_background[:, 0], coords_background[:, 1], 
               color='lightgrey', s=point_size, alpha=background_alpha, rasterized=True)
    
    # 2. Scatter plots for each cell type
    scatter_objects = []
    for i, data in enumerate(cell_type_data):
        sc = ax.scatter(
            data['coords'][:, 0], data['coords'][:, 1],
            c=data['expression'], cmap=cmaps[i], s=point_size, 
            label=data['label'], alpha=scatter_alpha,
            rasterized=True,
            **color_limits # Pass vmin/vmax if specified or if normalized/shared scale is used
        )
        scatter_objects.append(sc)
    
    # 3. Set up colorbars based on layout
    if colorbar_layout == 'side':
        gs = side_cbar_config['gs']
        cbar_cols = side_cbar_config['cbar_cols']
        cbar_rows = side_cbar_config['cbar_rows']

        for i, (sc, data) in enumerate(zip(scatter_objects, cell_type_data)):
            # Column-major (top-to-bottom, then left-to-right)
            row = i % cbar_rows
            col = i // cbar_rows
            cax = fig.add_subplot(gs[row, 1 + col])
            cb = plt.colorbar(sc, cax=cax)

            label_text = data['label'] + (f"\n(n={data['count']})" if show_counts else "")
            # Encourage consistent, readable labels on the right:
            cb.ax.set_ylabel(label_text, rotation=270, labelpad=12, va='bottom')
            cb.ax.yaxis.set_label_position('right')
            cb.ax.tick_params(labelsize=8)
                        
    elif colorbar_layout == 'bottom':
        # Create horizontal colorbars at bottom
        n_bars = len(scatter_objects)
        
        # Calculate positioning for horizontal colorbars
        total_width = 0.8
        bar_margin = 0.02 # space between bars
        bar_width = (total_width - (n_bars - 1) * bar_margin) / n_bars
        
        for i, (sc, data) in enumerate(zip(scatter_objects, cell_type_data)):
            # Position: [left, bottom, width, height]
            left = 0.1 + i * (bar_width + bar_margin)
            cax = fig.add_axes([left, 0.08, bar_width, 0.03])
            cb = plt.colorbar(sc, cax=cax, orientation='horizontal')
            
            # Build label
            label_text = data['label']
            if show_counts:
                label_text += f" (n={data['count']})"
            
            cb.set_label(label_text, fontsize=9)
            
            # Rotate tick labels if needed
            cb.ax.tick_params(labelsize=8)
    
    # colorbar_layout == None means no colorbars
    
    # 4. Apply bounding box to axes if specified
    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
    # 5. Title handling
    if title == 'auto':
        title_text = f"{interaction} score by cell type"
        ax.set_title(title_text, fontsize=14, pad=10)
    elif title is not None:
        ax.set_title(title, fontsize=14, pad=10)
    # If title is None, no title is set
    
    ax.set_axis_off()
    
    # Only use tight_layout if constrained_layout is NOT active
    # This prevents conflicts when using GridSpec with constrained_layout=True
    engine = fig.get_layout_engine()
    if engine is None or "constrained" not in str(engine).lower():
        try:
            plt.tight_layout()
        except RuntimeError:
            pass # Avoid error if no space for tight layout

    
    # 6. Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax
