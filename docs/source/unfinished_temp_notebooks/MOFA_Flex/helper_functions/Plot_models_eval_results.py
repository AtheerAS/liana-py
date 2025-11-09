import math
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def spatial_statistics_summary_plot(models_dict, obs_keys=("cell_type", "major_brain_region"), figsize=(14, 12), ncols=2):
    """
    Line plot summary of spatial/clustering metrics across models.
    - models_dict: mapping name -> dict (must include 'results_df' DataFrame with same columns)
    - obs_keys: tuple (obskey1, obskey2). Creates one figure per obs_key (facet).
    Notes:
      - columns expected in each results_df:
          - 'n_clusters'
          - 'spatial_coherence_clusters_k3', 'silhouette_clusters'
          - '{metric}_vs_{obskey}' for metric in ['nmi','ami','homogeneity','ari']
          - baseline columns starting with 'baseline_...'
      - Baseline horizontal lines are drawn when matching baseline columns are found.
    """

    metric_types = [
        ("spatial_coherence_clusters_k3", "Spatial coherence (k=3)"),
        ("silhouette_clusters", "Silhouette (spatial)"),
        ("nmi", "NMI"),
        ("ami", "AMI"),
        ("homogeneity", "Homogeneity"),
        ("ari", "ARI"),
    ]

    # pick a reference results_df to detect baseline column names (assume shared across models)
    ref_df = None
    for v in models_dict.values():
        if isinstance(v, dict) and "results_df" in v and isinstance(v["results_df"], pd.DataFrame):
            ref_df = v["results_df"]
            break
    if ref_df is None:
        raise ValueError("No results_df found in models_dict entries")

    # helper to find baseline column for a given metric (returns column name or None)
    def _find_baseline_col(metric):
        candidates = [c for c in ref_df.columns if c.startswith("baseline_") and metric in c]
        return candidates[0] if candidates else None

    # For each obs_key create a multi-panel figure (one subplot per metric type)
    for obs in obs_keys:
        n_plots = len(metric_types)
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        fig.suptitle(f"Spatial / Agreement metrics (obs_key: {obs})", fontsize=14)

        for i, (mkey, mlabel) in enumerate(metric_types):
            ax = axes[i // ncols, i % ncols]

            # determine the column name in results_df for this metric and obs
            if mkey in ["spatial_coherence_clusters_k3", "silhouette_clusters"]:
                col_name = mkey
            else:
                col_name = f"{mkey}_vs_{obs}"

            # plot each model's line (skip if column missing)
            any_plotted = False
            for model_name, mdl in models_dict.items():
                if not (isinstance(mdl, dict) and "results_df" in mdl):
                    continue
                df = mdl["results_df"]
                if "n_clusters" not in df.columns:
                    continue
                if col_name not in df.columns:
                    # skip this model for this metric
                    continue
                ax.plot(df["n_clusters"], df[col_name], marker="o", label=model_name)
                any_plotted = True

            # plot baseline if available
            # For spatial/silhouette baseline column names include the obs key
            if mkey in ["spatial_coherence_clusters_k3", "silhouette_clusters"]:
                bl_col = f"baseline_spatial_coherence_{obs}" if mkey.startswith("spatial") else f"baseline_silhouette_{obs}"
                if bl_col not in ref_df.columns:
                    # fallback - try general search
                    bl_col = _find_baseline_col("spatial_coherence") if mkey.startswith("spatial") else _find_baseline_col("silhouette")
                if bl_col in ref_df.columns:
                    bl_val = float(ref_df[bl_col].iloc[0])
                    ax.axhline(bl_val, color="gray", linestyle="--", linewidth=1.5, label="baseline")
                    any_plotted = True
            else:
                # agreement metrics: baseline is the pair baseline between the two obs_keys
                # search for any baseline column that contains the metric name
                bl_col = _find_baseline_col(mkey)
                if bl_col is not None:
                    bl_val = float(ref_df[bl_col].iloc[0])
                    ax.axhline(bl_val, color="gray", linestyle="--", linewidth=1.5, label="baseline (obs pair)")
                    any_plotted = True

            if not any_plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")

            ax.set_xlabel("n_clusters")
            ax.set_xticks(ref_df["n_clusters"] if "n_clusters" in ref_df.columns else [])
            ax.set_ylabel(mlabel)
            ax.set_title(mlabel)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(loc="best", fontsize="small")

        # remove unused axes
        total_axes = nrows * ncols
        for j in range(n_plots, total_axes):
            fig.delaxes(axes[j // ncols, j % ncols])

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def _read_mean_r2_from_path(p):
    p = Path(p)
    if not p.exists():
        return np.nan
    try:
        df = pd.read_csv(p, index_col=0)
        num = df.select_dtypes(include=[np.number])
        if num.size == 0:
            return np.nan
        # prefer group_1 if present (matches earlier notebook)
        if "group_1" in num.columns:
            return float(num["group_1"].mean())
        # otherwise average across numeric columns
        return float(num.mean(axis=0).mean())
    except Exception:
        return np.nan

def _read_accuracy_from_report(path):
    p = Path(path)
    if not p.exists():
        return np.nan
    try:
        txt = p.read_text().splitlines()
        for ln in txt:
            if ln.strip().lower().startswith("accuracy"):
                # line like "Accuracy: 0.8123"
                parts = ln.split(":")
                if len(parts) >= 2:
                    return float(parts[1].strip())
        # fallback: try JSON metrics file
        return np.nan
    except Exception:
        return np.nan

def _read_metrics_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def model_performance_summary_plot(models_dict, figsize=(8, 4), fmt="{:.3f}", title="Model performance summary"):
    """
    Create a grouped bar plot of model performance across three metrics:
      - Average R2 (mean R2 across views / modalities)
      - Classification accuracy (parsed from classification_report written by evaluate_mofaflex)
      - Classification weighted AUC (classification_auc_weighted or from saved json)

    models_dict: mapping name -> dict produced by evaluate_mofaflex (or similar).
    Returns (df, fig, ax)
    """
    rows = []
    for name, info in models_dict.items():
        mean_r2 = np.nan
        acc = np.nan
        auc_w = np.nan

        # Mean R2: try explicit numeric key, then r2_summary csv path
        if isinstance(info, dict):
            if "mean_r2" in info and pd.notna(info["mean_r2"]):
                mean_r2 = float(info["mean_r2"])
            elif "r2_summary" in info and info["r2_summary"] is not None:
                mean_r2 = _read_mean_r2_from_path(info["r2_summary"])
            # direct r2 dataframe may be provided
            elif "results_df" in info and hasattr(info["results_df"], "attrs") and "mean_r2" in info["results_df"].attrs:
                try:
                    mean_r2 = float(info["results_df"].attrs["mean_r2"])
                except Exception:
                    mean_r2 = np.nan

            # Accuracy: parse classification_report file if present
            if "classification_report" in info and info["classification_report"] is not None:
                acc = _read_accuracy_from_report(info["classification_report"])
            # try JSON metrics file or keys
            if "classification_metrics.json" in info and info["classification_metrics.json"] is not None:
                jm = _read_metrics_json(info["classification_metrics.json"])
                acc = acc if not np.isnan(acc) else jm.get("accuracy", acc)
                auc_w = jm.get("auc_weighted", auc_w)
            # direct numeric keys
            if "classification_auc_weighted" in info and pd.notna(info["classification_auc_weighted"]):
                auc_w = float(info["classification_auc_weighted"])
            if "classification_auc_macro" in info and pd.notna(info["classification_auc_macro"]) and np.isnan(auc_w):
                auc_w = float(info["classification_auc_macro"])
            # sometimes a metrics json path is saved under classification_metrics
            if np.isnan(auc_w) and "classification_metrics" in info and info["classification_metrics"] is not None:
                jm = _read_metrics_json(info["classification_metrics"])
                auc_w = jm.get("auc_weighted", auc_w)

        rows.append({"model": name, "mean_r2": mean_r2, "accuracy": acc, "auc_weighted": auc_w})

    df = pd.DataFrame(rows).set_index("model")
    # ensure numeric dtype
    df = df.astype(float)

    # Plot grouped bars: x axis = metrics, groups = metric categories, bars per model
    metrics = ["mean_r2", "accuracy", "auc_weighted"]
    metric_labels = {"mean_r2": "Average R2", "accuracy": "Accuracy", "auc_weighted": "AUC (weighted)"}

    n_metrics = len(metrics)
    n_models = len(df)
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_models)

    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    for i, (model_name, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        positions = x - 0.4 + i * bar_width + bar_width / 2
        bars = ax.bar(positions, vals, width=bar_width, label=model_name, color=palette[i])
        # annotate bars
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                lbl = "n/a"
                ax.text(b.get_x() + b.get_width() / 2, 0.01, lbl, ha="center", va="bottom", fontsize=8, rotation=0)
            else:
                ax.text(b.get_x() + b.get_width() / 2, h + max(0.01, 0.01 * abs(h)), fmt.format(h), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics])
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(bottom=0)  # performance metrics are non-negative here
    fig.tight_layout()
    return df, fig, ax

