import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_annotation_metrics(results):
    """
    Extract spatial_coherence_scores, pairwise_scores and silhouette_scores_true
    from a `results` dict and return a single-row DataFrame with flattened columns.

    Column naming:
      - spatial coherence:    "spatial_coherence_{obskey}"
      - silhouette (true):    "silhouette_true_{obskey}"
      - pairwise metrics:     "{metric}_{obskey1}_{obskey2}"   (e.g. "nmi_major_brain_region_cell_type")

    Handles multiple pairwise entries.
    """
    out = {}

    # spatial coherence (sometimes stored under different keys)
    sc = results.get("spatial_coherence_scores") or results.get("static_scores") or {}
    for obskey, val in sc.items():
        out[f"spatial_coherence_{obskey}"] = float(val)

    # silhouette of true labels
    sil = results.get("silhouette_scores_true") or results.get("silhouette_scores") or {}
    for obskey, val in sil.items():
        out[f"silhouette_true_{obskey}"] = float(val)

    # pairwise scores: list of dicts
    pairwise_list = results.get("pairwise_scores") or []
    for idx, entry in enumerate(pairwise_list):
        # support several possible naming conventions for obs keys
        obs1 = entry.get("obs_key_1") or entry.get("obs_key1") or entry.get("obs1") or entry.get("obs_key")
        obs2 = entry.get("obs_key_2") or entry.get("obs_key2") or entry.get("obs2")
        if obs1 is None or obs2 is None:
            # fallback to index to avoid collisions
            obs1 = obs1 or f"pair{idx}"
            obs2 = obs2 or f"pair{idx}"
        for metric_key, metric_val in entry.items():
            # skip the obs key fields themselves
            if metric_key in ("obs_key_1", "obs_key_2", "obs_key1", "obs_key2", "obs1", "obs2", "obs_key"):
                continue
            colname = f"{metric_key}_{obs1}_{obs2}"
            out[colname] = float(metric_val)

    return pd.DataFrame([out])

def plot_clustering_metrics_with_baselines(clust_df, true_labels, figsize=(8, 6), annotate=True):
    """
    Plot clustering sweep metrics from `clust_df` and overlay baseline values
    from `true_labels` when available.

    Parameters
    - clust_df: DataFrame containing columns "k", "model" and metric columns
    - true_labels: single-row DataFrame with baseline columns produced by extract_annotation_metrics()
    - figsize: figure size for each plot
    - annotate: whether to attempt overlaying baselines from true_labels
    """
    metrics = [c for c in clust_df.columns if c not in ("k", "model")]
    models = clust_df["model"].unique()
    x_max = clust_df["k"].max()

    for metric in metrics:
        plt.figure(figsize=figsize)
        for model in models:
            sub = clust_df[clust_df["model"] == model]
            plt.plot(sub["k"], sub[metric], marker="o", label=model)

        if annotate:
            try:
                # 1) cluster spatial coherence -> all true spatial coherences
                if metric == "spatial_coherence_score_cluster":
                    for col in [c for c in true_labels.columns if c.startswith("spatial_coherence_")]:
                        val = float(true_labels[col].iloc[0])
                        plt.axhline(val, linestyle="--", color="gray", linewidth=1)
                        plt.text(x_max + 0.1, val, f"{col}: {val:.3f}", va="center", fontsize=8)

                # 2) silhouette_spatial -> true silhouettes for annotations
                elif "silhouette" in metric:
                    for col in [c for c in true_labels.columns if c.startswith("silhouette_true_")]:
                        val = float(true_labels[col].iloc[0])
                        plt.axhline(val, linestyle="--", color="gray", linewidth=1)
                        plt.text(x_max + 0.1, val, f"{col}: {val:.3f}", va="center", fontsize=8)

                # 3) per-annotation cluster-vs-annotation metrics (nmi/ami/hom/ari)
                else:
                    prefixes = ["nmi_cluster_", "ami_cluster_", "hom_cluster_", "ari_cluster_"]
                    matched = False
                    for p in prefixes:
                        if metric.startswith(p):
                            obskey = metric[len(p):]
                            metric_base = p.split("_")[0]  # e.g. 'nmi'
                            candidates = [
                                c for c in true_labels.columns
                                if (metric_base in c or c.startswith(f"{metric_base}_")) and (obskey in c)
                            ]
                            if not candidates:
                                candidates = [c for c in true_labels.columns if obskey in c]
                            for col in candidates:
                                try:
                                    val = float(true_labels[col].iloc[0])
                                except Exception:
                                    continue
                                plt.axhline(val, linestyle="--", color="gray", linewidth=1)
                                plt.text(x_max + 0.1, val, f"{col}: {val:.3f}", va="center", fontsize=8)
                                matched = True
                                break
                            if matched:
                                break
            except Exception:
                # don't fail plotting if annotation lookup fails
                pass

        plt.title(f"{metric} vs #clusters")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

def summarize_cluster_spatial_metrics(models_dict, baseline_model=None, plot=True):
    """
    Summarize a dict of model results (as produced by evaluate_pipeline).

    Parameters
      - models_dict: dict[name] -> result dict (must contain 'summary' and/or 'clustering')
      - baseline_model: optional key in models_dict to use for extracting annotation baselines;
                        if None the first result with 'summary' is used
      - plot: if True, call plot_clustering_metrics_with_baselines(clustering, true_labels) when possible
    """
    # Performance table
    perf = {}
    for name, res in models_dict.items():
        s = res.get("summary", {})
        perf[name] = {
            "mean_r2": float(s.get("mean_r2", np.nan)),
            "classification_accuracy": float(s.get("classification_accuracy", np.nan)),
            "auc_macro": float(s.get("classification_auc_macro", s.get("auc_macro", np.nan))),
            "auc_weighted": float(s.get("classification_auc_weighted", s.get("auc_weighted", np.nan))),
        }
    perf_df = pd.DataFrame(perf).T

    # Combined clustering DataFrame
    df_list = []
    for name, res in models_dict.items():
        cl = res.get("clustering")
        if cl is None:
            continue
        tmp = cl.copy()
        # handle if 'n_clusters' is index
        if tmp.index.name == "n_clusters" or "n_clusters" in tmp.index.names:
            tmp = tmp.reset_index()
            if "n_clusters" in tmp.columns:
                tmp = tmp.rename(columns={"n_clusters": "k"})
        elif "n_clusters" in tmp.columns:
            tmp = tmp.rename(columns={"n_clusters": "k"})
        elif "k" not in tmp.columns and "k" in tmp.reset_index().columns:
            tmp = tmp.reset_index().rename(columns={"index": "k"})
        else:
            tmp = tmp.reset_index(drop=True)
        tmp["model"] = name
        df_list.append(tmp)

    clust_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    # Baseline annotation metrics (true labels) using extract_annotation_metrics
    base_res = None
    if baseline_model and baseline_model in models_dict:
        base_res = models_dict[baseline_model]
    else:
        for v in models_dict.values():
            if v and "summary" in v:
                base_res = v
                break

    true_labels = pd.DataFrame()
    if base_res is not None:
        try:
            true_labels = extract_annotation_metrics(base_res)
        except Exception:
            true_labels = pd.DataFrame()

    # Optional plotting (safe)
    if plot and not clust_df.empty and not true_labels.empty:
        try:
            plot_clustering_metrics_with_baselines(clust_df, true_labels, annotate=True)
        except Exception:
            pass


def plot_model_metrics(models, metrics, figsize=(8, 5), title="Model Performance", rotate_xticks=0):
    """
    Build a DataFrame of `metrics` for each model in `models` and plot a bar chart.

    Parameters
    - models: dict[name] -> result dict or mapping. If a dict contains a "summary" key,
              metrics are looked up there first.
    - metrics: list of metric keys to extract (e.g. ["mean_r2", "classification_accuracy", "classification_auc_macro"])
    - figsize: tuple for figure size
    - title: plot title
    - rotate_xticks: rotation for x tick labels
    """
    def _get_metric_from_res(res, metric):
        # prefer summary if present
        src = res.get("summary", res) if isinstance(res, dict) else res
        # direct hit
        val = src.get(metric) if isinstance(src, dict) else None

        # fallbacks for common alternate names
        if val is None:
            if metric.endswith("classification_auc_macro") or metric.endswith("auc_macro"):
                val = src.get("auc_macro")
            if val is None and (metric.endswith("classification_auc_weighted") or metric.endswith("auc_weighted")):
                val = src.get("auc_weighted")
            # sometimes stored as plain 'auc_macro' / 'auc_weighted' in summary
            if val is None and metric.startswith("classification_"):
                alt = metric.replace("classification_", "")
                val = src.get(alt)

        try:
            return float(val) if val is not None and not (isinstance(val, str) and val == "") else np.nan
        except Exception:
            return np.nan

    data = {
        model_name: {
            metric: _get_metric_from_res(model_res, metric)
            for metric in metrics
        }
        for model_name, model_res in models.items()
    }

    df = pd.DataFrame(data).T

    ax = df.plot(kind="bar", figsize=figsize)
    ax.set_title(title)
    ax.set_ylabel("Score")
    plt.xticks(rotation=rotate_xticks)

    for container in ax.containers:
        try:
            ax.bar_label(container, fmt="%.2f", padding=2)
        except Exception:
            pass

    plt.tight_layout()
    plt.show()