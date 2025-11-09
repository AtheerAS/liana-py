from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def spatial_coherence_score(X, labels, k=5):
    X = np.asarray(X)
    labels = np.asarray(labels)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)

    # exclude each point itself (indices[:,0])
    indices = indices[:, 1:]
    
    same_cluster_counts = np.sum(labels[indices] == labels[:, None], axis=1)
    scs = np.mean(same_cluster_counts / k)
    return scs


def evaluate_mofaflex(
    model,
    lrdata,
    output_dir,
    group="group_1",
    n_clusters=None,
    obs_keys=("cell_type", "major_brain_region"),
    classification_key=None,
    spatial_key="X_spatial_coords",
    random_state=42,
):
    """
    Run a benchmarking pipeline for a MOFA-FLEX model and save results & figures.

    Parameters
    - model: MOFAFLEX model object (must implement get_factors(), get_weights(), get_r2()).
    - lrdata: AnnData-like object with .obs and .obsm available.
    - output_dir: str or Path where outputs (csv, figs, reports) will be saved.
    - group: group name used when calling model.get_factors()[group].
    - n_clusters: iterable of ints (cluster counts). If None, uses range(2,10).
    - obs_keys: tuple/list of observation keys from lrdata.obs to use for baselines/metrics.
    - classification_key: key in lrdata.obs to use for regression/classification (optional).
    - spatial_key: key in lrdata.obsm storing spatial coordinates (default "X_spatial_coords").
    - random_state: random seed for clustering / classifiers.

    Returns:
    - dict with paths to saved files and results DataFrame under "results_df".
    """
    out = {}
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if n_clusters is None:
        n_clusters = list(range(2, 10))
    else:
        n_clusters = list(n_clusters)

    # 1) Factors and alignment with lrdata
    factors_all = model.get_factors()
    if group not in factors_all:
        raise KeyError(f"group '{group}' not found in model factors")
    factors = factors_all[group].copy()

    # intersect indices and subset
    common_idx = lrdata.obs.index.intersection(factors.index)
    factors = factors.loc[common_idx]
    # standardize factor values for clustering/classification
    X = StandardScaler().fit_transform(factors.values)

    # ensure spatial coords aligned to lrdata.obs
    spatial_coords = np.asarray(lrdata.obsm[spatial_key])
    if spatial_coords.shape[0] != lrdata.obs.shape[0]:
        raise ValueError("spatial coords length does not match lrdata.obs length")

    # helper to compute baseline metrics per obs_key (dropna)
    def _baseline_spatial_and_silhouette(obs_key):
        labels = lrdata.obs[obs_key].astype(str).values
        mask = pd.notna(labels)
        labels_f = labels[mask]
        coords_f = spatial_coords[mask]
        scs = spatial_coherence_score(coords_f, labels_f, k=3)
        try:
            sil = silhouette_score(coords_f, labels_f)
        except Exception:
            sil = float("nan")
        return float(scs), float(sil)

    # baseline pairwise metrics between first two obs_keys if available
    baseline_pair = {}
    if len(obs_keys) >= 2:
        a = lrdata.obs[obs_keys[0]].astype(str).values
        b = lrdata.obs[obs_keys[1]].astype(str).values
        mask = pd.notna(a) & pd.notna(b)
        if mask.sum() > 0:
            baseline_pair["nmi"] = float(normalized_mutual_info_score(a[mask], b[mask]))
            baseline_pair["ami"] = float(adjusted_mutual_info_score(a[mask], b[mask]))
            baseline_pair["homogeneity"] = float(homogeneity_score(a[mask], b[mask]))
            baseline_pair["ari"] = float(adjusted_rand_score(a[mask], b[mask]))
        else:
            baseline_pair = {"nmi": np.nan, "ami": np.nan, "homogeneity": np.nan, "ari": np.nan}

    # compute baselines per obs_key
    baselines = {}
    for k in obs_keys:
        scs, sil = _baseline_spatial_and_silhouette(k)
        baselines[f"baseline_spatial_coherence_{k}"] = scs
        baselines[f"baseline_silhouette_{k}"] = sil

    if baseline_pair:
        baselines.update(
            {
                f"baseline_nmi_{obs_keys[0]}_vs_{obs_keys[1]}": baseline_pair["nmi"],
                f"baseline_ami_{obs_keys[0]}_vs_{obs_keys[1]}": baseline_pair["ami"],
                f"baseline_homogeneity_{obs_keys[0]}_vs_{obs_keys[1]}": baseline_pair["homogeneity"],
                f"baseline_ari_{obs_keys[0]}_vs_{obs_keys[1]}": baseline_pair["ari"],
            }
        )

    # 2) Clustering evaluation across cluster counts
    results = []
    idx_lr = lrdata.obs.index  # full index for lrdata
    for k in n_clusters:
        km = KMeans(n_clusters=k, random_state=random_state)
        preds = km.fit_predict(X)  # aligned to factors.index
        preds_series = pd.Series(preds, index=factors.index, name=f"cluster_{k}")

        # add to factors and lrdata (only for matching idx)
        factors[f"cluster_{k}"] = preds_series
        preds_lr = preds_series.reindex(idx_lr)
        lrdata.obs[f"cluster_{k}"] = preds_lr.values

        # prepare mask to exclude NaNs (cells without cluster assignment)
        labels_full = lrdata.obs[f"cluster_{k}"]
        mask = pd.notna(labels_full)
        labels = labels_full[mask].astype(str).values
        coords = spatial_coords[mask.values]

        # spatial coherence (k=3) and silhouette (on spatial coords)
        scs_clusters = float(spatial_coherence_score(coords, labels, k=3)) if coords.shape[0] > 0 else np.nan
        try:
            sil_clusters = float(silhouette_score(coords, labels)) if coords.shape[0] > 0 else np.nan
        except Exception:
            sil_clusters = float("nan")

        # agreement metrics vs each obs_key
        row = {
            "n_clusters": int(k),
            "n_cells": int(len(lrdata.obs)),
            "largest_cluster_size": int(lrdata.obs[f"cluster_{k}"].value_counts(dropna=True).max()),
            "smallest_cluster_size": int(lrdata.obs[f"cluster_{k}"].value_counts(dropna=True).min()),
            "n_unique_clusters_assigned": int(lrdata.obs[f"cluster_{k}"].value_counts(dropna=True).size),
            "spatial_coherence_clusters_k3": scs_clusters,
            "silhouette_clusters": sil_clusters,
        }

        # compare to each obs_key
        for obs in obs_keys:
            true_labels = lrdata.obs[obs].astype(str).values[mask.values]
            if len(true_labels) == 0:
                nmi = ami = hom = ari = np.nan
            else:
                nmi = normalized_mutual_info_score(true_labels, labels)
                ami = adjusted_mutual_info_score(true_labels, labels)
                hom = homogeneity_score(true_labels, labels)
                ari = adjusted_rand_score(true_labels, labels)
            row.update(
                {
                    f"nmi_vs_{obs}": float(nmi),
                    f"ami_vs_{obs}": float(ami),
                    f"homogeneity_vs_{obs}": float(hom),
                    f"ari_vs_{obs}": float(ari),
                }
            )

        # include baselines
        row.update(baselines)
        results.append(row)

    results_df = pd.DataFrame(results)
    csv_path = outdir / "clustering_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    out["results_df"] = results_df
    out["clustering_csv"] = str(csv_path)

    # 3) Plot metrics (similar to notebook)
    sns.set(style="whitegrid")
    # prepare dynamic plots: spatial coherence, silhouette, and for each metric type across obs_keys
    metric_specs = []
    metric_specs.append(
        {"title": "Spatial coherence (clusters) vs baselines", "cols": ["spatial_coherence_clusters_k3"], "baselines": [(f"baseline_spatial_coherence_{obs_keys[0]}", f"baseline - {obs_keys[0]}")] if obs_keys else []}
    )
    metric_specs.append(
        {"title": "Silhouette (clusters) vs baselines", "cols": ["silhouette_clusters"], "baselines": [(f"baseline_silhouette_{obs_keys[0]}", f"baseline - {obs_keys[0]}")] if obs_keys else []}
    )
    # add NMI/AMI/Hom/ARI with all provided obs_keys
    for metric_name in ["nmi", "ami", "homogeneity", "ari"]:
        cols = [f"{metric_name}_vs_{obs}" for obs in obs_keys]
        bas = []
        # add pair baseline if exists (first two)
        if len(obs_keys) >= 2 and f"baseline_{metric_name}_{obs_keys[0]}_vs_{obs_keys[1]}" in baselines:
            bas = [(f"baseline_{metric_name}_{obs_keys[0]}_vs_{obs_keys[1]}", f"baseline ({obs_keys[0]} vs {obs_keys[1]})")]
        metric_specs.append({"title": f"{metric_name.upper()}: cluster vs true labels", "cols": cols, "baselines": bas})

    # create combined figure
    n_plots = len(metric_specs)
    ncols = 2
    nrows = (n_plots + 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), squeeze=False)
    baseline_colors = ["gray", "black", "tab:orange", "tab:green"]
    baseline_linestyles = ["--", ":", "-.", "-"]

    for i, spec in enumerate(metric_specs):
        ax = axes[i // ncols, i % ncols]
        for col in spec["cols"]:
            if col in results_df.columns:
                ax.plot(results_df["n_clusters"], results_df[col], marker="o", label=col)
        for j, (bl_col, bl_label) in enumerate(spec.get("baselines", [])):
            if bl_col in results_df.columns:
                bl_val = float(results_df[bl_col].iloc[0])
                color = baseline_colors[j % len(baseline_colors)]
                ls = baseline_linestyles[j % len(baseline_linestyles)]
                ax.axhline(bl_val, color=color, linestyle=ls, linewidth=1.5, label=bl_label)
        ax.set_xlabel("n_clusters")
        ax.set_xticks(results_df["n_clusters"])
        ax.set_title(spec["title"])
        ax.legend(loc="best", fontsize="small")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    # hide unused axes
    total_axes = nrows * ncols
    for j in range(n_plots, total_axes):
        fig.delaxes(axes[j // ncols, j % ncols])

    fig.tight_layout()
    metrics_path = outdir / "clustering_metrics_summary.png"
    fig.savefig(metrics_path, dpi=150)
    plt.close(fig)
    out["clustering_metrics_plot"] = str(metrics_path)

    # 4) Weight distributions and model diagnostics
    try:
        weights_dict = model.get_weights()
        fig = plt.figure(figsize=(8, 5))
        for key, df in weights_dict.items():
            values = df.values.flatten()
            sns.kdeplot(values, label=str(key), fill=False)
        plt.title("Distribution of Weights per View")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.legend(title="View/Key", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        wpath = outdir / "weights_distribution.png"
        fig.savefig(wpath, dpi=150)
        plt.close(fig)
        out["weights_distribution"] = str(wpath)
    except Exception:
        out["weights_distribution"] = None

    # --- Factor correlation (library-agnostic) ---
    try:
        fac = model.get_factors()[group].copy()  # cells x factors (columns are factors)
        # Ensure fac is a DataFrame
        if not isinstance(fac, pd.DataFrame):
            fac = pd.DataFrame(fac)
        corr = fac.corr(method="pearson")  # factor-factor correlation

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr.values, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(corr.shape[0]))
        ax.set_yticklabels(corr.index)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")
        ax.set_title("Factor Correlation")
        fig.tight_layout()

        fpath = outdir / "factor_correlation.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["factor_correlation"] = str(fpath)
    except Exception:
        out["factor_correlation"] = None

# --- Variance explained heatmap: factors (rows) x views (cols) for group_1 ---
    try:
        ve_mat = model.get_r2(ordered=True)["group_1"]  # rows=factors, cols=views
        if isinstance(ve_mat, pd.Series):
            ve_mat = ve_mat.to_frame()
        ve_mat = ve_mat.apply(pd.to_numeric, errors="coerce")
        vmax = float(np.nanmax(ve_mat.values)) if np.isfinite(ve_mat.values).any() else 1.0

        # Scale figure size to content
        fig_w = max(6, 0.48 * ve_mat.shape[1])
        fig_h = max(5, 0.42 * ve_mat.shape[0])

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(
            ve_mat.values,
            aspect="auto",
            cmap="Reds_r",   # lighter = higher variance (as in your example)
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )

        # Ticks & labels
        ax.set_xticks(np.arange(ve_mat.shape[1]))
        ax.set_xticklabels(ve_mat.columns, rotation=90)
        ax.set_yticks(np.arange(ve_mat.shape[0]))
        ax.set_yticklabels(ve_mat.index)

        # Title strip & colorbar like the example
        ax.set_title("group_1", pad=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label("Variance explained")

        fig.tight_layout()
        vpath = outdir / "variance_explained.png"
        fig.savefig(vpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["variance_explained"] = str(vpath)
    except Exception:
        out["variance_explained"] = None

    # save R2 summary if available
    try:
        r2 = model.get_r2(total=True)
        r2_path = outdir / "r2_summary.csv"
        r2.to_csv(r2_path)
        out["r2_summary"] = str(r2_path)
    except Exception:
        out["r2_summary"] = None

    # 5) Classification (optional)
    if classification_key is not None:
        # build factors with annotation, drop NaNs
        fac = model.get_factors()[group].copy()
        fac["anno"] = lrdata.obs[classification_key].reindex(fac.index)
        fac = fac.dropna(subset=["anno"])
        Xc = fac.drop(columns=["anno"]).values
        y = fac["anno"].values

        X_train, X_test, y_train, y_test = train_test_split(
            Xc, y, test_size=0.3, random_state=random_state, stratify=y
        )
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced", multi_class="multinomial")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        crep = classification_report(y_test, y_pred, output_dict=False)
        # save report
        report_path = outdir / "classification_report.txt"
        with open(report_path, "w") as fh:
            fh.write(f"Accuracy: {acc:.4f}\n\n")
            fh.write(crep)
        out["classification_report"] = str(report_path)

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix")
        cm_path = outdir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out["confusion_matrix"] = str(cm_path)

        # multiclass ROC AUC (OvR) if possible
        try:
            y_proba = clf.predict_proba(X_test)
            auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            auc_weighted = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            out["classification_auc_macro"] = float(auc_macro)
            out["classification_auc_weighted"] = float(auc_weighted)
            # save numeric summary
            with open(outdir / "classification_metrics.json", "w") as fh:
                json.dump({"accuracy": float(acc), "auc_macro": float(auc_macro), "auc_weighted": float(auc_weighted)}, fh)
        except Exception:
            out["classification_auc_macro"] = None
            out["classification_auc_weighted"] = None

    return out