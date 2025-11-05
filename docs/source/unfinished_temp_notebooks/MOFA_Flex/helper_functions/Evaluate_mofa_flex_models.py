# Cell 1: Imports
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, silhouette_score,
    normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, adjusted_rand_score
)

import matplotlib.pyplot as plt
from plotnine import ggplot
import mofaflex as mfl

#Utility Functions
def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def spatial_coherence_score(X, labels, k=5):
    X = np.asarray(X)
    labels = np.asarray(labels)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]
    same_cluster_counts = np.sum(labels[indices] == labels[:, None], axis=1)
    scs = np.mean(same_cluster_counts / k)
    return scs

def save_plot_object(fig, path):
    if isinstance(fig, ggplot):
        fig.save(path, dpi=300)
    else:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
# Cell 3: Evaluate single MOFA model outputs (plots + R2)
def evaluate_single_mofa_model(model, output_dir, group="group_1", show_plots=False):
    out = {}
    out_dir = _ensure_dir(output_dir)

    try:
        fig = mfl.pl.factor_correlation(model)
        p = out_dir / "factor_correlation.png"
        save_plot_object(fig, p)
        out["factor_correlation_path"] = str(p)
    except Exception as e:
        out["factor_correlation_error"] = str(e)

    try:
        fig = mfl.pl.variance_explained(model, figsize=(8, 8))
        p = out_dir / "variance_explained.png"
        save_plot_object(fig, p)
        out["variance_explained_path"] = str(p)
    except Exception as e:
        out["variance_explained_error"] = str(e)

    try:
        r2 = model.get_r2(total=True)
        out["mean_r2"] = float(r2[group].mean())
    except Exception as e:
        out["mean_r2_error"] = str(e)

    return out
# Cell 4: Plot weights distribution
def plot_weights_distribution(model, output_dir):
    out_dir = _ensure_dir(output_dir)
    try:
        weights_dict = model.get_weights()
        plt.figure(figsize=(12, 6))
        for key, df in weights_dict.items():
            values = df.values.flatten()
            sns.kdeplot(values, label=key, fill=False)
        plt.title("Distribution of Weights per View")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.legend(title="View/Key", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        p = out_dir / "weights_distribution.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        return str(p)
    except Exception as e:
        return {"error": str(e)}
# Cell 6: Logistic regression on factor space
def classification_on_factors(factors, lrdata, output_dir, classification_label_key=None,
                              test_size=0.3, random_state=42):
    
    out_dir = _ensure_dir(output_dir)

    obskey1 = lrdata.obs[classification_label_key]
    factors = factors.copy()
    factors["anno"] = obskey1.reindex(factors.index)
    factors = factors.dropna()

    drop_cols = [c for c in ["cluster", "anno"] if c in factors.columns]
    X = factors.drop(columns=drop_cols).values
    y = factors["anno"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    clf = LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_).to_csv(out_dir / "confusion_matrix.csv")

    try:
        y_proba = clf.predict_proba(X_test)
        auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except:
        auc_macro = np.nan
        auc_weighted = np.nan

    metrics = {
        "accuracy": acc,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "classification_report": cls_report,
    }

    pd.DataFrame(cls_report).to_csv(out_dir / "classification_report.csv")
    return metrics, str(out_dir / "classification_report.csv")
# Cell 5: Clustering sweep (cleanest output, full annotation metrics)
def clustering_sweep(
    factors,
    lrdata_sub,
    output_dir,
    n_clusters_list=None,
    spatial_k_for_scs=3,
    obs_keys=None,
    spatial_key="spatial",
):

    out_dir = _ensure_dir(output_dir)

    if n_clusters_list is None:
        n_clusters_list = list(range(2, 10))
    if obs_keys is None:
        obs_keys = []

    # Drop non-factor columns
    drop_cols = [c for c in ["cluster", "anno"] if c in factors.columns]
    X_f = factors.drop(columns=drop_cols)
    X_std = StandardScaler().fit_transform(X_f.values)

    # Validate spatial key
    if spatial_key not in lrdata_sub.obsm:
        raise KeyError(
            f"spatial_key '{spatial_key}' not found in lrdata_sub.obsm. "
            f"Available keys: {list(lrdata_sub.obsm.keys())}"
        )

    X_spatial = lrdata_sub.obsm[spatial_key]

    # Ensure same ordering
    common_idx = factors.index.intersection(lrdata_sub.obs_names)
    if not list(common_idx) == list(lrdata_sub.obs_names):
        X_spatial = lrdata_sub.obsm[spatial_key][
            [lrdata_sub.obs_names.get_loc(i) for i in common_idx], :
        ]

    # Obs label arrays
    obs_arrays = {
        key: (lrdata_sub.obs[key].astype(str).values if key in lrdata_sub.obs else None)
        for key in obs_keys
    }

    # ---------------------- STATIC LABEL METRICS ----------------------

    # Spatial coherence of true labels
    obs_static_scores = {}
    for key, arr in obs_arrays.items():
        if arr is not None:
            obs_static_scores[key] = float(
                spatial_coherence_score(X_spatial, arr, k=spatial_k_for_scs)
            )
        else:
            obs_static_scores[key] = float("nan")

    static_df = pd.DataFrame(
        [{"obs_key": k, "spatial_coherence_score": v} for k, v in obs_static_scores.items()]
    )
    static_csv = out_dir / "annotation_static_metrics.csv"
    static_df.to_csv(static_csv, index=False)

    # Silhouette of true labels in factor space
    silhouette_scores_true = {}
    for key, arr in obs_arrays.items():
        try:
            silhouette_scores_true[key] = float(silhouette_score(X_spatial, arr))
        except:
            silhouette_scores_true[key] = float("nan")

    sil_df = pd.DataFrame([silhouette_scores_true])
    sil_csv = out_dir / "annotation_silhouette_scores.csv"
    sil_df.to_csv(sil_csv, index=False)

    # Pairwise consistency between annotation keys
    pairwise_scores = []
    obs_keys_list = list(obs_arrays.keys())
    for i in range(len(obs_keys_list)):
        for j in range(i + 1, len(obs_keys_list)):
            k1, k2 = obs_keys_list[i], obs_keys_list[j]
            arr1, arr2 = obs_arrays[k1], obs_arrays[k2]

            if arr1 is None or arr2 is None:
                continue

            pairwise_scores.append({
                "obs_key_1": k1,
                "obs_key_2": k2,
                "nmi": float(normalized_mutual_info_score(arr1, arr2)),
                "ami": float(adjusted_mutual_info_score(arr1, arr2)),
                "hom": float(homogeneity_score(arr1, arr2)),
                "ari": float(adjusted_rand_score(arr1, arr2)),
            })

    pairwise_df = pd.DataFrame(pairwise_scores)
    pairwise_csv = out_dir / "annotation_pairwise_metrics.csv"
    pairwise_df.to_csv(pairwise_csv, index=False)

    # ---------------------- CLUSTERING SWEEP ----------------------
    results = []
    for k in n_clusters_list:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_std)

        row = {"n_clusters": k}

        # Silhouette over spatial coordinates
        try:
            row["silhouette_spatial"] = float(silhouette_score(X_spatial, labels))
        except:
            row["silhouette_spatial"] = float("nan")

        # Cluster vs annotation agreement
        for key, arr in obs_arrays.items():
            pref = key.replace(" ", "_")

            if arr is None:
                row[f"nmi_cluster_{pref}"] = float("nan")
                row[f"ami_cluster_{pref}"] = float("nan")
                row[f"hom_cluster_{pref}"] = float("nan")
                row[f"ari_cluster_{pref}"] = float("nan")
                continue

            try:
                row[f"nmi_cluster_{pref}"] = float(normalized_mutual_info_score(arr, labels))
            except Exception:
                row[f"nmi_cluster_{pref}"] = float("nan")

            try:
                row[f"ami_cluster_{pref}"] = float(adjusted_mutual_info_score(arr, labels))
            except Exception:
                row[f"ami_cluster_{pref}"] = float("nan")

            try:
                row[f"hom_cluster_{pref}"] = float(homogeneity_score(arr, labels))
            except Exception:
                row[f"hom_cluster_{pref}"] = float("nan")

            try:
                row[f"ari_cluster_{pref}"] = float(adjusted_rand_score(arr, labels))
            except Exception:
                row[f"ari_cluster_{pref}"] = float("nan")

        # Cluster spatial coherence
        try:
            row["spatial_coherence_score_cluster"] = float(
                spatial_coherence_score(X_spatial, labels, k=spatial_k_for_scs)
            )
        except:
            row["spatial_coherence_score_cluster"] = float("nan")

        results.append(row)

    results_df = pd.DataFrame(results).set_index("n_clusters").sort_index()
    sweep_csv = out_dir / "clustering_metrics_sweep.csv"
    results_df.to_csv(sweep_csv)

    return (
        results_df,
        obs_static_scores,
        str(sweep_csv),
        str(static_csv),
        pairwise_scores,
        str(pairwise_csv),
        silhouette_scores_true,
        str(sil_csv),
    )

# Cell 7: Full pipeline
def evaluate_pipeline(model, lrdata, output_dir, group="group_1",
                      n_clusters_list=None, obs_keys=None, classification_label_key=None, spatial_key="spatial"):

    out_dir = _ensure_dir(output_dir)
    plots_dir = _ensure_dir(out_dir / "plots")
    stats_dir = _ensure_dir(out_dir / "stats")

    if obs_keys is None or len(obs_keys) == 0:
        raise ValueError("obs_keys must be provided")

    if classification_label_key is None:
        raise ValueError("classification_label_key must be provided, it is the ground truth label for classification")

    # Model plots & R2
    mofa_results = evaluate_single_mofa_model(model, plots_dir, group=group)
    _ = plot_weights_distribution(model, plots_dir)

    # Extract factors
    factors = model.get_factors()[group]

    # Clean AnnData
    lrdata_clean = lrdata.copy()
    for key in obs_keys:
        if key in lrdata_clean.obs:
            lrdata_clean = lrdata_clean[~lrdata_clean.obs[key].isna()].copy()

    # Align indices
    common_idx = factors.index.intersection(lrdata_clean.obs_names)
    factors_sub = factors.loc[common_idx].copy()
    lrdata_sub = lrdata_clean[common_idx, :].copy()

    # Classification
    try:
        class_metrics, class_report_path = classification_on_factors(
            factors_sub, lrdata_sub, stats_dir,
            classification_label_key=classification_label_key
        )
    except Exception as e:
        class_metrics = {"error": str(e)}
        class_report_path = ""

    # Clustering analysis (NEW expanded returns)
    (
        clustering_df,
        static_scores,
        clustering_csv,
        static_csv,
        pairwise_scores,
        pairwise_csv,
        silhouette_scores_true,
        silhouette_csv
    ) = clustering_sweep(
        factors_sub, lrdata_sub, stats_dir,
        n_clusters_list=n_clusters_list,
        obs_keys=obs_keys,
        spatial_key=spatial_key
    )

    # Summary table
    summary = {
        "mean_r2": mofa_results.get("mean_r2", np.nan),
        "classification_accuracy": class_metrics.get("accuracy", np.nan),
        "classification_auc_macro": class_metrics.get("auc_macro", np.nan),
        "classification_auc_weighted": class_metrics.get("auc_weighted", np.nan),
        "clustering_csv": clustering_csv,
        "static_annotation_metrics_csv": static_csv,
        "annotation_pairwise_csv": pairwise_csv,
        "annotation_silhouette_csv": silhouette_csv,
        "used_obs_keys": ",".join(obs_keys),
        "used_label_key_for_classification": classification_label_key,
    }

    # Add static spatial coherence
    for key, value in static_scores.items():
        summary[f"spatial_coherence_{key}"] = value

    # Add silhouette of true labels in factor space
    for key, value in silhouette_scores_true.items():
        summary[f"silhouette_true_{key}"] = value

    # Save summary CSV
    # The output CSV "summary_metrics.csv" contains a single row with the following columns:
    # - mean_r2: Mean R2 value from the MOFA model for the specified group.
    # - classification_accuracy: Accuracy of logistic regression classification on factor space.
    # - classification_auc_macro: Macro-averaged AUC for classification.
    # - classification_auc_weighted: Weighted-averaged AUC for classification.
    # - clustering_csv: Path to the clustering metrics sweep CSV.
    # - static_annotation_metrics_csv: Path to the static annotation metrics CSV.
    # - annotation_pairwise_csv: Path to the annotation pairwise metrics CSV.
    # - annotation_silhouette_csv: Path to the annotation silhouette scores CSV.
    # - used_obs_keys: Comma-separated list of observation keys used.
    # - used_label_key_for_classification: The label key used for classification.
    # - spatial_coherence_{key}: Spatial coherence score for each obs key.
    # - silhouette_true_{key}: Silhouette score of true labels in factor space for each obs key.
    pd.DataFrame([summary]).to_csv(stats_dir / "summary_metrics.csv", index=False)

    return {
        "summary": summary,
        "clustering": clustering_df,
        "classification_metrics": class_metrics,
        "spatial_coherence_scores": static_scores,
        "pairwise_scores": pairwise_scores,
        "silhouette_scores_true": silhouette_scores_true,
        "class_report": class_report_path,
        "plots_dir": str(plots_dir),
        "stats_dir": str(stats_dir),
    }
