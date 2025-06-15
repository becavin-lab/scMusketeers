import os
import logging
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


logger = logging.getLogger("Sc-Musketeers")

sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..tools.utils import nan_to_0
except ImportError:
    from scmusketeers.tools.utils import nan_to_0


def metric_confusion_matrix(workflow, y_pred, y_true, group, save_dir):
    logger.debug("Calculate confusion matrix")
    labels = list(
        set(np.unique(y_true)).union(set(np.unique(y_pred)))
    )
    cm_no_label = confusion_matrix(y_true, y_pred)
    logger.debug(f"ConfMatrix no label : {cm_no_label.shape}")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    logger.debug(f"ConfMatrix label : {cm.shape}")
    cm_to_plot = pd.DataFrame(
        cm_norm, index=labels, columns=labels
    )
    cm_to_save = pd.DataFrame(cm, index=labels, columns=labels)
    cm_to_plot = cm_to_plot.fillna(value=0)
    cm_to_save = cm_to_save.fillna(value=0)
    cm_to_save.to_csv(os.path.join(save_dir,f"confusion_matrix_{group}.csv"))
    workflow.run[
        f"evaluation/{group}/confusion_matrix_file"
    ].track_files(os.path.join(save_dir,f"confusion_matrix_{group}.csv"))
    size = len(labels)
    f, ax = plt.subplots(figsize=(size / 1.5, size / 1.5))
    sns.heatmap(
        cm_to_plot,
        annot=True,
        ax=ax,
        fmt=".2f",
        vmin=0,
        vmax=1,
    )
    show_mask = np.asarray(cm_to_plot > 0.01)
    logger.debug(f"ConfMatrix label df : {cm_to_plot.shape}")
    for text, show_annot in zip(
        ax.texts,
        (element for row in show_mask for element in row),
    ):
        text.set_visible(show_annot)
    # Upload matrix on Neptune
    workflow.run[f"evaluation/{group}/confusion_matrix"].upload(f)


def metric_batch_mixing(workflow, batch_list, group, enc, batches):
    logger.debug("Save batch mixing metrics")
    if (
        len(
            np.unique(
                np.asarray(batch_list[group].argmax(axis=1))
            )
        )
        >= 2
    ):  # If there are more than 2 batches in this group
        for metric in workflow.batch_metrics_list:
            workflow.run[f"evaluation/{group}/{metric}"] = (
                workflow.batch_metrics_list[metric](enc, batches)
            )
            type_batchlist = type(workflow.batch_metrics_list[metric](enc, batches))
            logger.debug(f"Printing type of batch_metrics_list {type_batchlist}")


def metric_classification(workflow, y_pred, y_true, group, sizes):
    logger.debug("Save classification metrics")
    for metric in workflow.pred_metrics_list:
        workflow.run[f"evaluation/{group}/{metric}"] = (
            workflow.pred_metrics_list[metric](y_true, y_pred)
        )

    for metric in workflow.pred_metrics_list_balanced:
        workflow.run[f"evaluation/{group}/{metric}"] = (
            workflow.pred_metrics_list_balanced[metric](
                y_true, y_pred
            )
        )

    # Metrics by size of ct
    logger.debug("Save classification metrics by size of cell type")
    for s in sizes:
        idx_s = np.isin(
            y_true, sizes[s]
        )  # Boolean array, no issue to index y_pred
        y_true_sub = y_true[idx_s]
        y_pred_sub = y_pred[idx_s]
        for metric in workflow.pred_metrics_list:
            workflow.run[f"evaluation/{group}/{s}/{metric}"] = (
                nan_to_0(
                    workflow.pred_metrics_list[metric](
                        y_true_sub, y_pred_sub
                    )
                )
            )

        for metric in workflow.pred_metrics_list_balanced:
            workflow.run[f"evaluation/{group}/{s}/{metric}"] = (
                nan_to_0(
                    workflow.pred_metrics_list_balanced[metric](
                        y_true_sub, y_pred_sub
                    )
                )
            )

def metric_clustering(workflow, y_pred, group, enc):
    logger.debug("Save clustering metrics")
    for metric in workflow.clustering_metrics_list:
        workflow.run[f"evaluation/{group}/{metric}"] = (
            workflow.clustering_metrics_list[metric](enc, y_pred)
        )


def save_results(workflow, y_pred, y_true, adata_list, group, save_dir, split, enc):
    y_pred_df = pd.DataFrame(
        {"pred": y_pred, "true": y_true, "split": split},
        index=adata_list[group].obs_names,
    )
    split = pd.DataFrame(
        split, index=adata_list[group].obs_names
    )
    np.save(os.path.join(save_dir,f"latent_space_{group}.npy"), enc.numpy())
    y_pred_df.to_csv(os.path.join(save_dir,f"predictions_{group}.csv"))
    split.to_csv(os.path.join(save_dir,f"split_{group}.csv"))
    workflow.run[
        f"evaluation/{group}/latent_space"
    ].track_files(os.path.join(save_dir,f"latent_space_{group}.npy"))
    workflow.run[
        f"evaluation/{group}/predictions"
    ].track_files(os.path.join(save_dir,f"predictions_{group}.csv"))

    # Saving umap representation
    pred_adata = sc.AnnData(
        X=adata_list[group].X,
        obs=adata_list[group].obs,
        var=adata_list[group].var,
    )
    pred_adata.obs[f"{workflow.class_key}_pred"] = y_pred_df[
        "pred"
    ]
    pred_adata.obsm["latent_space"] = enc.numpy()
    sc.pp.neighbors(pred_adata, use_rep="latent_space")
    sc.tl.umap(pred_adata)
    np.save(
        os.path.join(save_dir,f"umap_{group}.npy"),
        pred_adata.obsm["X_umap"],
    )
    workflow.run[f"evaluation/{group}/umap"].track_files(
        os.path.join(save_dir,f"umap_{group}.npy")
    )
    sc.set_figure_params(figsize=(15, 10), dpi=300)
    fig_class = sc.pl.umap(
        pred_adata,
        color=f"true_{workflow.class_key}",
        size=10,
        return_fig=True,
    )
    fig_pred = sc.pl.umap(
        pred_adata,
        color=f"{workflow.class_key}_pred",
        size=10,
        return_fig=True,
    )
    fig_batch = sc.pl.umap(
        pred_adata,
        color=workflow.batch_key,
        size=10,
        return_fig=True,
    )
    fig_split = sc.pl.umap(
        pred_adata,
        color="train_split",
        size=10,
        return_fig=True,
    )
    workflow.run[f"evaluation/{group}/true_umap"].upload(
        fig_class
    )
    workflow.run[f"evaluation/{group}/pred_umap"].upload(
        fig_pred
    )
    workflow.run[f"evaluation/{group}/batch_umap"].upload(
        fig_batch
    )
    workflow.run[f"evaluation/{group}/split_umap"].upload(
        fig_split
    )