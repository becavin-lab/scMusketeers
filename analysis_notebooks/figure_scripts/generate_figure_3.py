import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import logging

from .notebook_tools import diff_dict, aestetic_data_name, aestetic_model_name, test_fold_selection

logger = logging.getLogger(__name__)

def produce_fig_3(checkpoint_base_path, working_dir):
    task2_checkpoint_path = checkpoint_base_path.replace('.csv', '_task_2.csv')
    if not os.path.exists(task2_checkpoint_path):
        logger.error(f"Checkpoint file not found at {task2_checkpoint_path}. Run csv_process first!")
        return
        
    logger.info(f"Loading checkpoint from {task2_checkpoint_path}...")
    runs_table_df = pd.read_csv(task2_checkpoint_path)
    
    if runs_table_df.empty:
        logger.error("No valid task 2 runs found in checkpoint. Cannot plot figures.")
        return

    logger.info("Building DataFrames for Figure 3 plots...")
    task_dict_acc = {}
    
    for key in list(diff_dict.keys())[:3]:
        df_acc = pd.DataFrame()
        for ds in diff_dict[key]:
            if ds not in ['tran_2021', 'koenig_2022', 'litvinukova_2020', 'lake_2021']:
                if ds not in test_fold_selection:
                    continue
                expected_test_fold = test_fold_selection[ds]
                ds_df = runs_table_df.query(f"dataset_name == '{ds}' and test_fold_nb == {expected_test_fold}")
                
                for model in ds_df['model'].unique():
                    if model != 'scmap_cluster':
                        sub = ds_df.query(f"model == '{model}'").copy()
                        if not sub.empty:
                            best_val = sub['val_balanced_acc'].max()
                            best_sub = sub[sub['val_balanced_acc'] == best_val].head(1)
                            
                            best_sub['aestetic_data_name'] = aestetic_data_name.get(ds, ds)
                            best_sub['aestetic_model_name'] = aestetic_model_name.get(model, model)
                            df_acc = pd.concat([df_acc, best_sub])
        task_dict_acc[key] = df_acc

    logger.info("Generating Figure 3 Boxplots...")
    
    model_order = [aestetic_model_name.get(m, m) for m in ['scPermut_default', 'scPermut', 'celltypist', 'scmap_cells', 'pca_knn', 'pca_svm', 'uce', 'scanvi', 'harmony_svm']]
    
    figures_dir = os.path.join(working_dir, 'analysis_notebooks', 'figure_review', 'task_2')
    os.makedirs(figures_dir, exist_ok=True)
    
    def generate_row_plot(metric_col, ylabel, output_filename, title="Figure 3"):
        f, axes = plt.subplots(1, 3, figsize=(15, 6))
        titles = ["Cross technology", "Cross assay", "Cross suspension method"]
        
        for key, ax, title_name in zip(list(diff_dict.keys())[:3], axes, titles):
            df_plot = task_dict_acc.get(key, pd.DataFrame())
            if df_plot.empty:
                continue
                
            colors = sns.color_palette(n_colors=len(df_plot['aestetic_data_name'].unique()))
            
            order = [m for m in model_order if m in df_plot['aestetic_model_name'].unique()]
            
            sns.boxplot(data=df_plot, x="aestetic_model_name", y=metric_col, 
                        hue="aestetic_data_name", showfliers=False, order=order, 
                        palette=colors, ax=ax)
            sns.stripplot(data=df_plot, x="aestetic_model_name", y=metric_col, 
                          hue="aestetic_data_name", size=3, color="black", 
                          dodge=True, ax=ax, order=order)
                          
            pairs = []
            baseline_name = 'scMusketeers - default'
            if baseline_name in order:
                for m in order:
                    if m != baseline_name:
                        pairs.append((baseline_name, m))
                        
                if pairs:
                    annot = Annotator(ax, pairs, data=df_plot, x="aestetic_model_name", 
                                      y=metric_col, order=order)
                    try:
                        annot.configure(test='Wilcoxon', text_format='star', loc='inside', 
                                        comparisons_correction="Benjamini-Hochberg")
                        annot.apply_test().annotate()
                    except Exception as e:
                        logger.debug(f"Stats annotation err on {title_name} for {metric_col}: {e}")
            
            ax.set_title(title_name)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Model")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
        f.suptitle(title)
        plt.tight_layout()
        output_path = os.path.join(figures_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved {output_path}")
        plt.close()

    generate_row_plot(
        metric_col="test_balanced_acc",
        ylabel="Balanced accuracy", 
        output_filename="Figure_3_Balanced_Accuracy.png",
        title="Figure 3 - Balanced Accuracy"
    )
    
    generate_row_plot(
        metric_col="full_batch_mixing_entropy",
        ylabel="Batch mixing entropy", 
        output_filename="Figure_3_Entropy.png",
        title="Figure 3 - Entropy"
    )
