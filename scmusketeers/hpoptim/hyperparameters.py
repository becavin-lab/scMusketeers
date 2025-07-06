import argparse
import os
import sys
import gc
import json
import subprocess
import sys
import time
import logging
import neptune
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import keras
import functools
import scipy
import warnings
import gc
from neptune.utils import stringify_unsupported
import keras
from tensorflow.keras.mixed_precision import set_global_policy

from sklearn.utils import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, adjusted_mutual_info_score,
                             adjusted_rand_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             davies_bouldin_score, f1_score, matthews_corrcoef,
                             normalized_mutual_info_score)

# Import scmusketeers library
sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..arguments.neptune_log import NEPTUNE_INFO
    from ..arguments.ae_param import AE_PARAM
    from ..arguments.class_param import CLASS_PARAM
    from ..arguments.dann_param import DANN_PARAM
    from .dataset import Dataset, load_dataset
    from ..tools import freeze
    from . import metrics
    from ..tools.training_scheme import get_training_scheme
    from ..tools.models import DANN_AE
    from ..tools.permutation import batch_generator_training_permuted
    from ..tools.utils import (check_dir, default_value, nan_to_0,
                               scanpy_to_input, str2bool)
    from ..tools.clust_compute import (balanced_cohen_kappa_score,
                                       balanced_f1_score,
                                       balanced_matthews_corrcoef,
                                       batch_entropy_mixing_score, lisi_avg,
                                       nn_overlap)
except ImportError:
    from scmusketeers.arguments.neptune_log import NEPTUNE_INFO
    from scmusketeers.arguments.ae_param import AE_PARAM
    from scmusketeers.arguments.class_param import CLASS_PARAM
    from scmusketeers.arguments.dann_param import DANN_PARAM
    from scmusketeers.hpoptim.dataset import Dataset, load_dataset
    from scmusketeers.tools import freeze
    from scmusketeers.hpoptim import metrics
    from scmusketeers.tools.training_scheme import get_training_scheme
    from scmusketeers.tools.models import DANN_AE
    from scmusketeers.tools.permutation import batch_generator_training_permuted
    from scmusketeers.tools.utils import (check_dir, default_value, nan_to_0,
                             scanpy_to_input, str2bool)
    from scmusketeers.tools.clust_compute import (balanced_cohen_kappa_score,
                                     balanced_f1_score,
                                     balanced_matthews_corrcoef,
                                     batch_entropy_mixing_score, lisi_avg,
                                     nn_overlap)

logger = logging.getLogger("Sc-Musketeers")
try:
    from ax.service.managed_loop import optimize
except ImportError:
    logger.warning("Tried import scmusketeers.workflow but AX Platform not installed")
    logger.warning("Please consider installing AxPlatform if you want to perform hyperparameters optimization")
    logger.warning("poetry install --with workflow")


# Setup settings
f1_score = functools.partial(f1_score, average="macro")
physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
# Set the global policy to use mixed_float16
#set_global_policy('mixed_float16')

# Suppress the specific Keras UserWarning about non-existent gradients
warnings.filterwarnings(
    'ignore',
    message="Gradients do not exist for variables",
    category=UserWarning,
    module='keras.src.optimizers.base_optimizer' # This targets the specific source of the warning
)
# Suppress the warning about y_pred containing classes not in y_true
warnings.filterwarnings(
    'ignore',
    message="y_pred contains classes not in y_true",
    category=UserWarning
)

logger = logging.getLogger("Sc-Musketeers")


class Workflow:
    def __init__(self, run_file):
        """
        run_file : a dictionary outputed by the function load_runfile
        """
        ### Regroup here all parameters given by run_file
        self.run_file = run_file
        self.ae_param = AE_PARAM(run_file)
        self.class_param = CLASS_PARAM(run_file)
        self.dann_param = DANN_PARAM(run_file)
        self.training_scheme = self.run_file.training_scheme

        # get attributes from run_file
        for par, val_from_runfile in self.run_file.__dict__.items():
            # logger.debug(f"Processing parameter '{par}' from run_file:")
            # Check if 'self' has an attribute with the same name
            if not hasattr(self, par):
                setattr(self, par, val_from_runfile)
                #logger.debug(f"  - '{par}' from run_file DOES NOT exist in 'self'.")
                #logger.debug(f"    - Creating 'self.{par}' with value: {val_from_runfile} (from run_file).")
                
        self.n_perm = 1
        self.semi_sup = False  # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
        # train test split # TODO : Simplify this, or at first only use the case where data is split according to batch

        ### Hyperamateres attributes
        self.hparam_path = self.run_file.hparam_path
        self.hp_params = None
        self.opt_metric = default_value(self.run_file.opt_metric, None)

        ### Regroup here all attributes not in run_file
        self.start_time = time.time()
        self.stop_time = time.time()
        self.run_done = False
        self.predict_done = False
        self.umap_done = False
        self.dataset = None
        self.model = None
        self.predictor = None
        self.training_kwds = {}
        self.network_kwds = {}
        self.clas_loss_fn = None
        self.dann_loss_fn = None
        self.rec_loss_fn = None
        self.num_classes = None
        self.num_batches = None
        
        self.pred_metrics_list = {
            "acc": accuracy_score,
            "mcc": matthews_corrcoef,
            "f1_score": f1_score,
            "KPA": cohen_kappa_score,
            "ARI": adjusted_rand_score,
            "NMI": normalized_mutual_info_score,
            "AMI": adjusted_mutual_info_score,
        }

        self.pred_metrics_list_balanced = {
            "balanced_acc": balanced_accuracy_score,
            "balanced_mcc": balanced_matthews_corrcoef,
            "balanced_f1_score": balanced_f1_score,
            "balanced_KPA": balanced_cohen_kappa_score,
        }

        self.clustering_metrics_list = {  #'clisi' : lisi_avg,
            "db_score": davies_bouldin_score
        }

        self.batch_metrics_list = {
            "batch_mixing_entropy": batch_entropy_mixing_score,
            #'ilisi': lisi_avg
        }
        
        self.metrics = []

        # This is a running average : it keeps the previous values in memory when
        # it's called (ie computes the previous and current values)
        self.mean_loss_fn = keras.metrics.Mean(name="total loss")
        self.mean_clas_loss_fn = keras.metrics.Mean(name="classification loss")
        self.mean_dann_loss_fn = keras.metrics.Mean(name="dann loss")
        self.mean_rec_loss_fn = keras.metrics.Mean(name="reconstruction loss")

    def set_hyperparameters(self, params):

        logger.debug(f"Setting hparams {params}")
        if "use_hvg" in params:
            self.use_hvg = params["use_hvg"]
        if "batch_size" in params:
            self.batch_size = params["batch_size"]
        if "clas_w" in params:
            self.clas_w = params["clas_w"]
        if "dann_w" in params:
            self.dann_w = params["dann_w"]
        if "rec_w" in params:
            self.rec_w = params["rec_w"]
        if "ae_bottleneck_activation" in params:
            self.ae_bottleneck_activation = params["ae_bottleneck_activation"]
        if "clas_loss_name" in params:
            self.clas_loss_name = params["clas_loss_name"]
        if "size_factor" in params:
            self.size_factor = params["size_factor"]
        if "weight_decay" in params:
            self.weight_decay = params["weight_decay"]
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "warmup_epoch" in params:
            self.warmup_epoch = params["warmup_epoch"]
        if "dropout" in params:
            self.dropout = params["dropout"]
        if "layer1" in params:
            self.layer1 = params["layer1"]
        if "layer2" in params:
            self.layer2 = params["layer2"]
        if "bottleneck" in params:
            self.bottleneck = params["bottleneck"]
        if "training_scheme" in params:
            self.training_scheme = params["training_scheme"]
        self.hp_params = params


    def split_train_test(self):
        self.dataset.test_split(
            test_obs=self.test_obs,
            test_index_name=self.test_index_name,
        )

    def split_train_val(self):
        logger.info("Process train, test, val datasets")
        self.dataset.train_split(
            mode=self.mode,
            pct_split=self.pct_split,
            obs_key=self.run_file.obs_key,
            n_keep=self.n_keep,
            keep_obs=self.keep_obs,
            split_strategy=self.split_strategy,
            obs_subsample=self.obs_subsample,
            train_test_random_seed=self.train_test_random_seed,
        )

        logger.info("Dataset split performed")
        self.dataset.create_inputs()

    def process_dataset(self):
        # Loading dataset
        logger.info(f"Load dataset {self.run_file.ref_path}")
        adata = load_dataset(
            ref_path=self.run_file.ref_path,
            query_path='',class_key='',unlabeled_category='')

        self.dataset = Dataset(
            adata=adata,
            class_key=self.run_file.class_key,
            batch_key=self.run_file.batch_key,
            filter_min_counts=self.run_file.filter_min_counts,
            normalize_size_factors=self.run_file.normalize_size_factors,
            size_factor=self.size_factor,
            scale_input=self.scale_input,
            logtrans_input=self.logtrans_input,
            use_hvg=self.use_hvg,
            unlabeled_category=self.run_file.unlabeled_category,
            test_split_key=self.run_file.test_split_key,
            train_test_random_seed=self.train_test_random_seed
        )
        
        if not "X_pca" in self.dataset.adata.obsm:
            logger.debug("Did not find existing PCA, computing it")
            sc.tl.pca(self.dataset.adata)
            self.dataset.adata.obsm["X_pca"] = np.asarray(
                self.dataset.adata.obsm["X_pca"]
            )
        # Processing dataset. Splitting train/test.
        self.dataset.normalize()

    def train_val_split_yo(self):
        self.dataset.train_val_split()
        self.dataset.create_inputs()


    def make_experiment(self):
        logger.info("##-- Create scmusketeers model and the train/test/val datasets:")
        
        logger.info("Setup X,Y")
        adata_list = {
            "full": self.dataset.adata,
            "train": self.dataset.adata_train,
            "val": self.dataset.adata_val,
            "test": self.dataset.adata_test,
        }

        X_list = {
            "full": self.dataset.X,
            "train": self.dataset.X_train,
            "val": self.dataset.X_val,
            "test": self.dataset.X_test,
        }

        y_nooh_list = {
            "full": self.dataset.y,
            "train": self.dataset.y_train,
            "val": self.dataset.y_val,
            "test": self.dataset.y_test,
        }

        y_list = {
            "full": self.dataset.y_one_hot,
            "train": self.dataset.y_train_one_hot,
            "val": self.dataset.y_val_one_hot,
            "test": self.dataset.y_test_one_hot,
        }

        batch_list = {
            "full": self.dataset.batch_one_hot,
            "train": self.dataset.batch_train_one_hot,
            "val": self.dataset.batch_val_one_hot,
            "test": self.dataset.batch_test_one_hot,
        }

        X_pca_list = {
            "full": self.dataset.adata.obsm["X_pca"],
            "train": self.dataset.adata_train.obsm["X_pca"],
            "val": self.dataset.adata_val.obsm["X_pca"],
            "test": self.dataset.adata_test.obsm["X_pca"],
        }

        # Create pesudo labels
        logger.info("Create pseudo-labels pseudo_y_list")
        knn_cl = KNeighborsClassifier(n_neighbors=5)
        knn_cl.fit(X_pca_list["train"], y_nooh_list["train"])

        pseudo_y_val = pd.Series(
            knn_cl.predict(X_pca_list["val"]),
            index=adata_list["val"].obs_names,
        )
        pseudo_y_test = pd.Series(
            knn_cl.predict(X_pca_list["test"]),
            index=adata_list["test"].obs_names,
        )

        pseudo_y_full = pd.concat(
            [pseudo_y_val, pseudo_y_test, y_nooh_list["train"]]
        )
        pseudo_y_full = pseudo_y_full[
            adata_list["full"].obs_names
        ]  # reordering cells in the right order

        pseudo_y_list = {
            "full": self.dataset.ohe_celltype.transform(
                np.array(pseudo_y_full).reshape(-1, 1)
            )
            .astype(float)
            .todense(),
            "train": y_list["train"],
            "val": self.dataset.ohe_celltype.transform(
                np.array(pseudo_y_val).reshape(-1, 1)
            )
            .astype(float)
            .todense(),
            "test": self.dataset.ohe_celltype.transform(
                np.array(pseudo_y_test).reshape(-1, 1)
            )
            .astype(float)
            .todense(),
        }

        self.num_classes = len(np.unique(self.dataset.y_train))
        self.num_batches = len(np.unique(self.dataset.batch))

        # Setup model layers param
        logger.info("Setup model settings")
        if self.layer1:
            self.ae_param.ae_hidden_size = [
                self.layer1,
                self.layer2,
                self.bottleneck,
                self.layer2,
                self.layer1,
            ]

        if self.dropout:
            (
                self.dann_param.dann_hidden_dropout,
                self.class_param.class_hidden_dropout,
                self.ae_param.ae_hidden_dropout,
            ) = (self.dropout, self.dropout, self.dropout)
        # Correct size of layers depending on the number of classes and
        # on the bottleneck size
        bottleneck_size = int(
            self.ae_param.ae_hidden_size[
                int(len(self.run_file.ae_hidden_size) / 2)
            ]
        )
        self.class_param.class_hidden_size = default_value(
            self.class_param.class_hidden_size,
            (bottleneck_size + self.num_classes) / 2,
        )  # default value [(bottleneck_size + num_classes)/2]
        self.dann_param.dann_hidden_size = default_value(
            self.dann_param.dann_hidden_size,
            (bottleneck_size + self.num_batches) / 2,
        )  # default value [(bottleneck_size + num_batches)/2]

        # Creation of model
        self.dann_ae = DANN_AE(
            ae_hidden_size=self.ae_param.ae_hidden_size,
            ae_hidden_dropout=self.ae_param.ae_hidden_dropout,
            ae_activation=self.ae_param.ae_activation,
            ae_output_activation=self.ae_param.ae_output_activation,
            ae_bottleneck_activation=self.ae_param.ae_bottleneck_activation,
            ae_init=self.ae_param.ae_init,
            ae_batchnorm=self.ae_param.ae_batchnorm,
            ae_l1_enc_coef=self.ae_param.ae_l1_enc_coef,
            ae_l2_enc_coef=self.ae_param.ae_l2_enc_coef,
            num_classes=self.num_classes,
            class_hidden_size=self.class_param.class_hidden_size,
            class_hidden_dropout=self.class_param.class_hidden_dropout,
            class_batchnorm=self.class_param.class_batchnorm,
            class_activation=self.class_param.class_activation,
            class_output_activation=self.class_param.class_output_activation,
            num_batches=self.num_batches,
            dann_hidden_size=self.dann_param.dann_hidden_size,
            dann_hidden_dropout=self.dann_param.dann_hidden_dropout,
            dann_batchnorm=self.dann_param.dann_batchnorm,
            dann_activation=self.dann_param.dann_activation,
            dann_output_activation=self.dann_param.dann_output_activation,
        )

        logger.info("Setup optimizer")
        self.optimizer = self.get_optimizer(
            self.learning_rate, self.weight_decay, self.optimizer_type
        )
        logger.info("Setup losses")
        self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn = (
            self.get_losses(y_list)
        )  # redundant
        
        logger.info("Setup training scheme")
        training_scheme = get_training_scheme(self.training_scheme, self.run_file)
        
        # Training
        logger.info(f"##-- Run model training")
        start_time = time.time()
        history = self.train_scheme(
            training_scheme=training_scheme,
            verbose=False,
            adata_list=adata_list,
            X_list=X_list,
            y_list=y_list,
            batch_list=batch_list,
            pseudo_y_list=pseudo_y_list,
            #  optimizer= self.optimizer, # not an **loop_param since it resets between strategies
            clas_loss_fn=self.clas_loss_fn,
            dann_loss_fn=self.dann_loss_fn,
            rec_loss_fn=self.rec_loss_fn,
        )
        stop_time = time.time()
        logger.info("Training complete")

        logger.info("##-- Performing model prediction")
        with tf.device('/CPU:0'):
            if self.log_neptune:
                # TODO also make it on gpu with smaller batch size
                self.run_neptune["evaluation/training_time"] = stop_time - start_time
                neptune_run_id = self.run_neptune["sys/id"].fetch()
                save_dir = os.path.join(self.run_file.out_dir,str(neptune_run_id))
                logger.info(f"Save results to: {save_dir}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Setting model inference
                y_true_full = adata_list["full"].obs[f"true_{self.class_key}"]
                ct_prop = (
                    pd.Series(y_true_full).value_counts()
                    / pd.Series(y_true_full).value_counts().sum()
                )
                sizes = {
                    "xxsmall": list(ct_prop[ct_prop < 0.001].index),
                    "small": list(
                        ct_prop[(ct_prop >= 0.001) & (ct_prop < 0.01)].index
                    ),
                    "medium": list(
                        ct_prop[(ct_prop >= 0.01) & (ct_prop < 0.1)].index
                    ),
                    "large": list(ct_prop[ct_prop >= 0.1].index),
                }
                for group in ["full", "train", "val", "test"]:
                    logger.info(f"Prediction for dataset - {group}")
                    input_tensor = {
                        k: tf.convert_to_tensor(v)
                        for k, v in scanpy_to_input(
                            adata_list[group], ["size_factors"]
                        ).items()
                    }
                    enc, clas, dann, rec = self.dann_ae(
                        input_tensor, training=False
                    ).values()  # Model predict

                    if (
                        group == "full"
                    ):  # saving full predictions as probability output from the classifier
                        logger.debug(f"Saving predicted matrix and embedding - {group}")
                        y_pred_proba = pd.DataFrame(
                            np.asarray(clas),
                            index=adata_list["full"].obs_names,
                            columns=self.dataset.ohe_celltype.categories_[0],
                        )
                        y_pred_proba.to_csv(os.path.join(save_dir,"y_pred_proba_full.csv"))
                        self.run_neptune[
                            f"evaluation/{group}/y_pred_proba_full"
                        ].track_files(os.path.join(save_dir,"y_pred_proba_full.csv"))

                    # Create predicted cell types
                    clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
                    y_pred = self.dataset.ohe_celltype.inverse_transform(
                        clas
                    ).reshape(
                        -1,
                    )
                    y_true = adata_list[group].obs[f"true_{self.class_key}"]
                    batches = np.asarray(
                        batch_list[group].argmax(axis=1)
                    ).reshape(
                        -1,
                    )
                    split = adata_list[group].obs["train_split"]

                    # Saving confusion matrices
                    metrics.metric_confusion_matrix(self, y_pred, y_true, group, save_dir)

                    # Computing batch mixing metrics
                    metrics.metric_batch_mixing(self, batch_list, group, enc, batches)

                    # Save classification metrics
                    metrics.metric_classification(self, y_pred, y_true, group, sizes)

                    # save clustering metrics
                    metrics.metric_clustering(self, y_pred, group, enc)

                    logger.debug(f"Save all matrices and figures - {group}")
                    if group == "full":
                        metrics.save_results(self, y_pred, y_true, adata_list, group, save_dir, split, enc)

        if self.opt_metric:
            split, metric = self.opt_metric.split("-")
            self.run_neptune.wait()
            opt_metric = self.run_neptune[f"evaluation/{split}/{metric}"].fetch()
            logger.info(f"optimal metric:{opt_metric}")
        else:
            opt_metric = None
        return opt_metric
    
    def generate_pseudolabels_batched(self, adata, batch_size=None):
        """
        Generate pseudolabels for the full dataset using batch processing to avoid memory issues.
        
        Args:
            adata: AnnData object containing the full dataset
            batch_size: Batch size for processing. If None, uses self.batch_size
        
        Returns:
            numpy array of pseudolabels (one-hot encoded)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        n_obs = adata.n_obs
        steps = n_obs // batch_size + (1 if n_obs % batch_size > 0 else 0)
        
        all_predictions = []
        
        logger.debug(f"Generating pseudolabels for {n_obs} cells in {steps} batches")
        
        for i in range(steps):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_obs)
            
            # Create batch input
            batch_adata = adata[start_idx:end_idx].copy()
            
            # Convert to input format
            input_tensor = {
                k: tf.convert_to_tensor(v)
                for k, v in scanpy_to_input(batch_adata, ["size_factors"]).items()
            }
            
            # Model prediction for this batch
            with tf.device('/CPU:0'):  # Use CPU to save GPU memory
                enc, clas, dann, rec = self.dann_ae(input_tensor, training=False).values()
            
            # Store predictions
            all_predictions.append(clas.numpy())
            
            # Clean up tensors to free memory
            del input_tensor, enc, clas, dann, rec
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Concatenate all predictions
        full_predictions = np.concatenate(all_predictions, axis=0)
        
        # Convert to one-hot encoded pseudolabels
        pseudo_labels = np.eye(full_predictions.shape[1])[np.argmax(full_predictions, axis=1)]
        
        logger.debug(f"Generated pseudolabels shape: {pseudo_labels.shape}")
        return pseudo_labels
        

    def train_scheme(self, training_scheme, verbose=True, **loop_params):
        """
        training scheme : dictionnary explaining the succession of strategies to use as keys with the corresponding number of epochs and use_perm as values.
                        x :  training_scheme_3 = {"warmup_dann" : (10, False), "full_model":(10, False)}
        """
        history = {"train": {}, "val": {}}  # initialize history
        for group in history.keys():
            history[group] = {
                "total_loss": [],
                "clas_loss": [],
                "dann_loss": [],
                "rec_loss": [],
            }
            for m in self.pred_metrics_list:
                history[group][m] = []
            for m in self.pred_metrics_list_balanced:
                history[group][m] = []

        
        # Run scMusketeers training_scheme one after the other
        total_epochs = np.sum([n_epochs for _, n_epochs, _ in training_scheme])
        running_epoch = 0
        scheme_index = 0
        for strategy, n_epochs, use_perm in training_scheme:
            running_epoch = self.run_single_scheme(strategy, history, use_perm, loop_params, scheme_index, running_epoch, n_epochs, total_epochs)

        if self.run_file.log_neptune:
            self.run_neptune[f"training/{group}/total_epochs"] = running_epoch
        return history


    def run_single_scheme(self, strategy, history, use_perm, loop_params, scheme_index, running_epoch, n_epochs, total_epochs):
        """
        Run the training of one scMusketeers scheme given by the varialbe strategy
        full_model,
        classifier_branch
        permutation_only
        encoder_classifier
        warmup_dann_pseudolabels
        full_model_pseudolabels
        warmup_dann_semisup
        """
        optimizer = self.get_optimizer(
            self.learning_rate, self.weight_decay, self.optimizer_type
        )  # resetting optimizer state when switching strategy
        logger.info(
            f"##-- {strategy.upper()} - Step {scheme_index}, running {strategy} strategy with permutation = {use_perm} for {n_epochs} epochs"
        )
        time_in = time.time()
        scheme_index += 1

        # Early stopping setup 
        if strategy in [
            "full_model",
            "classifier_branch", 
            "permutation_only",
            "encoder_classifier",
        ]:
            wait = 0
            best_epoch = 0
            patience = 30
            min_delta = 0
            
            min_epochs = min(10, n_epochs)
            logger.debug(f"Early stopping active with a warm-up of {min_epochs} epochs.")


            if strategy == "permutation_only":
                monitored = "rec_loss"
                es_best = np.inf
            else:
                split, metric = self.opt_metric.split("-")
                monitored = metric
                es_best = -np.inf

        memory = {}

        # Early stopping for those strategies only
        if strategy in [
            "warmup_dann_pseudolabels",
            "full_model_pseudolabels",
        ]:
            logger.info("Generating pseudolabels using batch processing...")
            
            # Generate pseudolabels in batches
            pseudo_full = self.generate_pseudolabels_batched(
                loop_params["adata_list"]["full"], 
                batch_size=self.batch_size
            )
            
            # Replace train predictions with true labels
            train_mask = loop_params["adata_list"]["full"].obs["train_split"] == "train"
            pseudo_full[train_mask, :] = loop_params["pseudo_y_list"]["train"]
            
            # Update loop_params
            loop_params["pseudo_y_list"]["full"] = pseudo_full
            
            # Extract pseudolabels for val and test groups
            for group in ["val", "test"]:
                group_mask = loop_params["adata_list"]["full"].obs["train_split"] == group
                loop_params["pseudo_y_list"][group] = pseudo_full[group_mask, :]
                
            logger.info("Pseudolabel generation completed")
            
        elif strategy in ["warmup_dann_semisup"]:
            memory = {}
            memory["pseudo_full"] = loop_params["pseudo_y_list"]["full"]
            
            # Generate pseudolabels in batches first
            pseudo_full = self.generate_pseudolabels_batched(
                loop_params["adata_list"]["full"], 
                batch_size=self.batch_size
            )
            
            # Set val and test to unlabeled_category
            for group in ["val", "test"]:
                group_mask = loop_params["adata_list"]["full"].obs["train_split"] == group
                loop_params["pseudo_y_list"]["full"][group_mask, :] = self.run_file.unlabeled_category
                loop_params["pseudo_y_list"][group] = pseudo_full[group_mask, :]
                
        else:
            if memory:
                # Reset to previous known pseudolabels
                for group in ["val", "test", "full"]:
                    loop_params["pseudo_y_list"][group] = memory[group]
                memory = {}

        # Rest of the function remains the same...
        freeze.unfreeze_all(self.dann_ae)
        
        # trainable_unfrozen_variables = [v for v in ae.trainable_variables if v.trainable] # Should match your '6' count
        # logger.debug(f"After unfrozen trainable variables: {len(trainable_unfrozen_variables)}")
        # for i, var in enumerate(trainable_unfrozen_variables):
        #     logger.debug(f"  After Unfrozen Var {i}: {var.name}, Shape: {var.shape}") 
       
        if strategy == "full_model":
            group = "train"
        elif strategy == "full_model_pseudolabels":
            group = "full"
        elif strategy == "encoder_classifier":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "all_but_classifier")  # training only
            freeze.freeze_layers(layers_to_freeze)
        elif strategy in [
            "warmup_dann",
            "warmup_dann_pseudolabels",
            "warmup_dann_semisup",
        ]:
            group = "full"  # semi-supervised setting
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "warmup_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "warmup_dann_train":
            group = "train"  # semi-supervised setting
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "warmup_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "warmup_dann_no_rec":
            group = "full"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "all_but_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "dann_with_ae":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "warmup_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "classifier_branch":
            group = "train"
            layers_to_freeze = freeze.freeze_block(
                self.dann_ae, "all_but_classifier_branch"
            )  # training only classifier branch
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "permutation_only":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "all_but_autoencoder")
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "no_dann":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "freeze_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif strategy == "no_decoder":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "freeze_dec")
            freeze.freeze_layers(layers_to_freeze)

        logger.info(f"Use permutation strategy? use_perm = {use_perm}")

        for epoch in range(1, n_epochs + 1):
            running_epoch += 1
            logger.info(
                f"Epoch {running_epoch}/{total_epochs}, Current strat Epoch {epoch}/{n_epochs}"
            )
            history = self.training_loop(
                history=history,
                group=group,
                training_strategy=strategy,
                use_perm=use_perm,
                optimizer=optimizer,
                **loop_params,
            )

            if self.run_file.log_neptune:
                for group in history:
                    for par, value in history[group].items():
                        if len(value) > 0:
                            self.run_neptune[
                                f"training/{group}/{par}"
                            ].append(value[-1])
                        if physical_devices:
                            self.run_neptune[
                                "training/train/tf_GPU_memory"
                            ].append(
                                tf.config.experimental.get_memory_info(
                                    "GPU:0"
                                )["current"]
                                / 1e6
                            )
            if strategy in [
                "full_model",
                "classifier_branch",
                "permutation_only",
                "encoder_classifier",
            ]:
                monitored_value = history["val"][monitored][-1]

                # ALWAYS CHECK FOR IMPROVEMENT
                # This part runs on every epoch to ensure we capture the true best model.
                has_improved = False
                if "loss" in monitored:
                    if monitored_value < es_best - min_delta:
                        has_improved = True
                else:
                    if monitored_value > es_best + min_delta:
                        has_improved = True

                if has_improved:
                    logger.debug(f"New best score at epoch {epoch}: {monitored_value:.4f}")
                    best_epoch = epoch
                    es_best = monitored_value
                    wait = 0  # Reset patience since we found a better model
                    best_model = self.dann_ae.get_weights()
                else:
                    # INCREMENT WAIT COUNTER ONLY AFTER THE WARM-UP
                    if epoch > min_epochs:
                        wait += 1
                        # Early stopping
                
                # CHECK FOR STOPPING CONDITION ONLY AFTER THE WARM-UP
                if epoch > min_epochs and wait >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. Restoring best model from epoch {best_epoch} with score {es_best:.4f}."
                    )
                    self.dann_ae.set_weights(best_model)
                    break  # Exit the loop for this training scheme


        time_out = time.time()
        logger.info(f"Strategy {strategy} duration : {time_out - time_in} s")

        tf.keras.backend.clear_session()
        gc.collect()

        return running_epoch


    def training_loop(
        self,
        history,
        group,
        adata_list,
        X_list,
        y_list,
        pseudo_y_list,
        batch_list,
        optimizer,
        clas_loss_fn,
        dann_loss_fn,
        rec_loss_fn,
        use_perm=True,
        training_strategy="full_model",
        verbose=False,
    ):
        """
        A consolidated training loop function that covers common logic used in different training strategies.

        training_strategy : one of ["full", "warmup_dann", "warmup_dann_no_rec", "classifier_branch", "permutation_only"]
            - full_model : trains the whole model, optimizing the 3 losses (reconstruction, classification, anti batch discrimination ) at once
            - warmup_dann : trains the dann, encoder and decoder with reconstruction (no permutation because unsupervised), maximizing the dann loss and minimizing the reconstruction loss
            - warmup_dann_no_rec : trains the dann and encoder without reconstruction, maximizing the dann loss only.
            - dann_with_ae : same as warmup dann but with permutation. Separated in two strategies because this one is supervised
            - classifier_branch : trains the classifier branch only, without the encoder. Use to fine tune the classifier once training is over
            - permutation_only : trains the autoencoder with permutations, optimizing the reconstruction loss without the classifier
        use_perm : True by default except form "warmup_dann" training strategy. Note that for training strategies that don't involve the reconstruction, this parameter has no impact on training
        """
        batch_generator = batch_generator_training_permuted(
            X=X_list[group],
            y=pseudo_y_list[
                group
            ],  # We use pseudo labels for val and test. y_train are true labels
            batch_ID=batch_list[group],
            sf=adata_list[group].obs["size_factors"],
            ret_input_only=False,
            batch_size=self.batch_size,
            n_perm=1,
            unlabeled_category=self.run_file.unlabeled_category,  # Those cells are matched with themselves during AE training
            use_perm=use_perm,
        )
        n_obs = adata_list[group].n_obs
        steps = n_obs // self.batch_size + 1
        n_steps = steps
        n_samples = 0

        self.mean_loss_fn.reset_state()
        self.mean_clas_loss_fn.reset_state()
        self.mean_dann_loss_fn.reset_state()
        self.mean_rec_loss_fn.reset_state()

        # Batch steps
        for step in range(1, n_steps + 1):
            # self.tr = tracker.SummaryTracker()
            self.batch_step(
                step,
                clas_loss_fn,
                dann_loss_fn,
                rec_loss_fn,
                batch_generator,
                training_strategy,
                optimizer,
                n_samples,
                n_obs,
            )

        history = self.evaluation_pass(
            history,
            adata_list,
            X_list,
            y_list,
            batch_list,
            clas_loss_fn,
            dann_loss_fn,
            rec_loss_fn,
        )
        return history

    
    def batch_step(
        self,
        step,
        clas_loss_fn,
        dann_loss_fn,
        rec_loss_fn,
        batch_generator,
        training_strategy,
        optimizer,
        n_samples,
        n_obs
    ):
        if self.run_file.log_neptune:
            self.run_neptune["training/train/tf_GPU_memory_step"].append(
                tf.config.experimental.get_memory_info("GPU:0")["current"]
                / 1e6
            )
            self.run_neptune["training/train/step"].append(step)

        input_batch, output_batch = next(batch_generator)
        # X_batch, sf_batch = input_batch.values()
        clas_batch, dann_batch, rec_batch = output_batch.values()
        
        with tf.GradientTape() as tape:
            # Forward pass
            # logger.debug("Convert data to tensor")
            input_batch_new = {
                k: tf.convert_to_tensor(v.toarray() if isinstance(v, scipy.sparse.spmatrix) else v, dtype=tf.float32)
                for k, v in input_batch.items()
            }

            enc, clas, dann, rec = self.dann_ae(input_batch, training=True).values()

            # Computing loss
            # logger.debug("Calculate losses")
            clas_loss = tf.reduce_mean(clas_loss_fn(clas_batch, clas))
            dann_loss = tf.reduce_mean(dann_loss_fn(dann_batch, dann))
            rec_loss = tf.reduce_mean(rec_loss_fn(rec_batch, rec))
            
            if training_strategy in [
                "full_model",
                "full_model_pseudolabels",
            ]:
                loss = tf.add_n(
                    [self.clas_w * clas_loss]
                    + [self.dann_w * dann_loss]
                    + [self.rec_w * rec_loss]
                    + self.dann_ae.losses
                )
            elif training_strategy in [
                "warmup_dann",
                "warmup_dann_pseudolabels",
                "warmup_dann_train",
                "warmup_dann_semisup",
            ]:
                loss = tf.add_n(
                    [self.dann_w * dann_loss]
                    + [self.rec_w * rec_loss]
                    + self.dann_ae.losses
                )
            elif training_strategy == "warmup_dann_no_rec":
                loss = tf.add_n([self.dann_w * dann_loss] + self.dann_ae.losses)
            elif training_strategy == "dann_with_ae":
                loss = tf.add_n(
                    [self.dann_w * dann_loss]
                    + [self.rec_w * rec_loss]
                    + self.dann_ae.losses
                )
            elif training_strategy == "classifier_branch":
                loss = tf.add_n([self.clas_w * clas_loss])
            elif training_strategy == "encoder_classifier":
                loss = tf.add_n([self.clas_w * clas_loss] + self.dann_ae.losses)
            elif training_strategy == "permutation_only":
                loss = tf.add_n([self.rec_w * rec_loss] + self.dann_ae.losses)
            elif training_strategy == "no_dann":
                loss = tf.add_n(
                    [self.rec_w * rec_loss]
                    + [self.clas_w * clas_loss]
                    + self.dann_ae.losses
                )
            elif training_strategy == "no_decoder":
                loss = tf.add_n(
                    [self.dann_w * dann_loss]
                    + [self.clas_w * clas_loss]
                    + self.dann_ae.losses
                )
        n_samples += enc.shape[0]
        
        # Backpropagation
        # --- Main Gradients for the Total Loss ---
        # logger.debug("Decipher gradients")
        gradients = tape.gradient(loss, self.dann_ae.trainable_variables)

        # logger.debug("\n--- Gradients from TOTAL LOSS ---")
        # for grad, var in zip(gradients, ae.trainable_variables):
        #     logger.debug(f"  - Var: {var.name}, Grad: {'None' if grad is None else 'OK (shape: '+str(grad.shape)+')'}")
        
        del tape # Don't forget to delete persistent tape
    
        # logger.debug("Back propagation")
        optimizer.apply_gradients(zip(gradients, self.dann_ae.trainable_variables))

        self.mean_loss_fn(loss.__float__())
        self.mean_clas_loss_fn(clas_loss.__float__())
        self.mean_dann_loss_fn(dann_loss.__float__())
        self.mean_rec_loss_fn(rec_loss.__float__())

        # print_status_bar(
        #     n_samples,
        #     n_obs,
        #     [
        #         self.mean_loss_fn,
        #         self.mean_clas_loss_fn,
        #         self.mean_dann_loss_fn,
        #         self.mean_rec_loss_fn,
        #     ],
        #     self.metrics,
        # )


    def evaluation_pass(
        self,
        history,
        adata_list,
        X_list,
        y_list,
        batch_list,
        clas_loss_fn,
        dann_loss_fn,
        rec_loss_fn,
    ):
        """
        evaluate model and logs metrics. Depending on "on parameter, computes it on train and val or train,val and test.

        on : "epoch_end" to evaluate on train and val, "training_end" to evaluate on train, val and "test".
        """
        for group in ["train", "val"]:  # evaluation round
            n_obs = X_list[group].shape[0]
            steps = n_obs // self.batch_size + (1 if n_obs % self.batch_size > 0 else 0)
            
            all_clas = []
            batch_rec_losses = []
            all_dann = []
            
            # Create a simple generator for evaluation
            for i in range(steps):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_obs)
                
                # 1. Prepare the count data for the batch
                batch_X = X_list[group][start_idx:end_idx]
                if scipy.sparse.issparse(batch_X):
                    batch_X = batch_X.toarray()
                
                # 2. Prepare the size factors for the batch
                batch_sf = adata_list[group].obs["size_factors"].iloc[start_idx:end_idx].values
                
                # 3. Build the input dictionary with the CORRECT keys: 'counts' and 'size_factors'
                inp = {
                    'counts': tf.convert_to_tensor(batch_X),
                    'size_factors': tf.convert_to_tensor(batch_sf)
                }

                _, clas_batch, dann_batch, rec_batch = self.dann_ae(inp, training=False).values()
                all_clas.append(clas_batch.numpy())
                all_dann.append(dann_batch.numpy())
                # Calculate reconstruction loss for this batch and append the scalar value
                rec_loss_for_batch = tf.reduce_mean(rec_loss_fn(batch_X, rec_batch))
                batch_rec_losses.append(rec_loss_for_batch.numpy())
            

            # Concatenate results from all batches
            clas = np.concatenate(all_clas, axis=0)
            dann = np.concatenate(all_dann, axis=0)

            # # Calc val for all batches
            # inp = scanpy_to_input(adata_list[group], ["size_factors"])
            # inp = {k: tf.convert_to_tensor(v) for k, v in inp.items()}
            # _, clas, dann, rec = self.dann_ae(inp, training=False).values()

            #         return _, clas, dann, rec
            clas_loss = tf.reduce_mean(
                clas_loss_fn(y_list[group], clas)
            ).numpy()
            history[group]["clas_loss"] += [clas_loss]
            dann_loss = tf.reduce_mean(
                dann_loss_fn(batch_list[group], dann)
            ).numpy()
            history[group]["dann_loss"] += [dann_loss]
            rec_loss = np.mean(batch_rec_losses)

            history[group]["rec_loss"] += [rec_loss]
            history[group]["total_loss"] += [
                self.clas_w * clas_loss
                + self.dann_w * dann_loss
                + self.rec_w * rec_loss
                + np.sum(self.dann_ae.losses)
            ]  # using numpy to prevent memory leaks
            # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

            clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
            for (
                metric
            ) in self.pred_metrics_list:  # only classification metrics ATM
                history[group][metric] += [
                    self.pred_metrics_list[metric](
                        np.asarray(y_list[group].argmax(axis=1)).reshape(
                            -1,
                        ),
                        clas.argmax(axis=1),
                    )
                ]  # y_list are onehot encoded
            for (
                metric
            ) in (
                self.pred_metrics_list_balanced
            ):  # only classification metrics ATM
                history[group][metric] += [
                    self.pred_metrics_list_balanced[metric](
                        np.asarray(y_list[group].argmax(axis=1)).reshape(
                            -1,
                        ),
                        clas.argmax(axis=1),
                    )
                ]  # y_list are onehot encoded
        del inp
        return history


    def get_losses(self, y_list):
        if self.rec_loss_name == "MSE":
            self.rec_loss_fn = tf.keras.losses.MSE
        else:
            print(self.rec_loss_name + " loss not supported for rec")

        if self.balance_classes:
            y_integers = np.argmax(np.asarray(y_list["train"]), axis=1)
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_integers),
                y=y_integers,
            )

        if self.clas_loss_name == "categorical_crossentropy":
            self.clas_loss_fn = tf.keras.losses.categorical_crossentropy
        elif self.clas_loss_name == "categorical_focal_crossentropy":
            self.clas_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=class_weights, gamma=3
            )
        else:
            logger.debug(self.clas_loss_name + " loss not supported for classif")

        if self.dann_loss_name == "categorical_crossentropy":
            self.dann_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            print(self.dann_loss_name + " loss not supported for dann")
        return self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn


    def get_optimizer(
        self, learning_rate, weight_decay, optimizer_type, momentum=0.9
    ):
        """
        This function takes a  learning rate, weight decay and optionally momentum and returns an optimizer object
        Args:
            learning_rate: The optimizer's learning rate
            weight_decay: The optimizer's weight decay
            optimizer_type: The optimizer's type [adam or sgd]
            momentum: The optimizer's momentum
        Returns:
            an optimizer object
        """
        logger.debug(f"Optimizer type: {optimizer_type}")
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                #  decay=weight_decay
            )
        elif optimizer_type == "adamw":
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        elif optimizer_type == "adafactor":
            optimizer = tf.keras.optimizers.Adafactor(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            optimizer = tf.keras.optimizers(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        return optimizer

    def print_status_bar(self, iteration, total, loss, metrics=None):
        metrics = " - ".join(
            [
                "{}: {:.4f}".format(m.name, m.result())
                for m in loss + (metrics or [])
            ]
        )

        end = "" if int(iteration) < int(total) else "\n"
        #     print(f"{iteration}/{total} - "+metrics ,end="\r")
        #     print(f"\r{iteration}/{total} - " + metrics, end=end)
        print("\r{}/{} - ".format(iteration, total) + metrics, end=end)

