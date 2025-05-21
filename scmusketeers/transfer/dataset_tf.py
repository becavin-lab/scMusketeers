import os
import sys

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..tools.utils import densify
except ImportError:
    from tools.utils import densify


def get_hvg_common(
    adata_, n_hvg=2000, flavor="seurat", batch_key="manip", reduce_adata=True
):
    """
    Computes a list of hvg which are common to the most different batches.
    """
    adata = adata_.copy()
    check_nnz = np.asarray(adata.X[adata.X != 0][0])
    while len(check_nnz.shape) >= 1:  # Get first non zero element of X
        check_nnz = check_nnz[0]
    if int(check_nnz) == float(check_nnz):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    n_batches = len(adata.obs[batch_key].unique())

    sc.pp.highly_variable_genes(
        adata=adata, n_top_genes=n_hvg, flavor=flavor, batch_key=batch_key
    )
    dispersion_nbatches = adata.var["dispersions_norm"][
        adata.var["highly_variable_nbatches"] == n_batches
    ]  # Starts with genes hvg for every batches
    dispersion_nbatches = dispersion_nbatches.sort_values(ascending=False)
    print(f"Searching for highly variable genes in {n_batches} batches")
    if (
        len(dispersion_nbatches) >= n_hvg
    ):  # If there are already enough hvg in every batch, returns them
        top_genes = list(dispersion_nbatches[:n_hvg].index)
        if reduce_adata:
            return adata[:, top_genes]
        else:
            return top_genes
    print(
        f"Found {len(dispersion_nbatches)} highly variable genes using {n_batches} batches"
    )
    top_genes = list(
        dispersion_nbatches.index
    )  # top_genes is the final selection of genes
    remaining_genes = n_hvg
    enough = False

    while not enough:
        n_batches = (
            n_batches - 1
        )  # Looks for genes hvg for one fewer batch at each iteration
        print(f"Searching for highly variable genes in {n_batches} batches")
        remaining_genes = n_hvg - len(top_genes)  # nb of genes left to find
        dispersion_nbatches = adata.var["dispersions_norm"][
            adata.var["highly_variable_nbatches"] == n_batches
        ]
        dispersion_nbatches = dispersion_nbatches.sort_values(ascending=False)
        if (
            len(dispersion_nbatches) > remaining_genes
        ):  # Enough genes to fill in the rest
            print(
                f"Found {len(dispersion_nbatches)} highly variable genes using {n_batches} batches. Selecting top {remaining_genes}"
            )
            # print(dispersion_nbatches)
            top_genes += list(dispersion_nbatches[:remaining_genes].index)
            enough = True
        else:
            print(
                f"Found {len(dispersion_nbatches)} highly variable genes using {n_batches} batches"
            )
            top_genes += list(dispersion_nbatches.index)

    if reduce_adata:
        return adata_[:, top_genes]

    return top_genes


def load_dataset(ref_path, query_path, class_key, unlabeled_category):
    if not query_path:
        adata = sc.read_h5ad(ref_path)
    else:
        ref = sc.read_h5ad(ref_path)
        query = sc.read_h5ad(query_path)
        query.obs[class_key] = unlabeled_category
        adata = ref.concatenate(query, join="inner")
    if not adata.raw:
        adata.raw = adata
    return adata


class Dataset:
    def __init__(
        self,
        adata,  # full anndata, query is the cells which have unlabeled_category as class_key.
        class_key,
        batch_key,
        filter_min_counts,
        normalize_size_factors,
        size_factor,
        scale_input,
        logtrans_input,
        use_hvg,
        test_split_key,
        unlabeled_category,
        train_test_random_seed,
    ):
        self.adata = adata
        self.class_key = class_key
        self.adata.obs[f"true_{self.class_key}"] = self.adata.obs[
            self.class_key
        ]  # Duplicate original annotation in true_class_key
        self.batch_key = batch_key
        self.filter_min_counts = filter_min_counts
        self.normalize_size_factors = normalize_size_factors
        self.size_factor = size_factor
        self.scale_input = scale_input
        self.logtrans_input = logtrans_input
        self.use_hvg = use_hvg
        self.test_split_key = test_split_key
        # if not semi_sup:
        #     self.semi_sup = True # semi_sup defaults to True, especially for scANVI
        self.unlabeled_category = unlabeled_category
        self.adata_train_extended = (
            anndata.AnnData()
        )  # Contains the train and to-be-defined val data
        self.adata_test = anndata.AnnData()  # The query data
        self.adata_train = anndata.AnnData()
        self.adata_val = anndata.AnnData()
        self.adta_test = anndata.AnnData()
        self.train_test_random_seed = train_test_random_seed
        # self.markers_path = self.dataset_dir + '/' + f'markers/markers_{dataset_name}.csv'

    def normalize(self):
        if self.filter_min_counts:
            sc.pp.filter_genes(self.adata, min_counts=1)
            sc.pp.filter_cells(self.adata, min_counts=1)
        nonzero_genes, _ = sc.pp.filter_genes(self.adata.X, min_counts=1)
        assert (
            nonzero_genes.all()
        ), "Please remove all-zero genes before using DCA."

        if self.size_factor == "raw":  # Computing sf on raw data
            self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )

        if self.use_hvg:
            self.adata = get_hvg_common(
                self.adata, n_hvg=self.use_hvg, batch_key=self.batch_key
            )

        if (
            self.normalize_size_factors
            or self.scale_input
            or self.logtrans_input
        ):
            self.adata.raw = self.adata.copy()
        else:
            self.adata.raw = self.adata

        if self.normalize_size_factors:

            sc.pp.normalize_total(self.adata)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )

        if self.logtrans_input:
            sc.pp.log1p(self.adata)

        if self.scale_input:
            sc.pp.scale(self.adata)

        if self.size_factor == "default":  # Computing sf on preprocessed data
            self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )
        elif self.size_factor == "constant":
            self.adata.obs["size_factors"] = 1.0

    def train_val_split(self):
        self.adata_train_extended = self.adata[
            self.adata.obs[self.class_key] != self.unlabeled_category
        ]  # Contains the train and to-be-defined val data

        # Splits the train in val and train in a stratified way, 80/20
        pct_split = 0.2
        # Dirty workaround for celltypes with 1 cell only, which is a rare case
        stratification = self.adata_train_extended.obs[self.class_key].copy()
        if min(stratification.value_counts()) == 1:
            solo_ct = list(
                stratification.value_counts()[
                    stratification.value_counts() == 1
                ].index
            )
            max_ct = list(
                stratification.value_counts()[
                    stratification.value_counts()
                    == stratification.value_counts().max()
                ].index
            )[0]
            mapping = {}
            for i in solo_ct:
                w = np.where(stratification == i)[0]
                stratification.iloc[w] = max_ct
                mapping[i] = w
        train_idx, val_idx = train_test_split(
            np.arange(self.adata_train_extended.n_obs),
            train_size=pct_split,
            stratify=stratification,
            random_state=self.train_test_random_seed,
        )  # split on the index
        if min(stratification.value_counts()) == 1:
            for ct, w in mapping.items():
                # Adding the single celltype to the training dataset.
                if not w in train_idx:
                    train_idx = np.append(train_idx, w)
                    val_idx = val_idx[val_idx != w]

        spl = pd.Series(
            ["train"] * self.adata_train_extended.n_obs,
            index=self.adata_train_extended.obs.index,
        )
        spl.iloc[val_idx] = "val"
        self.adata_train_extended.obs["train_split"] = spl.values
        print(self.unlabeled_category)
        test_idx = (
            self.adata.obs[self.class_key] == self.unlabeled_category
        )  # boolean
        print(self.adata.obs[self.class_key].unique())
        print(test_idx.sum())
        split = pd.Series(
            ["train"] * self.adata.n_obs, index=self.adata.obs.index
        )
        split[test_idx] = "test"

        split[self.adata_train_extended.obs_names] = (
            self.adata_train_extended.obs["train_split"]
        )
        self.adata.obs["train_split"] = split

    def create_inputs(self):
        """
        Must be called after train_split
        """
        self.adata_train = self.adata[
            self.adata.obs["train_split"] == "train"
        ].copy()
        self.adata_val = self.adata[
            self.adata.obs["train_split"] == "val"
        ].copy()
        self.adata_test = self.adata[
            self.adata.obs["train_split"] == "test"
        ].copy()
        self.X = densify(self.adata.X)
        self.X_train = densify(self.adata_train.X)
        self.X_val = densify(self.adata_val.X)
        self.X_test = densify(self.adata_test.X)
        self.y = self.adata.obs[f"true_{self.class_key}"]
        self.y_train = self.adata_train.obs[f"true_{self.class_key}"]
        self.y_val = self.adata_val.obs[f"true_{self.class_key}"]
        self.y_test = self.adata_test.obs[f"true_{self.class_key}"]
        self.ohe_celltype = OneHotEncoder(
            handle_unknown="ignore"
        )  # TODO : handle the case where there can be unknown celltypes in val, pex adding an output node for 'UNK'
        y = np.array(self.y_train).reshape(-1, 1)
        self.ohe_celltype.fit(y)
        self.y_one_hot = (
            self.ohe_celltype.transform(np.array(self.y).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_train_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_train).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_val_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_val).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_test_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_test).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch = self.adata.obs[self.batch_key]
        self.batch_train = self.adata_train.obs[self.batch_key]
        self.batch_val = self.adata_val.obs[self.batch_key]
        self.batch_test = self.adata_test.obs[self.batch_key]
        self.ohe_batches = OneHotEncoder()
        batch_ID = np.array(self.batch).reshape(
            -1, 1
        )  # We know batches from the whole dataset so we fit on the entire dataset
        self.ohe_batches.fit_transform(batch_ID).astype(float).todense()
        self.batch_one_hot = (
            self.ohe_batches.transform(np.array(self.batch).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch_train_one_hot = (
            self.ohe_batches.transform(
                np.array(self.batch_train).reshape(-1, 1)
            )
            .astype(float)
            .todense()
        )
        self.batch_val_one_hot = (
            self.ohe_batches.transform(np.array(self.batch_val).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch_test_one_hot = (
            self.ohe_batches.transform(
                np.array(self.batch_test).reshape(-1, 1)
            )
            .astype(float)
            .todense()
        )
        self.sf = self.adata.obs["size_factors"].values  # .reshape(-1,1)
        self.sf_train = self.adata_train.obs[
            "size_factors"
        ].values  # .reshape(-1,1)
        self.sf_val = self.adata_val.obs[
            "size_factors"
        ].values  # .reshape(-1,1)
        self.sf_test = self.adata_test.obs[
            "size_factors"
        ].values  # .reshape(-1,1)

    def normalize(self):
        if self.filter_min_counts:
            sc.pp.filter_genes(self.adata, min_counts=1)
            sc.pp.filter_cells(self.adata, min_counts=1)
        nonzero_genes, _ = sc.pp.filter_genes(self.adata.X, min_counts=1)
        assert (
            nonzero_genes.all()
        ), "Please remove all-zero genes before using DCA."

        if self.size_factor == "raw":  # Computing sf on raw data
            self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )

        if self.use_hvg:
            self.adata = get_hvg_common(
                self.adata, n_hvg=self.use_hvg, batch_key=self.batch_key
            )

        if (
            self.normalize_size_factors
            or self.scale_input
            or self.logtrans_input
        ):
            self.adata.raw = self.adata.copy()
        else:
            self.adata.raw = self.adata

        if self.normalize_size_factors:

            sc.pp.normalize_total(self.adata)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )

        if self.logtrans_input:
            sc.pp.log1p(self.adata)

        if self.scale_input:
            sc.pp.scale(self.adata)

        if self.size_factor == "default":  # Computing sf on preprocessed data
            self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )
        elif self.size_factor == "constant":
            self.adata.obs["size_factors"] = 1.0
        # print('right after loading')
        # print(self.adata)
        # print(self.adata_test)
        # print(self.adata_train_extended)
        # print(self.adata_train_extended.obs[self.class_key].value_counts())
        # self.adata_train = self.adata_train_extended.copy()

    def test_split(self, test_index_name=None, test_obs=None):

        if test_index_name:
            test_idx = self.adata.obs_names.isin(test_index_name)
        if test_obs:
            print(f"test obs :{test_obs}")
            test_idx = self.adata.obs[self.batch_key].isin(test_obs)

            split = pd.Series(
                ["train"] * self.adata.n_obs, index=self.adata.obs.index
            )
            split[test_idx] = "test"
            self.adata.obs[self.test_split_key] = split

        # If none of test_index_name or test_obs is defined, defaults to the saved value of self.adata.obs[self.test_split_key]
        self.adata_test = self.adata[
            self.adata.obs[self.test_split_key] == "test"
        ]
        self.adata_train_extended = self.adata[
            self.adata.obs[self.test_split_key] == "train"
        ]
    
    def train_split(
        self,
        mode=None,
        pct_split=None,
        obs_key=None,
        n_keep=None,
        split_strategy="random",
        keep_obs=None,
        obs_subsample=None,
        train_test_random_seed=None,
    ):
        """
        Splits train and val datasets according to several modalities.
        percentage : Classic train test split
            pct_split : proportion (between 0 and 1) of the dataset to use as train
            split_strategy : Method/metric to use to determine which cells to chose from. Currently supported is 'random
        entire_condition : Splits by keeping certain batches of obs_key in train and others in val
            obs_key : in task2, one of adata.obs. Key to use to split the batch
            keep_obs : list of observations of obs_key to keep in train
        fixed_number : Symmetrical subsampling. We keep n_keep cells of each class of obs_key condition. Number of cells in training set will be equal to n_keep * adata.obs[obs_key].unique()
            obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
            n_keep : number of cells to keep for each class.
            split_strategy : Method/metric to use to determine which cells to chose from. Currently supported is 'random
        Asymetrical_subsampling : We subsample one class while keeping the other ones intact. We keep n_keep cells for the obs_subsample class of obs_key.
            obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
            obs_subsample : class of obs_key to subsample
            n_keep : number of cells to keep
        """
        self.mode = mode
        self.train_test_random_seed = train_test_random_seed
        if split_strategy == "avg_marker_ranking":
            markers = load_ref_markers(
                self.adata_train_extended, marker_path=self.markers_path
            )
            if obs_key:
                self.obs_key = obs_key
                avg_scores = marker_ranking(
                    markers,
                    adata=self.adata_train_extended,
                    obs_key=self.obs_key,
                )
            else:
                avg_scores = marker_ranking(
                    markers,
                    adata=self.adata_train_extended,
                    obs_key=self.class_key,
                )
        if split_strategy == "sum_marker_score":
            markers = load_ref_markers(
                self.adata_train_extended, marker_path=self.markers_path
            )
            if obs_key:
                self.obs_key = obs_key
                sum_scores = sum_marker_score(
                    markers,
                    adata=self.adata_train_extended,
                    obs_key=self.obs_key,
                )
            else:
                sum_scores = sum_marker_score(
                    markers,
                    adata=self.adata_train_extended,
                    obs_key=self.class_key,
                )
        if mode == "percentage":
            self.pct_split = pct_split
            print(self.adata_train_extended.obs[self.class_key].value_counts())
            if split_strategy == "random" or not split_strategy:
                # Dirty workaround for celltypes with 1 cell only, which is a rare case
                stratification = self.adata_train_extended.obs[
                    self.class_key
                ].copy()
                if min(stratification.value_counts()) == 1:
                    solo_ct = list(
                        stratification.value_counts()[
                            stratification.value_counts() == 1
                        ].index
                    )
                    max_ct = list(
                        stratification.value_counts()[
                            stratification.value_counts()
                            == stratification.value_counts().max()
                        ].index
                    )[0]
                    mapping = {}
                    for i in solo_ct:
                        w = np.where(stratification == i)[0]
                        stratification.iloc[w] = max_ct
                        mapping[i] = w
                train_idx, val_idx = train_test_split(
                    np.arange(self.adata_train_extended.n_obs),
                    train_size=self.pct_split,
                    stratify=stratification,
                    random_state=self.train_test_random_seed,
                )  # split on the index
                if min(stratification.value_counts()) == 1:
                    for ct, w in mapping.items():
                        # Adding the single celltype to the training dataset.
                        if not w in train_idx:
                            train_idx = np.append(train_idx, w)
                            val_idx = val_idx[val_idx != w]
            if split_strategy == "avg_marker_ranking":
                val_idx = []
                for ct in self.adata_train_extended.obs[
                    self.class_key
                ].unique():
                    sub_adata = self.adata_train_extended[
                        self.adata_train_extended.obs[self.class_key] == ct, :
                    ]
                    val_idx += list(
                        sub_adata.obs["ranking_marker_average"]
                        .sort_values()
                        .tail(int(sub_adata.n_obs * (1 - self.pct_split)))
                        .index
                    )  # select bottom 1-pct % to use as val
            if split_strategy == "sum_marker_score":
                val_idx = []
                for ct in self.adata_train_extended.obs[
                    self.class_key
                ].unique():
                    sub_adata = self.adata_train_extended[
                        self.adata_train_extended.obs[self.class_key] == ct, :
                    ]
                    val_idx += list(
                        sub_adata.obs["sum_marker_score"]
                        .sort_values()
                        .tail(int(sub_adata.n_obs * (1 - self.pct_split)))
                        .index
                    )  # select bottom 1-pct % to use as val

            spl = pd.Series(
                ["train"] * self.adata_train_extended.n_obs,
                index=self.adata_train_extended.obs.index,
            )
            spl.iloc[val_idx] = "val"
            self.adata_train_extended.obs["train_split"] = spl.values
        elif mode == "entire_condition":
            self.obs_key = obs_key
            self.keep_obs = keep_obs
            print(
                f"splitting this adata train/val : {self.adata_train_extended}"
            )
            keep_idx = self.adata_train_extended.obs[obs_key].isin(
                self.keep_obs
            )
            to_keep = pd.Series(
                ["val"] * self.adata_train_extended.n_obs,
                index=self.adata_train_extended.obs.index,
            )
            to_keep[keep_idx] = "train"
            self.adata_train_extended.obs["train_split"] = to_keep
        elif mode == "fixed_number":
            self.obs_key = obs_key
            self.n_keep = n_keep
            keep_idx = []
            for obs_class in self.adata_train_extended.obs[
                self.obs_key
            ].unique():
                if split_strategy == "random":
                    n_keep = min(
                        self.adata_train_extended.obs[
                            self.obs_key
                        ].value_counts()[obs_class],
                        self.n_keep,
                    )  # For celltypes with nb of cells < n_keep, we keep every cells
                    keep = list(
                        self.adata_train_extended[
                            self.adata_train_extended.obs[self.obs_key]
                            == obs_class
                        ]
                        .obs.sample(
                            n_keep, random_state=self.train_test_random_seed
                        )
                        .index
                    )
                if split_strategy == "avg_marker_ranking":
                    sub_adata = self.adata_train_extended[
                        self.adata_train_extended.obs[self.obs_key]
                        == obs_class,
                        :,
                    ]
                    keep = list(
                        sub_adata.obs["ranking_marker_average"]
                        .sort_values()
                        .head(self.n_keep)
                        .index
                    )  # For celltypes with nb of cells < n_keep, we keep every cells
                if split_strategy == "sum_marker_score":
                    sub_adata = self.adata_train_extended[
                        self.adata_train_extended.obs[self.obs_key]
                        == obs_class,
                        :,
                    ]
                    keep = list(
                        sub_adata.obs["sum_marker_score"]
                        .sort_values()
                        .head(self.n_keep)
                        .index
                    )
                keep_idx += keep
            to_keep = pd.Series(
                ["val"] * self.adata_train_extended.n_obs,
                index=self.adata_train_extended.obs.index,
            )
            to_keep[keep_idx] = "train"
            self.adata_train_extended.obs["train_split"] = to_keep
        elif mode == "Asymetrical_subsampling":
            self.obs_key = obs_key
            self.obs_subsample = obs_subsample
            self.n_keep = n_keep
            n_remove = (
                self.adata_train_extended[
                    self.adata_train_extended.obs[self.class_key]
                    == self.obs_subsample
                ].n_obs
                - self.n_keep
            )
            remove_idx = (
                self.adata_train_extended[
                    self.adata_train_extended.obs[self.class_key]
                    == self.obs_subsample
                ]
                .obs.sample(n_remove, random_state=self.train_test_random_seed)
                .index
            )
            to_keep = pd.Series(
                ["train"] * self.adata_train_extended.n_obs,
                index=self.adata_train_extended.obs.index,
            )
            to_keep[remove_idx] = "val"
            self.adata_train_extended.obs["train_split"] = to_keep
        else:
            print(f"{mode} is not a valid splitting mode")
            return

        train_split = self.adata.obs[self.test_split_key].astype(
            "str"
        )  # Replace the test, train, val values in the global adata object
        train_split[self.adata_train_extended.obs_names] = (
            self.adata_train_extended.obs["train_split"]
        )
        self.adata.obs["train_split"] = train_split

        print(
            f'train, test, val proportions : {self.adata.obs["train_split"].value_counts()}'
        )

    def create_inputs(self):
        """
        Must be called after train_split
        """
        self.adata_train = self.adata[
            self.adata.obs["train_split"] == "train"
        ].copy()
        self.adata_val = self.adata[
            self.adata.obs["train_split"] == "val"
        ].copy()
        self.adata_test = self.adata[
            self.adata.obs["train_split"] == "test"
        ].copy()
        self.X = densify(self.adata.X)
        self.X_train = densify(self.adata_train.X)
        self.X_val = densify(self.adata_val.X)
        self.X_test = densify(self.adata_test.X)
        self.y = self.adata.obs[f"true_{self.class_key}"]
        self.y_train = self.adata_train.obs[f"true_{self.class_key}"]
        self.y_val = self.adata_val.obs[f"true_{self.class_key}"]
        self.y_test = self.adata_test.obs[f"true_{self.class_key}"]
        self.ohe_celltype = OneHotEncoder(
            handle_unknown="ignore"
        )  # TODO : handle the case where there can be unknown celltypes in val, pex adding an output node for 'UNK'
        y = np.array(self.y_train).reshape(-1, 1)
        self.ohe_celltype.fit(y)
        self.y_one_hot = (
            self.ohe_celltype.transform(np.array(self.y).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_train_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_train).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_val_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_val).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_test_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_test).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch = self.adata.obs[self.batch_key]
        self.batch_train = self.adata_train.obs[self.batch_key]
        self.batch_val = self.adata_val.obs[self.batch_key]
        self.batch_test = self.adata_test.obs[self.batch_key]
        self.ohe_batches = OneHotEncoder()
        batch_ID = np.array(self.batch).reshape(
            -1, 1
        )  # We know batches from the whole dataset so we fit on the entire dataset
        self.ohe_batches.fit_transform(batch_ID).astype(float).todense()
        self.batch_one_hot = (
            self.ohe_batches.transform(np.array(self.batch).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch_train_one_hot = (
            self.ohe_batches.transform(
                np.array(self.batch_train).reshape(-1, 1)
            )
            .astype(float)
            .todense()
        )
        self.batch_val_one_hot = (
            self.ohe_batches.transform(np.array(self.batch_val).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch_test_one_hot = (
            self.ohe_batches.transform(
                np.array(self.batch_test).reshape(-1, 1)
            )
            .astype(float)
            .todense()
        )
        self.sf = self.adata.obs["size_factors"].values  # .reshape(-1,1)
        self.sf_train = self.adata_train.obs[
            "size_factors"
        ].values  # .reshape(-1,1)
        self.sf_val = self.adata_val.obs[
            "size_factors"
        ].values  # .reshape(-1,1)
        self.sf_test = self.adata_test.obs[
            "size_factors"
        ].values  # .reshape(-1,1)

    def fake_annotation(
        self,
        true_celltype,
        false_celltype,
        pct_false,
        train_test_random_seed=None,
    ):
        """
        Creates fake annotation by modifying true_celltype to false_celltype only in adata_train. Changes pct_false cells to the wrong label
            true_celltype : celltype to fake/rig from
            false_celltype : celltype to fake/rig to
            pct_false : percentage of true false_celltype to keep
        """
        self.true_celltype = true_celltype
        self.false_celltype = false_celltype
        self.pct_false = pct_false
        true_series = self.adata_train.obs[self.class_key][
            self.adata_train.obs[self.class_key] == self.true_celltype
        ]
        false_series = self.adata_train.obs[self.class_key][
            self.adata_train.obs[self.class_key] == self.false_celltype
        ]
        n = len(true_series)
        true_series = shuffle(true_series, random_state=train_test_random_seed)
        false_idx = true_series[: round(n * self.pct_false)].index
        true_idx = true_series[round(n * self.pct_false) :].index
        obs_series_true = self.adata_train.obs[
            f"true_{self.class_key}"
        ].astype(
            "str"
        )  ## The true labels
        obs_series = self.adata_train.obs[self.class_key].astype(
            "str"
        )  # The training labels, which include nan and faked
        # print(f'1 {obs_series_true.value_counts()}')
        # print(f'2 {obs_series_true[false_idx]}')
        # print(f'2.5 {false_idx}')
        # print(f'2.75 {len(false_idx)}')
        obs_series_true[false_idx] = self.false_celltype
        obs_series[false_idx] = self.false_celltype
        # print(f'3 {obs_series_true.value_counts()}')
        self.adata_train.obs[self.class_key] = obs_series
        adata_obs = self.adata.obs[f"true_{self.class_key}"].astype(
            "str"
        )  # The true labels
        adata_obs_train = self.adata.obs[self.class_key].astype(
            "str"
        )  # The training labels, which should include nan and faked
        # print(f'4 {adata_obs.value_counts()}')
        # print(f'adata_obs.index : {adata_obs.index}')
        # print(f'obs_series.index : {obs_series.index}')
        adata_obs.loc[obs_series_true.index] = (
            obs_series_true  # Les valeurs trafiquées
        )
        adata_obs_train.loc[obs_series.index] = (
            obs_series  # Les valeurs trafiquées
        )
        # print(f"il y a {len(true_series)} True ({self.true_celltype}) et {len(false_series)} False ({self.false_celltype})")
        # print(f"on fake donc {len(false_idx)} cellules avec un pct_false de {self.pct_false}")
        # print(f"adata_obs  (faked) : {adata_obs.value_counts()}")
        # print(f"true celltype : {self.adata.obs[f'true_{self.class_key}']}")
        self.adata.obs[f"fake_{self.class_key}"] = adata_obs
        self.adata.obs[self.class_key] = adata_obs_train
        self.adata.obs["faked"] = (
            self.adata.obs[f"fake_{self.class_key}"]
            != self.adata.obs[f"true_{self.class_key}"]
        )
        self.adata.obs["faked_color"] = self.adata.obs["faked"].replace(
            {True: "faked", False: "not faked"}
        )
        # print(self.adata.obs['faked'].value_counts())

    def small_clusters_totest(self):
        inf_n_cells = ~self.adata_train.obs[self.class_key].isin(
            self.adata_train.obs[self.class_key]
            .value_counts()[
                self.adata_train.obs[self.class_key].value_counts()
                > self.n_perm
            ]
            .index
        )  # Celltypes with too few cells in train are transferred to test
        self.adata_train = self.adata_train[~inf_n_cells].copy()
        inf_n_cells = inf_n_cells[inf_n_cells].index
        self.adata.obs.loc[inf_n_cells, "train_split"] = "test"
        self.adata_test = self.adata[
            self.adata.obs["train_split"] == "test"
        ].copy()



def process_dataset(workflow):
    """
    Used for hyperparameters optimization processes
    """
    adata = load_dataset(workflow.run_file.ref_path,
                         workflow.run_file.query_path,
                         workflow.run_file.class_key, 
                         workflow.run_file.unlabeled_category       
    )

    workflow.dataset = Dataset(
        adata=adata,
        class_key=workflow.run_file.class_key,
        batch_key=workflow.run_file.batch_key,
        filter_min_counts=workflow.run_file.filter_min_counts,
        normalize_size_factors=workflow.run_file.normalize_size_factors,
        size_factor=workflow.run_file.size_factor,
        scale_input=workflow.run_file.scale_input,
        logtrans_input=workflow.run_file.logtrans_input,
        use_hvg=workflow.run_file.use_hvg,
        unlabeled_category=workflow.run_file.unlabeled_category,
        test_split_key=workflow.run_file.test_split_key,
        train_test_random_seed=workflow.run_file.train_test_random_seed
    )

    # Processing dataset. Splitting train/test.
    workflow.dataset.normalize()


def split_train_test(workflow):
    workflow.dataset.test_split(
        test_obs=workflow.run_file.test_obs,
        test_index_name=workflow.run_file.test_index_name,
    )


def split_train_val(workflow):
    workflow.dataset.train_split(
        mode=workflow.run_file.mode,
        pct_split=workflow.run_file.pct_split,
        obs_key=workflow.run_file.obs_key,
        n_keep=workflow.run_file.n_keep,
        keep_obs=workflow.run_file.keep_obs,
        split_strategy=workflow.run_file.split_strategy,
        obs_subsample=workflow.run_file.obs_subsample,
        train_test_random_seed=workflow.run_file.train_test_random_seed,
    )

    print("dataset has been preprocessed")
    workflow.dataset.create_inputs()

