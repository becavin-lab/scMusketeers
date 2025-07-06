import gc
import os
import random
import subprocess as sp
import time
import logging

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

logger = logging.getLogger("Sc-Musketeers")

def get_gpu_memory(txt):
    # command = "nvidia-smi --query-gpu=memory.used --format=csv"
    # memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    memory_free_values = tf.config.experimental.get_memory_info("GPU:0")[
        "current"
    ]
    logger.debug(f"GPU memory usage in {txt} HERE: {memory_free_values/1e6} MB ")


import random


def make_random_seed():
    random_seed = time.time()
    random_seed = int((random_seed * 10 - int(random_seed * 10)) * 10e7)
    return random_seed


def random_derangement(array):
    while True:
        array = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if array[p] == j:
                break
            else:
                array[j], array[p] = array[p], array[j]
        else:
            if array[0] != 0:
                return array


def make_training_pairs(ind_classes, n_perm):
    """
    Creates n_perm permutations of indices for the indices of a particular class.
    ind_classes : the dataframe index of interest for a particular class
    n_perm : number of permutations
    """
    n_c = len(ind_classes)
    # X_1 = [[ind_classes[i]] * n_perm for i in range(n_c)] # Duplicate the index
    X_1 = ind_classes * n_perm
    # X_1 = [x for sublist in X_1 for x in sublist]
    X_perm = [
        sklearn.utils.shuffle(ind_classes, random_state=make_random_seed())
        for i in range(n_perm)
    ]  # The corresponding permuted value
    X_perm = [x for sublist in X_perm for x in sublist]
    return X_1, X_perm


def make_training_set(
    y, n_perm, same_class_pct=None, unlabeled_category="UNK"
):
    """
    Creates the total list of permutations to be used by the generator to create shuffled training batches
    same_class_pct : When using contrastive loss, indicates the pct of samples to permute within their class. set to None when not using contrastive loss
    """
    permutations = [[], []]
    logger.debug("Change the cell permutations")

    # y = np.array(y).astype(str) If passing labels as string

    y_cl = np.asarray(y.argmax(axis=1)).flatten()  # convert one_hot to classes
    classes = np.unique(y_cl)
    ind_c = list(range(len(y_cl)))
    if same_class_pct:
        ind_c_same, ind_c_diff = train_test_split(
            ind_c, train_size=same_class_pct, random_state=make_random_seed()
        )  # shuffling same_class_pct % in same class and the rest at random
        X1, Xperm = make_training_pairs(ind_c_diff, n_perm)
        permutations[0] += X1
        permutations[1] += Xperm
    else:
        ind_c_same = ind_c
    for classe in classes:
        if (
            classe == unlabeled_category
        ):  # We mark unknown classes with the 'UNK' tag
            ind_c = list(
                set(list(np.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1 = make_training_pairs(ind_c, n_perm)[0]
            permutations[0] += X1
            permutations[
                1
            ] += X1  # if the class is unknown, we reconstruct the same cell
        else:
            ind_c = list(
                set(list(np.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1, Xperm = make_training_pairs(ind_c, n_perm)
            permutations[0] += X1
            permutations[1] += Xperm
    return [(a, b) for a, b in zip(permutations[0], permutations[1])]


def generate_permutation_map(y, unlabeled_category="UNK"):
    """
    Generates a memory-efficient permutation map for an entire dataset.
    For each index `i`, permutation_map[i] will be its paired index.

    Args:
        y: The one-hot encoded class labels.
        unlabeled_category: The category for cells that should be reconstructed as themselves.

    Returns:
        A NumPy array representing the permutation map.
    """
    logger.debug("Generate permutation map")
    y_cl = np.asarray(y.argmax(axis=1)).flatten()
    classes = np.unique(y_cl)
    
    # Create an array of original indices and an empty map to store the results.
    original_indices = np.arange(len(y_cl))
    permutation_map = np.empty_like(original_indices)

    for classe in classes:
        # Find the indices that belong to the current class
        (class_indices,) = np.where(y_cl == classe)

        if classe == unlabeled_category:
            # For unlabeled cells, the paired index is the cell itself (identity mapping)
            shuffled_class_indices = class_indices
        else:
            # For labeled cells, shuffle their indices to create new pairs
            shuffled_class_indices = np.random.permutation(class_indices)
        
        # Assign the shuffled indices to their original positions in the map
        permutation_map[class_indices] = shuffled_class_indices
    
    return permutation_map

def make_training_set_tf(
    y, n_perm, same_class_pct=None, unlabeled_category="UNK"
):
    """
    Creates the total list of permutations to be used by the generator to create shuffled training batches
    same_class_pct : When using contrastive loss, indicates the pct of samples to permute within their class. set to None when not using contrastive loss
    """
    permutations = [[], []]
    print("switching perm")
    # y = np.array(y).astype(str) If passing labels as string
    y_cl = tf.math.argmax(y, axis=1)
    classes = tf.unique(y_cl)
    ind_c = list(range(len(y_cl)))

    if same_class_pct:
        ind_c_same, ind_c_diff = train_test_split(
            ind_c, train_size=same_class_pct, random_state=make_random_seed()
        )  # shuffling same_class_pct % in same class and the rest at random
        X1, Xperm = make_training_pairs(ind_c_diff, n_perm)
        permutations[0] += X1
        permutations[1] += Xperm
    else:
        ind_c_same = ind_c
    for classe in classes:
        if (
            classe == -1
        ):  # unlabeled_category # We mark unknown classes with the 'UNK' tag
            ind_c = list(
                set(list(tf.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1 = make_training_pairs(ind_c, n_perm)[0]
            permutations[0] += X1
            permutations[
                1
            ] += X1  # if the class is unknown, we reconstruct the same cell
        else:
            ind_c = list(
                set(list(tf.where(y_cl == classe)[0])) & set(ind_c_same)
            )
            X1, Xperm = make_training_pairs(ind_c, n_perm)
            permutations[0] += X1
            permutations[1] += Xperm
    return [(a, b) for a, b in zip(permutations[0], permutations[1])]


def batch_generator_training_permuted(
    X,
    y,
    sf,
    batch_ID=None,
    batch_size=128,
    ret_input_only=False,
    n_perm=1,  # TODO : remove n_perm. We always use n_perm=1, one epoch = one pass of the dataset
    change_perm=True,
    same_class_pct=None,
    unlabeled_category="UNK",
    use_perm=True,
):
    """
    Memory-optimized permuted batch generator.
    """
    logger.debug("Generate batches")
    n_samples = X.shape[0]
    
    # 1. The main index is now just a simple range of integers.
    main_indices = np.arange(n_samples)

    while True:
        # 2. Generate a memory-efficient permutation map at the start of each epoch.
        if use_perm:
            # This creates a single array map, not a giant list of tuples.
            permutation_map = generate_permutation_map(y, unlabeled_category)
        
        # Shuffle the main indices to randomize batch order
        np.random.shuffle(main_indices)
        
        for i in range(0, n_samples, batch_size):
            # Get the input indices for the current batch
            index_in_batch = main_indices[i : i + batch_size]
            
            # 3. Get output indices directly from the map or use input indices for AE.
            if use_perm:
                ind_out_batch = permutation_map[index_in_batch]
            else:
                ind_out_batch = index_in_batch  # Standard autoencoder case

            # --- Batch data preparation (same as before) ---
            sf_in_batch = sf.iloc[index_in_batch]
            y_in_batch = y[index_in_batch]
            batch_ID_in_batch = batch_ID[index_in_batch, :] if batch_ID is not None else None

            X_in_batch = X[index_in_batch, :]
            X_out_batch = X[ind_out_batch, :]

            if scipy.sparse.issparse(X):
                X_in_batch = X_in_batch.toarray()
                X_out_batch = X_out_batch.toarray()
            
            # --- Yielding logic (same as before) ---
            if batch_ID is not None:
                yield (
                    {"counts": X_in_batch, "size_factors": sf_in_batch},
                    {
                        "classifier": y_in_batch,
                        "batch_discriminator": batch_ID_in_batch,
                        "reconstruction": X_out_batch,
                    },
                )
            # Add other yielding conditions if necessary (e.g., ret_input_only)
            else:
                if ret_input_only:
                    yield ({"counts": X_in_batch, "size_factors": sf_in_batch})
                else:
                    yield ({"counts": X_in_batch, "size_factors": sf_in_batch}, X_out_batch)

        # The loop naturally restarts for the next epoch, and a new map will be generated.
        gc.collect()