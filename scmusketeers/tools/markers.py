def load_ref_markers(adata, marker_path):
    """
    loads markers as a dict and filters out the ones which are absent from the adata
    """
    markers_ref_df = pd.read_csv(marker_path, sep=";")
    markers_ref = dict.fromkeys(markers_ref_df.columns)
    for col in markers_ref_df.columns:
        markers_ref[col] = list(
            [
                gene
                for gene in markers_ref_df[col].dropna()
                if gene in adata.var_names
            ]
        )
    return markers_ref


def marker_ranking(markers, adata, obs_key):
    """
    markers : dict of the shape {celltype: [marker list]}
    adata : the dataset to compute markers on
    obs_key : the key where to look up the celltypes. must be coherent with the celltypes of markers

    Computes a score equal to the average ranking of the cell for the expression of each marker
    """
    avg_scores = pd.Series(
        index=adata.obs_names, name=("ranking_marker_average")
    )
    celltypes = np.unique(adata.obs[obs_key])
    for ct in celltypes:
        markers_ct = markers[ct]
        sub_adata = adata[
            adata.obs[obs_key] == ct, markers_ct
        ]  # subset to only keep markers
        marker_scores = pd.DataFrame(
            sub_adata.X.toarray(),
            index=sub_adata.obs_names,
            columns=sub_adata.var_names,
        )
        marker_scores = marker_scores.assign(
            **marker_scores.rank(axis=0, ascending=False, method="min").astype(
                int
            )
        )
        avg_scores[sub_adata.obs_names] = marker_scores.mean(axis=1)
    adata.obs["ranking_marker_average"] = avg_scores
    return avg_scores


def sum_marker_score(markers, adata, obs_key):
    """
    markers : dict of the shape {celltype: [marker list]}
    adata : the dataset to compute markers on
    obs_key : the key where to look up the celltypes. must be coherent with the celltypes of markers

    Computes a score equal to the sum of the expression of each marker for a cell. No need to normalize since it is celltype specific.
    TODO : Add a weighing on each marker if we consider that some are more important than others
    """
    sum_scores = pd.Series(index=adata.obs_names, name=("sum_marker_score"))
    celltypes = np.unique(adata.obs[obs_key])
    for ct in celltypes:
        markers_ct = markers[ct]
        sub_adata = adata[
            adata.obs[obs_key] == ct, markers_ct
        ]  # subset to only keep markers
        marker_scores = pd.DataFrame(
            sub_adata.X.toarray(),
            index=sub_adata.obs_names,
            columns=sub_adata.var_names,
        )
        sum_scores[sub_adata.obs_names] = marker_scores.sum(axis=1)
    adata.obs["sum_marker_score"] = sum_scores
