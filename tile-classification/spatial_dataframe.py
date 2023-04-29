import pandas as pd
import numpy as np

import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def binned_coords(
    df, bin_size,
    spatial_x_name='Xt', spatial_y_name='Yt',
):
    # snap coordinates to grid
    df_coords = df[[spatial_y_name, spatial_x_name]] / bin_size
    df_coords = df_coords.apply(lambda x: pd.to_numeric(x.round(), downcast='integer'))
    return df_coords


def pca_channels(df, columns, n_pcs=10, transform_func=None, standardize=True):
    if transform_func is None:
        transform_func = lambda x: x
    if columns is None:
        columns = df.columns

    df_channels = df[columns].transform(transform_func)
    scaled_data = df_channels
    if standardize:
        scaled_data = sklearn.preprocessing.scale(df_channels)
    pca = PCA(n_components=n_pcs, random_state=1001)
    pca.fit(scaled_data)
    print(np.cumsum(pca.explained_variance_ratio_))
    return pca


def gmm_cluster_by_pcs(
    df, columns,
    bin_size=224,
    spatial_x_name='X_centroid', spatial_y_name='Y_centroid',
    transform_func=None,
    n_pcs=10, n_components=20,
    standardize=True,
    viz=False
):
    if transform_func is None:
        transform_func = lambda x: x
    df_coords = binned_coords(df, bin_size, spatial_x_name, spatial_y_name)
    binned_df = (
        df[columns]
        .transform(transform_func)
        .groupby([df_coords[spatial_y_name], df_coords[spatial_x_name]])
        .mean()
    )
    pca = PCA(n_components=n_pcs, random_state=1001)
    # rescaled features seems to give cleaner result
    if standardize:
        pcs = pca.fit_transform(
            sklearn.preprocessing.scale(binned_df)
        )
    else:
        pcs = pca.fit_transform(binned_df)
    print(np.cumsum(pca.explained_variance_ratio_))

    gmm = GaussianMixture(n_components=n_components, random_state=1001)
    clusters = gmm.fit_predict(pcs)
    scores = gmm.score_samples(pcs)

    # kmeans seems to give noisy result
    # from sklearn.cluster import KMeans
    # clusters = KMeans(
    #     n_clusters=n_components, random_state=1001
    # ).fit_predict(pc_raw)
    binned_df['cluster'] = clusters
    binned_df['score'] = scores
    if viz:
        plot_cluster(binned_df, column_name='cluster', bg_value=-1, cmap='tab20')
    return binned_df


def gmm_cluster_by_intensity(
    df, columns,
    bin_size=224,
    spatial_x_name='X_centroid', spatial_y_name='Y_centroid',
    transform_func=None,
    n_components=20,
    standardize=True,
    viz=False
):
    if transform_func is None:
        transform_func = np.array
    if columns is None:
        columns = df.columns
    
    df_coords = binned_coords(df, bin_size, spatial_x_name, spatial_y_name)
    binned_df = (
        df[columns]
        .transform(transform_func)
        .groupby([df_coords[spatial_y_name], df_coords[spatial_x_name]])
        .mean()
    )
    scaled_data = binned_df
    if standardize:
        scaled_data = sklearn.preprocessing.scale(binned_df)
    gmm = GaussianMixture(n_components=n_components, random_state=1001)
    clusters = gmm.fit_predict(scaled_data)
    scores = gmm.score_samples(scaled_data)

    # kmeans seems to give noisy result
    # from sklearn.cluster import KMeans
    # clusters = KMeans(
    #     n_clusters=n_components, random_state=1001
    # ).fit_predict(pc_raw)
    binned_df['cluster'] = clusters
    binned_df['score'] = scores
    if viz:
        plot_cluster(binned_df, column_name='cluster', bg_value=-1, cmap='tab20')
    return binned_df


def df2img(df, column_name, bg_value=0):
    coords = df.index.to_frame().values
    h, w = coords.max(axis=0) + 1
    y, x = coords.T

    img = bg_value * np.ones((h, w))
    img[y, x] = df[column_name]
    return img


def plot_cluster(df, column_name='cluster', bg_value=-1, cmap='tab20', ax=None, **kwargs):
    import matplotlib.pyplot as plt
    img = df2img(df, column_name, bg_value)
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(
        np.where(img == -1, np.nan, img),
        cmap=cmap,
        interpolation='nearest',
        **kwargs
    )
    return ax.get_figure()


