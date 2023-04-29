import pathlib
import pandas as pd
import tifffile
import skimage.util
import numpy as np
import palom
from joblib import Parallel, delayed


def downscale_mean(img, block_size):
    assert img.ndim == 3
    wimg = skimage.util.view_as_windows(
        img, (1, block_size, block_size), (1, block_size, block_size)
    )
    return wimg.mean(axis=(-3, -2, -1))


def img2df(img, mask=None, column_names=None):
    assert img.ndim == 3
    C, Y, X = img.shape
    if mask is None:
        mask = np.ones((Y, X), dtype=bool)
    coords = np.mgrid[:Y, :X]
    channels = img[:, mask]
    coords = coords[:, mask]
    df = pd.DataFrame(
        channels.T,
        columns=column_names
    )
    df.loc[:, ['Y_centroid', 'X_centroid']] = coords.T
    return df

in_dir = pathlib.Path(r'W:\crc-scans\C1-C40-downsized-16')
in_files = sorted(in_dir.glob('*.ome.tif'))[:20]
markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'Argo550', 'CD4',
    'FOXP3', 'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163',
    'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
]
valid_markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
    'PD-1', 'Ki67', 'Pan-CK', 'SMA',
]
imgs = Parallel(n_jobs=10, verbose=1)(delayed(downscale_mean)(
    tifffile.imread(p), 14
) for p in in_files)
masks = [
    palom.img_util.entropy_mask(img[0], 3)
    for img in imgs
]
dfs = [
    img2df(img, mask, markers)
    for img, mask in zip(imgs, masks)
]
from spatial_dataframe import (
    gmm_cluster_by_pcs,
    df2img,
    pca_channels
)
for df, p in zip(dfs, in_files):
    df['case'] = p.name.split('-')[0]


import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


group_df = pd.concat(dfs[:])
pca = PCA(n_components=10, random_state=1001)
pcs = pca.fit_transform(
    sklearn.preprocessing.scale(
        group_df[valid_markers].transform(np.log1p)
    )
)
print(np.cumsum(pca.explained_variance_ratio_))
gmm = GaussianMixture(n_components=30, random_state=1001)
clusters = gmm.fit_predict(pcs)
scores = gmm.score_samples(pcs)

group_df[['cluster', 'score']] = np.array([clusters, scores]).T
group_df.set_index(['Y_centroid', 'X_centroid'], inplace=True)
group_df.to_csv(in_dir / 'gmm30-pooled_standardize-20_cases.csv')


def _standardize_df(df):
    std_df = df.copy()
    std_df.loc[:, valid_markers] = sklearn.preprocessing.scale(
        df[valid_markers].transform(np.log1p)
    )
    return std_df
std_dfs = [
    _standardize_df(df)
    for df in dfs
]
std_group_df = pd.concat(std_dfs)
pca = PCA(n_components=10, random_state=1001)
pcs = pca.fit_transform(
    std_group_df[valid_markers]
)
print(np.cumsum(pca.explained_variance_ratio_))
std_gmm = GaussianMixture(n_components=30, random_state=1001)
clusters = std_gmm.fit_predict(pcs)
scores = std_gmm.score_samples(pcs)

std_group_df[['cluster', 'score']] = np.array([clusters, scores]).T
std_group_df.set_index(['Y_centroid', 'X_centroid'], inplace=True)
std_group_df.to_csv(in_dir / 'gmm30-individual_standardize-20_cases.csv')



group_df = pd.read_csv(r"W:\crc-scans\C1-C40-downsized-16\gmm30-pooled_standardize-20_cases.csv", index_col=['Y_centroid', 'X_centroid'])

pca = PCA(n_components=10, random_state=1001)
pcs = pca.fit_transform(
    sklearn.preprocessing.scale(
        group_df[valid_markers].transform(np.log1p)
    )
)
print(np.cumsum(pca.explained_variance_ratio_))
import sklearn.manifold

tsne = sklearn.manifold.TSNE(init='random', learning_rate='auto', perplexity=10)
coords_tsne = tsne.fit_transform(pcs[::10])

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(*coords_tsne.T, s=3, linewidths=0, c=group_df.case[::10].apply(lambda x: int(x.replace('CRC', ''))), cmap='tab20')
plt.gca().axis('equal')


import umap
reducer = umap.UMAP()
coords_umap = reducer.fit_transform(pcs[::10])