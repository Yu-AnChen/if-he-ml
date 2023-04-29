import pandas as pd
import numpy as np


def hist_2d(
    df, bin_size, query=None,
    spatial_x_name='Xt', spatial_y_name='Yt',
    kde=False,
    kde_kwargs=None
):
    # snap coordinates to grid
    df_coords = df[[spatial_y_name, spatial_x_name]] / bin_size
    df_coords = df_coords.apply(lambda x: pd.to_numeric(x.round(), downcast='integer'))
    # set output image size
    h, w = df_coords.max() + 1
    # query by condition string
    if query is not None:
        df_coords = df_coords[df.eval(query)]
    counts = df_coords.groupby([spatial_y_name, spatial_x_name]).size()
    y, x = counts.index.to_frame().values.T
    img = np.zeros((h, w))
    img[y, x] = counts

    if kde:
        img_kde = _kde_2d(img, kde_kwargs)
        return img, img_kde

    return img


def binned_coords(
    df, bin_size,
    spatial_x_name='Xt', spatial_y_name='Yt',
):
    # snap coordinates to grid
    df_coords = df[[spatial_y_name, spatial_x_name]] / bin_size
    df_coords = df_coords.apply(lambda x: pd.to_numeric(x.round(), downcast='integer'))
    return df_coords


import KDEpy

def _kde_2d(img, kde_kwargs=None):
    assert img.ndim == 2
    rs, cs = img.nonzero()
    values = img[rs, cs]

    h, w = img.shape
    if kde_kwargs is None: kde_kwargs = {}
    kde = KDEpy.FFTKDE(**kde_kwargs)
    # 1. FIXME does it need to be shifted by half pixel?
    # 2. ValueError: Every data point must be inside of the grid.
    grid_pts = np.mgrid[-1:h+1, -1:w+1].reshape(2, -1).T
    return (
        kde
        .fit(np.vstack([rs, cs]).T, weights=values)
        .evaluate(grid_pts)
        .reshape(h+2, w+2)[1:-1, 1:-1]
    )


import matplotlib.pyplot as plt
import napari
import matplotlib.cm

def napari_contour(
    img_contour, n_levels,
    cmap_name='viridis',
    viewer=None, add_shape_kwargs=None
):
    _, ax = plt.subplots()
    ax.imshow([[0]])
    ctr = ax.contour(img_contour, levels=n_levels, cmap=cmap_name)

    colors = matplotlib.cm.get_cmap(cmap_name)(np.linspace(0, 1, n_levels+2))[1:-1]
    level_values = ctr.levels[1:-1]

    if viewer is None:
        viewer = napari.Viewer()
    
    kwargs = dict(
        shape_type='polygon',
        face_color=[0]*4,
    )
    if add_shape_kwargs is None: add_shape_kwargs = {}

    for level, color, lv in zip(ctr.allsegs[1:-1], colors, level_values):
        kwargs.update(dict(edge_color=color, name=str(lv)))
        kwargs.update(add_shape_kwargs)
        viewer.add_shapes(
            [np.fliplr(seg)[:-1, :] for seg in level],
            **kwargs
        )
    return viewer

# jdf = pd.read_csv(r"Y:\sorger\data\computation\Yu-An\YC-20230126-squidpy_demo\dataC04.csv")
jdf = pd.read_csv(r"demo-c4-small.csv")
jdf = jdf.set_index('CellID')

topic8 = hist_2d(jdf, bin_size=25, query='topics == 8', kde=True, kde_kwargs=dict(bw=5))
v = napari_contour(topic8[1], n_levels=5, cmap_name='gist_earth')
v.add_image(np.array(topic8), channel_axis=0)


import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def gmm_cluster_pcs(
    df, columns,
    bin_size=224,
    spatial_x_name='X_centroid', spatial_y_name='Y_centroid',
    transform_func=None,
    n_pcs=10, n_components=20
):
    if transform_func is None:
        transform_func = np.array
    if columns is None:
        columns = df.columns
    
    df_coords = binned_coords(df, bin_size, spatial_x_name, spatial_y_name)
    binned_df = (
        df[columns]
        .transform(transform_func)
        .groupby([*df_coords.values.T])
        .mean()
    )
    scaled_data = sklearn.preprocessing.scale(binned_df)
    pca = PCA(n_components=n_pcs, random_state=1001)
    pc_scaled = pca.fit_transform(scaled_data)
    print(np.cumsum(pca.explained_variance_ratio_))
    # rescaled features seems to give cleaner result
    # pc_raw = pca.fit_transform(binned_df)
    # print(pca.explained_variance_ratio_)
    clusters = GaussianMixture(
        n_components=n_components, random_state=1001
    ).fit_predict(pc_scaled)
    # kmeans seems to give noisy result
    # from sklearn.cluster import KMeans
    # clusters = KMeans(
    #     n_clusters=n_components, random_state=1001
    # ).fit_predict(pc_raw)

    binned_df['cluster'] = clusters
    return binned_df


def df2img(df, column_name, bg_value=0):
    coords = df.index.to_frame().values
    h, w = coords.max(axis=0) + 1
    y, x = coords.T

    img = bg_value * np.ones((h, w))
    img[y, x] = df[column_name]
    return img


df = pd.read_csv(r"W:\crc-scans\C1-C40-sc-tables\P37_S29-CRC01\quantification\P37_S29_A24_C59kX_E15@20220106_014304_946511_cellRingMask.csv")

valid_markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
    'PD-1', 'Ki67', 'Pan-CK', 'SMA',
]

tdf = gmm_cluster_pcs(df, valid_markers, transform_func=np.log1p)
timg = df2img(tdf, 'cluster', bg_value=-1)
sampled_tdf = tdf.groupby('cluster').apply(
    lambda x: x.sample(1000, random_state=1001) 
    if x.index.size >=1000 
    else x.sample(x.index.size, random_state=1001)
)
sampled_tdf.shape
plt.figure()
plt.imshow(np.where(timg == -1, np.nan, timg), cmap='tab20', interpolation='nearest')