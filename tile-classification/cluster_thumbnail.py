import pathlib
import pandas as pd
import palom
import napari
import seaborn as sns
import matplotlib.cm
import matplotlib.pyplot as plt
import skimage.util
import tifffile
import zarr
import numpy as np
from spatial_dataframe import (
    gmm_cluster_by_pcs,
    df2img
)

files = pd.read_csv('files.csv')
valid_markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
    'PD-1', 'Ki67', 'Pan-CK', 'SMA',
]


row = files.iloc[1]

df = pd.read_csv(row['Quantification table'])
grid_df = gmm_cluster_by_pcs(df, valid_markers, bin_size=224, viz=True, transform_func=np.log1p, n_components=20)

plt.figure()
sns.heatmap(grid_df.groupby('cluster').mean(), annot=True, vmin=5.5, vmax=7)
plt.figure()
grid_df.groupby('cluster').size().plot(kind='bar')

img = df2img(grid_df, 'cluster', -1)
cimg = matplotlib.cm.tab20(img.astype(int))
cimg[..., 3][img == -1] = 0

reader = palom.reader.OmePyramidReader(row['H&E GT450 filepath'])
v = napari.Viewer()
v.add_image([np.moveaxis(p, 0, 2) for p in reader.pyramid], scale=(1/224, 1/224))

v.add_image(cimg, translate=(.5, .5))



from joblib import Parallel, delayed
def block_mean(path, block_size):

    def _compute_channel(path, block_size, channel):
        channel = tifffile.imread(path, key=channel)
        wimg = skimage.util.view_as_windows(channel, block_size, block_size)
        return wimg.mean(axis=(2, 3))

    out = Parallel(n_jobs=4, verbose=1)(
        delayed(_compute_channel)(
            path, block_size, channel
        ) for channel in range(19)
    )

    return np.array(out)

img_block = block_mean(row['Orion filepath'], 224)
markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'Argo550', 'CD4',
    'FOXP3', 'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163',
    'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
]
df_block = pd.DataFrame(img_block.reshape(19, -1).T, columns=markers)

coords = np.mgrid[:img_block.shape[1], :img_block.shape[2]]
df_block['X_centroid'] = coords[1].flatten()
df_block['Y_centroid'] = coords[0].flatten()

mask = palom.img_util.entropy_mask(img_block[0], 3)
block_grid_df =gmm_cluster_by_pcs(
    df_block.loc[mask.flatten(), :],
    valid_markers,
    bin_size=1,
    viz=True,
    transform_func=np.log1p,
    n_components=30
)

img = df2img(block_grid_df, 'cluster', -1)
cimg2 = matplotlib.cm.tab20(img.astype(int))
cimg2[..., 3][img == -1] = 0


sample_block_grid_df = (
    block_grid_df
    .query('score >= 0')
    .groupby('cluster')
    .apply(
        lambda x: x.sample(200, random_state=1001) 
        if x.index.size >= 200 
        else x.sample(x.index.size, random_state=1001)
    )
    .sort_index(level=[1, 2])
)

index_diff = block_grid_df.index.difference(
    pd.MultiIndex.from_frame(
        sample_block_grid_df
            .index
            .to_frame()[['Y_centroid', 'X_centroid']]
    )
)

unused_block = block_grid_df.loc[index_diff]

sample_block_grid_df.index.rename(
    dict(X_centroid='col_start', Y_centroid='row_start'),
    inplace=True
)
crop_coords = sample_block_grid_df.index.to_frame()
crop_coords.loc[:, ['row_start', 'col_start']] *= 224
sample_block_grid_df.set_index(pd.MultiIndex.from_frame(crop_coords), inplace=True)

out_dir = pathlib.Path(r'W:\crc-scans\C1-C40-patches\gmm30') / row['Name']
out_dir.mkdir(exist_ok=True, parents=True)
path_img_he = row['H&E GT450 filepath']

sample_block_grid_df.to_csv(out_dir / 'selected_tiles.csv', index=False)

for i in np.unique(sample_block_grid_df['cluster']):
    (out_dir / str(i)).mkdir(exist_ok=True)

PREFIX = row['Name']
N_JOBS = 12
n_patches = crop_coords.shape[0]

def write_patch_cluster(df, in_path, out_path, size=224, prefix=PREFIX):
    zimg = zarr.open(tifffile.imread(in_path, aszarr=True, level=0))
    for r, c, cluster in zip(df['row_start'], df['col_start'], df['cluster']):
        img = zimg[:, r:r+size, c:c+size]
        tifffile.imwrite(out_path / f"{cluster}" / f"{prefix}-rs_{r}-cs_{c}.tif", img)
    return

_ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch_cluster)(
    crop_coords.iloc[i:i+n_patches // N_JOBS + 1],
    path_img_he,
    out_dir,
) for i in range(0, n_patches, n_patches // N_JOBS + 1))


# write 200 patches for test dataset
sample_unused_block = (
    unused_block
    .query('score >= 0')
    .groupby('cluster')
    .apply(
        lambda x: x.sample(200, random_state=1001) 
        if x.index.size >= 200 
        else x.sample(x.index.size, random_state=1001)
    )
    .sort_index(level=[1, 2])
)
sample_unused_block.index.rename(
    dict(X_centroid='col_start', Y_centroid='row_start'),
    inplace=True
)
crop_coords = sample_unused_block.index.to_frame()
crop_coords.loc[:, ['row_start', 'col_start']] *= 224
sample_unused_block.set_index(pd.MultiIndex.from_frame(crop_coords), inplace=True)


out_dir = pathlib.Path(r'W:\crc-scans\C1-C40-patches\gmm30') / f"{row['Name']}-test-set"
out_dir.mkdir(exist_ok=True, parents=True)
path_img_he = row['H&E GT450 filepath']

sample_unused_block.to_csv(out_dir / 'selected_tiles.csv', index=False)

for i in np.unique(sample_unused_block['cluster']):
    (out_dir / str(i)).mkdir(exist_ok=True)

_ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch_cluster)(
    crop_coords.iloc[i:i+n_patches // N_JOBS + 1],
    path_img_he,
    out_dir,
) for i in range(0, n_patches, n_patches // N_JOBS + 1))