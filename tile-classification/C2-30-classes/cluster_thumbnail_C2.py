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

in_dir = pathlib.Path(r'W:\crc-scans\C1-C40-downsized-16-uint16')
in_file = in_dir / 'CRC02-downsized-16.ome.tif'

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

img = tifffile.imread(in_file)
img = downscale_mean(img, 14)
mask = palom.img_util.entropy_mask(img[0], 3)
df = img2df(img, mask, markers)

from spatial_dataframe import (
    gmm_cluster_by_pcs
)

grid_df = gmm_cluster_by_pcs(
    df, valid_markers, bin_size=1, viz=True,
    transform_func=np.log1p, n_components=30
)
grid_df[['train', 'test']] = False, False

trainings = (
    grid_df[['cluster', 'score']]
    .query('score >= -20')
    .groupby('cluster')
    .apply(
        lambda x: x.sample(300, random_state=1001) 
        if x.index.size >= 300 
        else x.sample(x.index.size, random_state=1001)
    )
    .sort_index(level=[1, 2])
    .droplevel(0)
    .index
)
grid_df.loc[trainings, 'train'] = True

testings = (
    grid_df[['cluster', 'score', 'train']]
    .query('score >= -20')
    .query('train == False')
    .groupby('cluster')
    .apply(
        lambda x: x.sample(300, random_state=1001) 
        if x.index.size >= 300 
        else x.sample(x.index.size, random_state=1001)
    )
    .sort_index(level=[1, 2])
    .droplevel(0)
    .index
)
grid_df.loc[testings, 'test'] = True


# write patches
import zarr

files = pd.read_csv('files.csv')
row = files.iloc[1]
CASE_ID = row['Name']

out_dir = pathlib.Path(r'W:\crc-scans\C1-C40-patches\gmm30') / CASE_ID
out_dir.mkdir(exist_ok=True, parents=True)
path_img_he = row['H&E GT450 filepath']

grid_df.to_csv(out_dir.parent / f"{CASE_ID}-gmm30.csv")

for i in np.unique(grid_df['cluster']):
    (out_dir / f"{i:02}").mkdir(exist_ok=True)

N_JOBS = 12
patch_df = grid_df.query('train')
n_patches = patch_df.shape[0]

def write_patch_cluster(df, in_path, out_path, size=224, prefix=CASE_ID):
    zimg = zarr.open(tifffile.imread(in_path, aszarr=True, level=0))
    for idx, row in df.iterrows():
        cluster = int(row['cluster'])
        r, c = idx
        r = int(r*size)
        c = int(c*size)
        img = zimg[:, r:r+size, c:c+size]
        tifffile.imwrite(
            out_path / f"{cluster:02}" / f"{prefix}-rs_{r}-cs_{c}.tif",
            img,
            compression='zlib'
        )
    return

_ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch_cluster)(
    patch_df.iloc[i:i+n_patches // N_JOBS + 1],
    path_img_he,
    out_dir,
) for i in range(0, n_patches, n_patches // N_JOBS + 1))


# clean up by removing blank tiles
import tqdm

class_dirs = sorted(out_dir.glob('*/'))
mean_tiles = []
for dd in tqdm.tqdm(class_dirs):
    mean_tiles.append([
        tifffile.imread(p)[1].mean()
        for p in sorted(dd.glob('*.tif'))[:50]
    ])
mean_tiles = [np.mean(mm) for mm in mean_tiles]
threshold = np.nanmax(mean_tiles) * 0.95
bg_class = np.nanargmax(mean_tiles)

class_dirs.pop(bg_class)

def delete_bg_tiles(in_dir, threshold):
    for pp in sorted(in_dir.glob('*.tif')):
        img = tifffile.imread(pp)[1]
        if (np.sum(img > threshold) / img.size) > 0.5:
            pp.unlink()

_ = Parallel(n_jobs=-1, verbose=1)(
    delayed(delete_bg_tiles)(dd, threshold)
    for dd in class_dirs
)
for pp in class_dirs:
    print(pp.name, len(sorted(pp.glob('*.tif'))))


# write test patches
out_dir = pathlib.Path(r'W:\crc-scans\C1-C40-patches\gmm30') / f"{CASE_ID}-test"
out_dir.mkdir(exist_ok=True, parents=True)

for i in np.unique(grid_df['cluster']):
    (out_dir / f"{i:02}").mkdir(exist_ok=True)

N_JOBS = 12
patch_df = grid_df.query('test')
n_patches = patch_df.shape[0]

_ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch_cluster)(
    patch_df.iloc[i:i+n_patches // N_JOBS + 1],
    path_img_he,
    out_dir,
) for i in range(0, n_patches, n_patches // N_JOBS + 1))


class_dirs = sorted(out_dir.glob('*/'))
mean_tiles = []
for dd in tqdm.tqdm(class_dirs):
    mean_tiles.append([
        tifffile.imread(p)[1].mean()
        for p in sorted(dd.glob('*.tif'))[:50]
    ])
mean_tiles = [np.mean(mm) for mm in mean_tiles]
threshold = np.nanmax(mean_tiles) * 0.95
bg_class = np.nanargmax(mean_tiles)

_ = Parallel(n_jobs=-1, verbose=1)(
    delayed(delete_bg_tiles)(dd, threshold)
    for dd in class_dirs
)
for pp in class_dirs:
    print(pp.name, len(sorted(pp.glob('*.tif'))))

# manually delete class 23 directory, only 8 valid patches
'''
python "Z:\RareCyte-S3\YC-analysis\P37_CRCstudy_Round1\scripts-ml\if-he-ml\tile-classification\huggingface-training\run_image_classification_no_trainer.py"
    --train_dir CRC02 
    --output_dir model_CRC02_10epochs-microsoft-beit-base-patch16-224-pt22k-ft22k 
    --per_device_train_batch_size 32
    --per_device_eval_batch_size 32
    --num_train_epochs 10
    --model_name_or_path microsoft/beit-base-patch16-224-pt22k-ft22k
    --ignore_mismatched_sizes
'''