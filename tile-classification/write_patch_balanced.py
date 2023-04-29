from spatial_dataframe import (
    gmm_cluster_by_pcs
)
import numpy as np

valid_markers = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3',
    'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin',
    'PD-1', 'Ki67', 'Pan-CK', 'SMA',
]

SIZE = 224
N_PATCHES_PER_CLASS = 1100

import pathlib
import pandas as pd

files = pd.read_csv('files.csv')

for _, row in files.iloc[:10].iterrows():
    PREFIX = row['Name']
    # create output directories
    out_dir = pathlib.Path(
        r'W:\crc-scans\C1-C40-patches\20k'
    ) / row['Name']
    (out_dir / 'mask').mkdir(exist_ok=True, parents=True)
    (out_dir / 'img_if').mkdir(exist_ok=True, parents=True)
    (out_dir / 'img_he').mkdir(exist_ok=True, parents=True)

    path_table = row['Quantification table']
    path_img_he = row['H&E GT450 filepath']
    path_img_if = row['Orion filepath']

    if next((out_dir / 'img_he').iterdir(), None) is not None:
        print(out_dir, 'already processed')
        continue

    print('processing', out_dir)
    
    df = pd.read_csv(path_table)
    grid_df = gmm_cluster_by_pcs(df, valid_markers, transform_func=np.log1p, viz=True)
    # drop tiles touching right and bottom edge
    # so one doesn't have to handle out-of-bound croppings
    row_max, col_max = grid_df.index.max()
    grid_df = grid_df.loc[(slice(0, row_max-1), slice(0, col_max-1)), :]

    sampled_grid_df = grid_df.groupby('cluster').apply(
        lambda x: x.sample(N_PATCHES_PER_CLASS, random_state=1001) 
        if x.index.size >= N_PATCHES_PER_CLASS 
        else x.sample(x.index.size, random_state=1001)
    )

    crop_coords = sampled_grid_df.sort_index(level=[1, 2]).index.to_frame()
    crop_coords.loc[:, ['Y_centroid', 'X_centroid']] *= SIZE

    # write crop_coords to disk
    crop_coords.to_csv(out_dir / 'selected_tiles.csv', index=False)


    import tifffile
    import zarr
    from joblib import Parallel, delayed

    N_JOBS = 12
    orion = zarr.open(tifffile.imread(path_img_if, aszarr=True, level=0))

    n_patches = crop_coords.shape[0]


    def write_patch(df, in_path, out_path, size=SIZE, prefix=PREFIX):
        zimg = zarr.open(tifffile.imread(in_path, aszarr=True, level=0))
        for r, c in zip(df['Y_centroid'], df['X_centroid']):
            img = zimg[:, r:r+size, c:c+size]
            tifffile.imwrite(out_path / f"{prefix}-rs_{r}-cs_{c}.tif", img)
        return 

    _ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch)(
        crop_coords.iloc[i:i+n_patches // N_JOBS + 1],
        path_img_he,
        out_dir / 'img_he',
    ) for i in range(0, n_patches, n_patches // N_JOBS + 1))

    _ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(write_patch)(
        crop_coords.iloc[i:i+n_patches // N_JOBS + 1],
        path_img_if,
        out_dir / 'img_if',
    ) for i in range(0, n_patches, n_patches // N_JOBS + 1))


    # remove blurry image and shifted image pairs
    import scipy.ndimage
    import skimage.restoration
    import skimage.registration
    # Pre-calculate the Laplacian operator kernel. We'll always be using 2D images.
    _laplace_kernel = skimage.restoration.uft.laplacian(2, (3, 3))[1]

    def whiten(img, sigma):
        img = skimage.img_as_float32(img)
        if sigma == 0:
            output = scipy.ndimage.convolve(img, _laplace_kernel)
        else:
            output = scipy.ndimage.gaussian_laplace(img, sigma)
        return output

    def register(img1, img2, sigma, upsample=1):
        img1w = whiten(img1, sigma)
        img2w = whiten(img2, sigma)
        return skimage.registration.phase_cross_correlation(
            img1w,
            img2w,
            upsample_factor=upsample,
            normalization=None,
            return_error=False,
        )

    img_he_files = sorted(pathlib.Path(out_dir / 'img_he').glob('*.tif'))
    img_if_files = sorted(pathlib.Path(out_dir / 'img_if').glob('*.tif'))

    assert len(img_he_files) == len(img_if_files)

    def remove_bad_tiles(p1, p2):
        img = tifffile.imread(p1)[1]
        img2 = tifffile.imread(p2, key=0)
        if whiten(np.where(img == 0, img.max(), img), 0).var() < 8e-4:
            p1.unlink()
            p2.unlink()
            return
        if np.linalg.norm(register(img, img2, 1)) > (1.5 / 0.325):
            p1.unlink()
            p2.unlink()
        return 


    _ = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(remove_bad_tiles)(
        ih, ii
    ) for ih, ii in zip(img_he_files, img_if_files))


