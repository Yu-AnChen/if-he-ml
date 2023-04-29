import pathlib

import pandas as pd
import skimage.util
import tifffile
from joblib import Parallel, delayed
import numpy as np

files = pd.read_csv('files.csv')
out_dir = pathlib.Path(r'W:\crc-scans\C1-C40-downsized-16')

def block_mean(path, block_size):

    def _compute_channel(path, block_size, channel):
        channel = tifffile.imread(path, key=channel)
        dtype = channel.dtype
        is_int = np.issubdtype(dtype, np.integer)
        wimg = skimage.util.view_as_windows(channel, block_size, block_size)
        mimg = wimg.mean(axis=(2, 3))
        if is_int:
            np.round(mimg, out=mimg)
            mimg = mimg.astype(dtype)
        return mimg

    out = Parallel(n_jobs=4, verbose=1)(
        delayed(_compute_channel)(
            path, block_size, channel
        ) for channel in range(19)
    )

    return np.array(out)


for _, row in files.iloc[:10].iterrows():
    print('Processing', row['Name'])
    img_block = block_mean(row['Orion filepath'], 16)
    # FIXME round image block to source dtype before writing to disk
    tifffile.imwrite(
        out_dir / f"{row['Name']}-downsized-16.ome.tif",
        img_block,
        compression='zlib',
        tile=(1024, 1024)
    )
    print()


def fix_float_images():
    import pathlib
    import tifffile
    import numpy as np
    from joblib import Parallel, delayed


    ometiffs = sorted(pathlib.Path('.').glob('C1-C40-downsized-16/*.tif'))
    out_dir = pathlib.Path('C1-C40-downsized-16-uint16')
    out_dir.mkdir(exist_ok=True)


    def read_write(pp):
        print('processing', pp.name)
        img = tifffile.imread(pp)
        img = np.around(img).astype(np.uint16)
        tifffile.imwrite(out_dir / pp.name, img, compression='zlib', tile=(1024, 1024))

    Parallel(n_jobs=-1, verbose=1)(
        delayed(read_write)(pp)
        for pp in ometiffs
    )
