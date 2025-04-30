import shutil

from pathlib import Path
from datetime import datetime
from os import cpu_count

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from rasterio import DatasetReader
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import reproject, calculate_default_transform, Resampling

import settings


def save2dir(
    source: Path,
    backup_name: str,
    target: Path = settings.backup_dir,
    mkdir: bool = True,
    overwrite: bool = False,
    prefix: str = "",
    suffix: str = "",
) -> Path:
    print(f"try to backup {source} to {target}")
    assert source.exists(), "Source directory does not exist"

    if mkdir:
        target.mkdir(exist_ok=True)
    else:
        assert target.exists(), "Target directory does not exist"

    temp_name = source.stem + ".tmp"
    temp = source.parent / temp_name
    fp = shutil.copy(source, temp)

    if backup_name is None:
        backup_name = source.stem
    elif target.is_file():
        backup_name = target.stem
    else:
        backup_name = str(backup_name).rstrip(source.suffix)

    if (not overwrite) and Path(target).exists():
        # Exception("Target already exists")
        backup_name = backup_name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        fp.unlink()

    fp = Path(fp).rename(target / f"{prefix}{backup_name}{suffix}{source.suffix}")

    print(f"Backup {source} to {fp}")

    return fp


def save_figure(
    fig: Figure,
    name: str,
    *args,
    backup: bool = False,
    **kwargs,
) -> bool:

    if name is None:
        name = fig.get_label() + ".png"

    fig.savefig(name)

    fp = Path(name)

    if backup:
        fp = save2dir(
            fp,
            backup_name=name,
            *args,
            **kwargs,
        )

    return fp.exists()


def save_csv(
    df: pd.DataFrame,
    data_name: Path,
    tag: str,
    *args,
    backup: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> bool:

    filename = data_name + "_" + tag + ".csv"
    filepath = Path(filename)

    if filepath.exists() and not overwrite:
        print(f"{filepath} already exists, will be overwritten!")
        return False

    df.to_csv(filepath)
    print(f"Save {filepath} successfully!")

    if backup:
        filepath = save2dir(
            filepath,
            backup_name=filename,
            *args,
            **kwargs,
        )

    return filepath.exists()


def get_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), "File does not exist"
    assert path.suffix == ".csv", "File is not a csv file"

    data = None
    with open(path, "r", encoding="utf-8") as f:
        data = pd.read_csv(f)

    assert data is not None, "Failed to read csv file!"

    return pd.read_csv(path)


def get_value_from_raster(
    src: DatasetReader, xs: float, ys: float, index: int | list[int] = 0
) -> np.ndarray:
    """
    Get the value from a raster file at a specific longitude and latitude.
    :param raster: The raster file to read from.
    :param xs: The longitude list of the point to read.
    :param ys: The latitude list of the point to read.
    :param index: The index of the band to read from. **If index is 0, all band will be read.**
    :return: The values at the specified longitude and latitude.
    """
    print(f"{src.read_crs()}")

    rows, cols = rowcol(src.transform, xs, ys)

    assert isinstance(index,int) or isinstance(index, list[int]), "Band index must be an int or a list of ints"

    if isinstance(index, list):
        assert all(isinstance(i, int) for i in index), "Band index must be an int or a list of ints"
        assert len(index) < src.count, "Band index list length does not match the number of bands"
        assert all(i <= src.count for i in index), "Band index out of range"
        assert all(i > 0 for i in index), "Band index must be greater than 0"

        # Read only the specified bands
        values = np.zeros((len(index), len(xs)), dtype=src.dtypes[0])
        for i, band_index in enumerate(index):
            raster_data = src.read(band_index)
            values[i] = raster_data[rows, cols]

    elif index == 0:
        # Read all bands
        values = np.zeros((src.count, len(xs)), dtype=src.dtypes[0])
        for i in range(src.count):
            raster_data = src.read(i + 1)
            values[i] = raster_data[rows, cols]

    else:
        # Read only the specified band
        assert index <= src.count, "Band index out of range"
        assert index > 0, "Band index must be greater than 0"
        # Read only the specified band
        values = np.zeros((len(xs),), dtype=src.dtypes[0])
        raster_data = src.read(index)
        values = raster_data[rows, cols]

    return values


def reproject2(src: str, dst: str, epsg: int = 4326):
    """
    Reproject a raster file to a new coordinate reference system (CRS). [refer](https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-a-geotiff-dataset)

    :param src: The source raster file to reproject.
    :param dst: The target raster file to save the reprojected data.
    :param epsg: The EPSG code of the target CRS. Default is 4326 (WGS84).
    :return: None
    """
    assert Path(src).exists(), "Source file does not exist!"
    if Path(dst).exists():
        raise FileExistsError("Target file has already exist!")

    dst_crs = rasterio.crs.CRS.from_epsg(epsg)

    with rasterio.open(src) as src:
        transform, width, height = calculate_default_transform(
            src_crs=src.crs,
            dst_crs=dst_crs,
            width=src.width,
            height=src.height,
            left=src.bounds.left,
            bottom=src.bounds.bottom,
            right=src.bounds.right,
            top=src.bounds.top,
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(dst, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    num_threads=cpu_count(),
                )
