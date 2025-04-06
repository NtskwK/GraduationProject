import shutil
from pathlib import Path
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

import settings

def save2dir(
    source: Path,
    target: Path = settings.backup_dir,
    name: str = None,
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

    if name is None:
        name = source.stem
    elif target.is_file():
        name = target.stem
    else:
        name = str(name).rstrip(source.suffix)

    if (not overwrite) and Path(target).exists():
        # Exception("Target already exists")
        name = name + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        fp.unlink()

    fp = Path(fp).rename(target / f"{prefix}{name}{suffix}{source.suffix}")

    print(f"Backup {source} to {fp}")

    return fp


def save_figure(
    fig: Figure,
    name: Path = None,
    backup: bool = False,
    *args,
    **kwargs,
) -> bool:
    
    if name is None:
        name = fig.get_label() + ".png"
        
    fig.savefig(name)
    
    fp = Path(name)
    
    if backup:
        fp = save2dir(
            fp,
            name=name,
            *args,
            **kwargs,
        )

    return fp.exists()


def save_csv(
    df: pd.DataFrame,
    source: Path = None,
    tag: str = None,
    backup: bool = False,
    *args,
    **kwargs,
) -> bool:

    assert "Tag is required!"

    if source is None:
        filename ="_" + tag + ".csv"
    elif source.is_file():
        filename = source.stem + "_" + tag + ".csv"
    else:
        assert "Source file does not exist!"

    filepath = Path(filename)

    df.to_csv(filepath)
    print(f"Save {filepath} successfully!")

    if backup:
        filepath = save2dir(
            filepath,
            name=filename,
            *args,
            **kwargs,
        )

    return filepath.exists()

def get_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), "File does not exist"
    assert path.suffix == ".csv", "File is not a csv file"

    data = None
    with open(path, "r") as f:
        data = pd.read_csv(f)

    assert data is not None, "Failed to read csv file!"

    return pd.read_csv(path)
