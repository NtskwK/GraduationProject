import shutil
from pathlib import Path
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def save2dir(
    source: Path,
    target: Path = Path(__file__).parent,
    name: str = None,
    mkdir: bool = True,
    overwrite: bool = False,
    prefix: str = "",
    suffix: str = "",
) -> Path:

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
