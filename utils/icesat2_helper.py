import re
import itertools

from pathlib import Path
from typing import Counter

from loguru import logger

import h5py
import pandas as pd


def is_h5_valid(file: Path) -> bool:
    assert file.exists(), "File does not exist"

    # like this: "processed_ATL03_20211126014738_09871301_006_01.h5"
    pattern = r"processed_ATL03_\d{14}_\d{8}_\d{3}_\d{2}\.h5"
    if re.search(pattern, file.name):
        return True
    else:
        logger.warning("It isn't a atl03.h5 file!")
        return False


# https://www.cnblogs.com/sw-code/p/18161987
def read_hdf5_atl03_beam_h5py(file_path, beam, verbose=False):
    """
    ATL03 原始数据读取
    Args:
        filename (str): h5文件路径
        beam (str): 光束
        verbose (bool): 输出HDF5信息

    Returns:
        返回ATL03光子数据的heights和geolocation信息
    """

    # 打开HDF5文件进行读取
    file_id = h5py.File(file_path, "r")

    # 输出HDF5文件信息
    if verbose:
        logger.info(f"file name: {file_id.filename}")
        logger.info(f"file keys: {list(file_id.keys())}")
        logger.info(f"Metadata keys: {list(file_id['METADATA'].keys())}")

    # 为ICESat-2 ATL03变量和属性分配python字典
    atl03_mds = {}

    # 读取文件中每个输入光束
    beams = [k for k in file_id.keys() if bool(re.match("gt\\d[lr]", k))]
    if beam not in beams:
        logger.error("请填入正确的光束代码")
        return

    atl03_mds["heights"] = {}
    atl03_mds["geolocation"] = {}
    atl03_mds["bckgrd_atlas"] = {}

    # -- 获取每个HDF5变量
    # -- ICESat-2 Measurement Group
    try:
        for key, val in file_id[beam]["heights"].items():
            atl03_mds["heights"][key] = val[:]

        # -- ICESat-2 Geolocation Group
        for key, val in file_id[beam]["geolocation"].items():
            atl03_mds["geolocation"][key] = val[:]

        for key, val in file_id[beam]["bckgrd_atlas"].items():
            atl03_mds["bckgrd_atlas"][key] = val[:]
    except KeyError as e:
        logger.warning(f"{beam}: KeyError: {e}")
        return None

    return atl03_mds


def get_beams(file: Path) -> dict:
    atl03 = {}
    with h5py.File(file, "r") as f:
        beams = [k for k in f.keys() if bool(re.match("gt\\d[lr]", k))]
        for beam in beams:
            data = read_hdf5_atl03_beam_h5py(file, beam, verbose=False)
            if data is None:
                beams.remove(beam)
                atl03.pop(beam, None)
                continue
            atl03[beam] = data

    return atl03


def merge_nested_keys(data):
    # 用于存储最终合并后的数据
    merged_data = {}
    # 遍历数据中的每个键值对
    for key, value in data.items():
        if isinstance(value, dict):
            # 如果值是字典，递归调用 merge_nested_keys 函数
            nested_merged = merge_nested_keys(value)
            # 将递归合并后的数据更新到 merged_data 中
            for nested_key, nested_value in nested_merged.items():
                if nested_key in merged_data:
                    # 如果键已经存在，将值合并（这里简单地使用 extend 方法，假设值是可迭代的）
                    merged_data[nested_key].extend(nested_value)
                else:
                    # 如果键不存在，直接添加到 merged_data 中
                    merged_data[nested_key] = nested_value
        else:
            # 如果值不是字典，直接添加到 merged_data 中
            if key in merged_data:
                merged_data[key].extend([value])
            else:
                merged_data[key] = [value]
    return merged_data


def export_h5_to_csv(source: Path, target: Path = None):
    if not is_h5_valid(source):
        logger.warning("It isn't a atl03.h5 file!")
        logger.warning(f"{source}")
        return None

    if target is None:
        target = source.parent
    elif target.is_dir():
        assert target.exists(), "Directory does not exist"
    elif target.is_file():
        assert "target must be a directory"

    logger.info(f"Reading {source}")
    atl03 = get_beams(source)

    if len(atl03) == 0:
        return None

    logger.debug(f"Exporting {source.stem}")
    for x, y in itertools.product([1, 2, 3], ["l", "r"]):
        if not f"gt{x}{y}" in atl03.keys():
            continue
        logger.info(f"Exporting gt{x}{y}")

        merged_data = merge_nested_keys(atl03[f"gt{x}{y}"])
        # logger.debug({k: len(v) for k, v in merged_data.items() if len(v) == num})

        merged_data = {k: v[0] for k, v in merged_data.items()}
        logger.debug({k: len(v) for k, v in merged_data.items()})

        # 使用列表推导式获取所有列表的长度，并计算众数
        values = [len(v) for v in merged_data.values()]
        mode = Counter(values).most_common(1)[0][0]

        num = len(merged_data["h_ph"])
        logger.debug({k: len(v) for k, v in merged_data.items() if len(v) == num})

        # 使用字典推导式保留长度等于众数的键值对
        merged_data = {k: v for k, v in merged_data.items() if len(v) == mode}
        # 航天器姿态
        merged_data.pop("velocity_sc")
        # 地表类型
        merged_data.pop("surf_type")

        df = pd.DataFrame(merged_data)

        filename = f"{source.stem}_gt{x}{y}_original.csv"
        fp = target / filename

        df.to_csv(fp, index=False)
