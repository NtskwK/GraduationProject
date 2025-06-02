import shutil

from loguru import logger

import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
from pathlib import Path

from utils.denoise import *
from utils.data_io import *
from utils.plot import get_plt
from utils.property import ICESAT2Properties
from utils.anti_rasterization import get_keypoint

# 创建隐藏的主窗口（避免显示额外窗口）
# root = tk.Tk()
# root.withdraw()


# PATH_STR = r".\data\icepyx\GoldenBay\20220325\processed_ATL03_20220325200334_00421501_006_01_gt2l.csv"
# PATH_STR = r".\data\icepyx\GoldenBay\20230929\processed_ATL03_20230929055034_01562107_006_02_gt3l.csv"
# PATH_STR = r".\data\icepyx\GoldenBay\20211126\processed_ATL03_20211126014738_09871301_006_01_gt3l.csv"
PATH_STR = r".\data\icepyx\GoldenBay\20220325\processed_ATL03_20220325200334_00421501_006_01_gt2l.csv"


BLOCK_SIZE = 10  # unit: m
BLOCK_MIN = 2


data_path = (
    Path(PATH_STR)
    if Path(PATH_STR).exists()
    else Path(
        filedialog.askopenfilename(
            title="选择phoreal处理过的csv数据",
            filetypes=[("所有文件", "*.*"), ("CSV文件", "*.csv")],
        )
    )
)

if data_path:
    logger.info(f"选择的文件: {data_path}")
else:
    logger.error("未选择任何文件")
    exit()

new_path = shutil.copy2(data_path, Path.cwd() / data_path.name)
logger.info("Copy file to current directory:")

with open(new_path, "r", encoding="utf-8") as f:
    df = pd.read_csv(f)

logger.info(df.columns)

df["point_type"] = PointType.NotClassified.value
get_plt(df, "Original Data")
get_plt(
    df, "confidence coefficient", type_lable=ICESAT2Properties.SignalConfidence.value
)

df = df.loc[df[ICESAT2Properties.SignalConfidence.value] > 0]

####################################################################################################
####################################################################################################

sea_level, water_surface = get_sea_level(df, n_sigmas=0.8)

logger.info(
    f"sea_level: {sea_level}, limit: {float(water_surface[0])},{float(water_surface[1])}"
)

df.loc[(df["Height (m MSL)"] >= water_surface[1]), "point_type"] = PointType.Noise.value

df.loc[(df["Height (m MSL)"] < water_surface[1]), "point_type"] = (
    PointType.WaterSurface.value
)

df.loc[(df["Height (m MSL)"] < water_surface[0]), "point_type"] = (
    PointType.UnderWater.value
)

# 筛选出高度低于水面的数据
underwater_mask = df["Height (m MSL)"] < water_surface[0]

# 获取需要调整高度的数据列
under_water_point = df.loc[underwater_mask, "Height (m MSL)"]

# 调整高度
real_depth = under_water_point.apply(get_real_depth, args=(sea_level,))

# 将调整后的高度赋值回原数据框
df.loc[underwater_mask, "Height (m MSL)"] = sea_level - real_depth

get_plt(df=df, title="Water Surface Distinctio")

####################################################################################################
####################################################################################################

get_plt(df=df, title="Set Limit", async_plt=True)


limits = input("input limit please.(split with ','): ").split(",")

if len(limits) == 4:
    xlim = [
        float(limits[0]),
        float(limits[1]),
    ]
    ylim = [
        float(limits[2]),
        float(limits[3]),
    ]

    logger.info(f"xlim: {xlim}, ylim: {ylim}")

    df = df[
        (df[ICESAT2Properties.AlongTrack.value] >= xlim[0])
        & (df[ICESAT2Properties.AlongTrack.value] <= xlim[1])
        & (df[ICESAT2Properties.Height_MSL.value] >= ylim[0])
        & (df[ICESAT2Properties.Height_MSL.value] <= ylim[1])
    ]

get_plt(df=df, title="Use Limit")

####################################################################################################
####################################################################################################

# optics_clustering_denoise
def optics_clustering():
    under_water_point = df.loc[underwater_mask].copy()
    noise_points = optics_clustering_denoise(
        under_water_points=under_water_point,
        min_samples=2,
        xi=0.01,
        min_cluster_size=0.01,
        threshold=5,
    )
    original_indices = under_water_point.index[noise_points]
    df.loc[original_indices, "point_type"] = PointType.Noise.value

    land_points = np.logical_not(noise_points)
    original_indices = under_water_point.index[land_points]
    df.loc[original_indices, "point_type"] = PointType.Valid.value

    get_plt(df.loc[underwater_mask], "Optics Clustering Denoise")

    not_noise_points = df.loc[df["point_type"] != PointType.Noise.value]

    not_noise_points["block"] = not_noise_points[ICESAT2Properties.AlongTrack.value] // 50
    unique_blocks = not_noise_points["block"].unique()
    for i in unique_blocks:
        above_water_points_num = not_noise_points[
            (not_noise_points["block"] == i)
            & (not_noise_points["Height (m MSL)"] >= water_surface[0])
        ].shape[0]
        under_water_points_num = not_noise_points[
            (not_noise_points["block"] == i)
            & (not_noise_points["Height (m MSL)"] < water_surface[0])
        ].shape[0]
        min_under_water_point = not_noise_points[not_noise_points["block"] == i][
            "Height (m MSL)"
        ].min()
        if (
            1 < under_water_points_num
            and under_water_points_num < 25
            and min_under_water_point > -3
        ):
            logger.info(f"区块 {i} 可能是陆地")
            not_noise_points.loc[
                (not_noise_points["block"] == i)
                & (not_noise_points["Height (m MSL)"] >= water_surface[0]),
                "point_type",
            ] = PointType.Valid.value

    get_plt(not_noise_points, title="Underwater Effective Points")

# optics_clustering()
df.loc[(df["point_type"] != PointType.Noise.value), "point_type"] = PointType.Valid.value

####################################################################################################
####################################################################################################
# rasterization
####################################################################################################
####################################################################################################

data = df[
    [
        "Latitude (deg)",
        "Longitude (deg)",
        "UTM Easting (m)",
        "UTM Northing (m)",
        "Cross-Track (m)",
        "Along-Track (m)",
        "Height (m HAE)",
        "Height (m MSL)",
        "Solar Elevation (deg)",
        "point_type",
    ]
]

under_water_point = data.loc[data["point_type"] == PointType.Valid.value]

block = {}

# 按照block_size进行分块
for index, point in under_water_point.iterrows():
    block_id = int(point["Along-Track (m)"] // BLOCK_SIZE)
    if block_id not in block:
        block[block_id] = []
    block[block_id].append(point)

new_ds = []
available_keys = sorted(list(block.keys()))
for block_id in available_keys:
    ds = pd.DataFrame(block[block_id])
    curren_id = block_id + 1
    # id + 1 的块是否存在
    if curren_id in available_keys:
        # 如果块是连续的则扩展窗口
        ds = pd.concat([ds, pd.DataFrame(block[curren_id])])

    ds["point_type"] = PointType.Submarine.value
    heights = ds["Height (m MSL)"].values
    mu, sigma = get_normal_distribution(heights)
    if sigma > 1:
        ds.loc[
            (ds["Height (m MSL)"] < mu - 0.5 * sigma)
            | (ds["Height (m MSL)"] > mu + 0.5 * sigma),
            "point_type",
        ] = PointType.Noise.value
    new_ds.append(ds)

seafloor_point = pd.concat(new_ds)

get_plt(seafloor_point, title="underwater point")

blocks = []
for id in block.keys():
    ds = pd.DataFrame(block[id])
    ds = seafloor_point.loc[seafloor_point["point_type"] == PointType.Submarine.value]

    # icesat-2的激光点间隔是0.7m
    # 如果block的点数小于一定值，说明这个block是无效的
    if len(ds) < BLOCK_MIN:
        continue
    else:
        blocks.append(id)

logger.info(f"有效的block数量: {len(blocks)}")

bs = []
tmp = []
for id in blocks:
    if len(tmp) < 1:
        tmp.append(id)
        continue

    if id - tmp[-1] == 1:
        tmp.append(id)
    else:
        bs.append(tmp)
        tmp = []
        tmp.append(id)

    if blocks.index(id) == len(blocks) - 1:
        bs.append(tmp)

# 拼接近似连续的block
for idx, b in enumerate(bs):
    if idx == 0:
        continue
    if bs[idx][0] - bs[idx - 1][-1] == 2:
        if (bs[idx - 1][-1] + 1) in block.keys():
            bs[idx] = bs[idx - 1] + [bs[idx - 1][-1] + 1] + bs[idx]

if len(blocks) > 1:
    select_ids = max(bs, key=len)
else:
    select_ids = blocks
logger.info(f"最大连续block的长度: {len(select_ids)}")
logger.info(f"最大连续block: {select_ids[:5]}...")

key_points = pd.DataFrame(columns=data.columns)
key_points_list = []
for id in select_ids:
    ds = pd.DataFrame(block[id])
    kp = get_keypoint(ds).to_frame().T
    key_points_list.append(kp)
key_points = pd.concat(key_points_list, ignore_index=True)

logger.info(
    "key_points Along-Track (m):\n"
    + key_points["Along-Track (m)"][:3].to_string(index=False)
)
logger.info(f"key_points数量: {len(key_points)}")

ds = key_points[
    [
        "Latitude (deg)",
        "Longitude (deg)",
        "UTM Easting (m)",
        "UTM Northing (m)",
        "Height (m MSL)",
        "Height (m HAE)",
        ICESAT2Properties.AlongTrack.value,
    ]
]

get_plt(
    key_points,
    title="Topographic cross-section",
    curve=True,
    straight=True,
    k=2,
    bc_type="not-a-knot",
    interpolations=10000,
)

####################################################################################################
####################################################################################################

def get_raster_data():
    s2a_path = Path(
        r".\data\sentinel-2\subset_1_of_S2A_MSIL2A_20250106T031121_N0511_R075_T49QCD_20250106T061847_s2resampled.tif"
    )

    s2a_wgs84_path = Path(
        r".\data\sentinel-2\subset_1_of_S2A_MSIL2A_20250106T031121_N0511_R075_T49QCD_20250106T061847_s2resampled_wgs84.tif"
    )

    # reproject the raster to WGS84
    try:
        reproject2(s2a_path, s2a_wgs84_path, epsg=4326)
    except FileExistsError:
        logger.info("target already exists. Skipping reprojection.")

    raster_data = None
    with rasterio.open(s2a_wgs84_path, mode="r") as src:
        raster_data = get_value_from_raster(
            src, ds["Longitude (deg)"].values, ds["Latitude (deg)"].values, index=0
        )
    
    return raster_data

raster_data = get_raster_data()

####################################################################################################
####################################################################################################

dem_path = Path("data/dem/GMRT_resample.tif")

real_height = []
with rasterio.open(dem_path, mode="r") as src:
    values = get_value_from_raster(
        src, ds["Longitude (deg)"].values, ds["Latitude (deg)"].values, index=1
    )
    real_height.extend(values)

####################################################################################################
####################################################################################################

add_data = {}

for i in range(raster_data.shape[0]):
    "第9波段是8A，没有第10波段"
    i = i + 1
    if i < 9:
        band_num = str(i)
    elif i == 9:
        band_num = "8A"
    elif i == 10:
        band_num = "9"
    elif i > 10:
        band_num = str(i)
    else:
        raise ValueError("Unexpected band index")

    add_data["B" + band_num] = raster_data[i - 1, :]

add_data["real_height"] = real_height

logger.info(f"add_data index:{list(add_data.keys())}")

ml_data = ds.assign(**add_data)
logger.info(ml_data.columns)

dp = Path(new_path)
save_csv(ml_data, data_path=dp, tag="interpolation", backup=True, overwrite=True)

input("Press Enter to exit...")