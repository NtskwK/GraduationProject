from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import Affine
from loguru import logger

from utils.property import ICESAT2Properties

# 读取数据
input_csv = Path("kriging.csv")
output_tif = Path("kriging_results.tif")

expand_factor = 2

logger.info(f"读取输入CSV文件: {input_csv}")
data = pd.read_csv(input_csv)
lables = [
    ICESAT2Properties.Latitude.value,
    ICESAT2Properties.Longitude.value,
    ICESAT2Properties.Height_MSL.value,
]

logger.info("检查并删除缺失值...")
for label in lables:
    data = data[data[label].notna()]

x = data["Longitude (deg)"].values
y = data["Latitude (deg)"].values
z = data["Height (m MSL)"].values

logger.info(f"数据点数量: {len(x)}")
logger.info("开始执行格网插值...")
# 计算原始数据范围
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# 计算扩展后的范围
x_range = x_max - x_min
y_range = y_max - y_min

x_min_expanded = x_min - (expand_factor - 1) * x_range / 2
x_max_expanded = x_max + (expand_factor - 1) * x_range / 2
y_min_expanded = y_min - (expand_factor - 1) * y_range / 2
y_max_expanded = y_max + (expand_factor - 1) * y_range / 2

# 构建扩展后的网格
grid_lon = np.linspace(x_min_expanded, x_max_expanded, 1000)
grid_lat = np.linspace(y_min_expanded, y_max_expanded, 1000)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

# 执行普通克里金插值
logger.info("开始执行普通克里金插值...")
OK = OrdinaryKriging(
    x,
    y,
    z,
    variogram_model="spherical",  # 可选用'linear'、'exponential'、'gaussian'等
    verbose=False,
    enable_plotting=False,
)

logger.info("执行插值...")
z_interp, ss = OK.execute("grid", grid_lon, grid_lat)

# 创建GeoTIFF文件（地理参考信息已更新）
transform = Affine.translation(
    grid_lon[0] - (grid_lon[1] - grid_lon[0]) / 2,
    grid_lat[-1] + (grid_lat[1] - grid_lat[0]) / 2,
) * Affine.scale(grid_lon[1] - grid_lon[0], -(grid_lat[1] - grid_lat[0]))

with rasterio.open(
    output_tif,
    "w",
    driver="GTiff",
    height=z_interp.shape[0],
    width=z_interp.shape[1],
    count=1,
    dtype=z_interp.dtype,
    crs="EPSG:4326",  # WGS 84坐标系
    transform=transform,
) as dst:
    dst.write(z_interp, 1)

print(f"GeoTIFF文件已保存至: {output_tif}")

# 可视化插值结果（带扩展区域）
plt.figure(figsize=(12, 10))
contour = plt.contourf(grid_x, grid_y, z_interp, levels=50, cmap="jet", extend="both")
cbar = plt.colorbar(contour)
cbar.set_label("Height (m MSL)")

# 标记原始数据点范围
# plt.scatter(x, y, c=z, cmap='jet', edgecolor='k', s=50, zorder=5)
plt.title("Kriging interpolation results", fontsize=16)
plt.xlabel("Longitude (deg)", fontsize=14)
plt.ylabel("Latitude (deg)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("kriging_interpolation_expanded.png", dpi=300)
plt.show()
