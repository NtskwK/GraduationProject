import math
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# from sklearn.cluster import OPTICS

from .backup import save2dir,save_figure

def get_normal_distribution(dataX: np.array, n_sigmas: float = 0.5, n: int = 0):
    mu, sigma = stats.norm.fit(dataX)
    n = n - 1
    if n >= 0:
        dataX = dataX[(dataX > mu - n_sigmas * sigma) & (dataX < mu + n_sigmas * sigma)]
        return get_normal_distribution(dataX, n_sigmas, n)
    return mu, sigma


def get_sea_level(df: pd.DataFrame, index: str = "Height (m MSL)", n_sigmas=0.5):
    """
    reference:

    https://kns.cnki.net/kcms2/article/abstract?v=2F0E8PSemVMiHlQ-VB7uXDrhTbmBEHKnObtQbKj8I-0N1WBtInpwVYlgS_NzT1cMxRAHxkIYbT-ie7vfyShquZpcOvVf2oT2_inI_xfQRZxtu3JG41aneQpL0wqpInS4aM26WnU3RAhUQAbok4AjqheNlwBbeWMSQADuf4OgZ9ayMd7ha5T3K-efEskmpOWJ&uniplatform=NZKPT&language=CHS
    """

    assert index in df.columns, f"索引 {index} 不在数据中"

    heights = df[index].values

    # 拟合高斯函数
    # mu, sigma = np.mean(heights), np.std(heights)
    # mu, sigma = stats.norm.fit(heights)
    mu, sigma = get_normal_distribution(heights, n_sigmas, 1)

    return mu, (mu - n_sigmas * sigma, mu + n_sigmas * sigma)


def get_sea_points(
    df: pd.DataFrame,
    index: str = "Height (m MSL)",
    sea_level: float = None,
    sea_range: tuple = None,
    n=0,
):
    if sea_level is None or sea_range is None:
        sea_level, sea_range = get_sea_level(df, index)

    sea_points = df[(df[index] >= sea_range[0]) & (df[index] <= sea_range[1])]

    return sea_points


def local_domain_distance(df: pd.DataFrame, k: int = 3):
    """
    局部邻域距离计算

    Args:
        df (pd.DataFrame): _description_
        index (str, optional): _description_. Defaults to "Height (m MSL)".
    """
    x = df["Along-Track (m)"].values
    h = df["Height (m MSL)"].values

    # 建立距离矩阵
    dist_matrix = np.sqrt((x[:, np.newaxis] - x) ** 2 + (h[:, np.newaxis] - h) ** 2)
    # 排除自身距离（对角线）
    # np.fill_diagonal(dist_matrix, np.nan)

    # 排除掉最近的元素（其本身），从第二位开始，取k个元素
    k_nearest_distances = np.partition(dist_matrix, k, axis=1)[:, 1 : k + 1]
    # 在忽略无效值的前提下计算平均值
    result = np.nanmean(k_nearest_distances, axis=1)

    return result


def adjust_height_underwater(X):
    theta = math.pi / 2 - X


def main(path_str):
    # 检查文件是否存在
    data_path = Path(path_str)
    assert data_path.exists(), "文件不存在"

    df = None

    with open(data_path, "r") as f:
        df = pd.read_csv(f)

    # 将信号类型初始化为0
    df["SignalType"] = 0

    # print(df.columns.tolist())

    # 数据点频率图
    plt.clf()
    plt.hist(df["Height (m MSL)"], bins="auto")
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.title("Depth Histogram")
    fig = plt.gcf()
    fig.show()
    
    backup_dir = Path(os.getcwd()) / "log"
    save_figure(fig, Path("depth_histogram.png"), backup=True, target=backup_dir)

    # 选择频率最高的深度为参考海平面
    sea_level, sea_range = get_sea_level(df)

    print(f"参考海平面: {sea_level}, 范围: {sea_range}")

    # 水上噪声
    df.loc[df["Height (m MSL)"] > sea_range[1], "SignalType"] = -1

    # 水面点
    df.loc[
        (df["Height (m MSL)"] <= sea_range[1]) & (df["Height (m MSL)"] >= sea_range[0]),
        "SignalType",
    ] = 1

    # 水下点
    df.loc[df["Height (m MSL)"] < sea_range[0], "SignalType"] = 2

    underwater_points["adjust_Height"] = adjust_height_underwater(underwater_points)

    underwater_points = df[df["SignalType"] == 2]
    underwater_points["LocalDomainDistance"] = local_domain_distance(underwater_points)

    # 领域搜索频率图
    plt.clf()
    plt.hist(underwater_points["LocalDomainDistance"], bins="auto")
    plt.xlabel("LocalDomainDistance")
    plt.ylabel("Frequency")
    plt.title("LocalDomainDistance Histogram")
    # 图例
    plt.legend
    
    fig = plt.gcf()
    fig.show()
    save_figure(fig, Path("local_domain_distance.png"), backup=True, target=backup_dir)

    mu = np.mean(underwater_points["LocalDomainDistance"])
    sigma = np.std(underwater_points["LocalDomainDistance"])

    limits = (mu - 1 * sigma, mu + 1 * sigma)

    # 水下噪声
    underwater_points.loc[
        underwater_points["LocalDomainDistance"] > limits[1], "SignalType"
    ] = -2

    # 创建一个图形对象
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title("Classified Data")
    # 未分类
    df[df["SignalType"] == 0].plot(
        x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="gray", ax=ax
    )
    # 水面点
    df[df["SignalType"] == 1].plot(
        x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="blue", ax=ax
    )
    # 水上噪声
    df[df["SignalType"] == -1].plot(
        x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="red", ax=ax
    )
    # 水下点
    underwater_points[underwater_points["SignalType"] == 2].plot(
        x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="green", ax=ax
    )
    # 水下噪声
    underwater_points[underwater_points["SignalType"] == -2].plot(
        x="Along-Track (m)",
        y="Height (m MSL)",
        kind="scatter",
        s=0.5,
        c="yellow",
        ax=ax,
    )
    # 添加图例
    ax.legend()
    
    fig = plt.gcf()
    fig.legend()
    fig.show()
    
    img_path = Path("classified.png")
    save_figure(fig, img_path, backup=True, target=backup_dir)
    
    file_name = data_path.stem + "_classified" + ".csv"
    df.to_csv(file_name, index=False)

    underwater_points.to_csv(file_name + "_underwater_points.csv", index=False)


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="ICESAT2数据去噪工具")

    # 添加参数
    parser.add_argument(
        "--FilePath",
        type=str,
        help="文件路径",
        default="processed_ATL03_20220105120423_02171407_006_01_gt2l_cut.csv",
    )

    args = parser.parse_args()

    main(args.FilePath)
