import math
import os
import argparse
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, OPTICS

from utils.data_io import save2dir, save_figure


backup_dir = Path(os.getcwd()) / "log"


class PointType(Enum):
    Noise = -1
    NotClassified = 0
    WaterSurface = 1
    LandSurface = 2


def get_normal_distribution(dataX: np.ndarray, n_sigmas: float = 0.5, n: int = 0):
    """
    # 获得正态分布

    计算n次，每次剔除n_sigmas倍标准差以外的点
    """
    mu, sigma = stats.norm.fit(dataX)
    if n > 0:
        n = n - 1
        dataX = dataX[(dataX > mu - n_sigmas * sigma) & (dataX < mu + n_sigmas * sigma)]
        return get_normal_distribution(dataX, n_sigmas, n)
    return mu, sigma


def get_sea_level(df: pd.DataFrame, index: str = "Height (m MSL)", n_sigmas=0.5):
    """
    # 获取海平面高度

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


def get_water_surface_points(
    sea_level: float,
    sea_range: tuple,
    df: pd.DataFrame,
    index: str = "Height (m MSL)"
) -> pd.DataFrame:
    if sea_level is None or sea_range is None:
        sea_level, sea_range = get_sea_level(df, index)

    sea_surface_points = df[(df[index] >= sea_range[0]) & (df[index] <= sea_range[1])]

    return sea_surface_points


# 水深校正
def get_real_depth(height, water_level) -> float:
    # input: Uncorrected water depth
    # output: Corrected water depth
    water_depth = height - water_level

    # /gt3l/geolocation/ref_elev
    theta_2 = 1.56387

    theta_1 = math.pi / 2 - theta_2

    phi = theta_1 - theta_2
    ideal_optical_path = water_depth / math.cos(theta_1)
    actual_optical_path = ideal_optical_path * (1.00029 / 1.33469)
    straight = math.sqrt(
        ideal_optical_path**2
        + actual_optical_path**2
        + 2 * actual_optical_path * ideal_optical_path * math.cos(phi)
    )
    gamma = math.pi / 2 - theta_1
    alpha = math.asin(actual_optical_path * math.sin(gamma) / straight)
    beta = gamma - alpha
    delta_x = straight * math.cos(beta)
    delta_z = straight * math.sin(beta)
    real_deep = delta_z - height
    return real_deep


def local_domain_distance(
    df: pd.DataFrame,
    x_label: str = "Along-Track (m)",
    y_label: str = "Height (m MSL)",
    k: int = 3,
    mean: bool = True,
) -> np.ndarray:
    """
    获取最近的k个点的距离

    Args:
        df (pd.DataFrame): 数据
        x_label (str, optional): x轴标签. Defaults to "Along-Track (m)".
        y_label (str, optional): y轴标签. Defaults to "Height (m MSL)".
        k (int, optional): 最近邻点数. Defaults to 3.
        mean (bool, optional): 是否取平均值. Defaults to True.
    Outputs:
        distances (np.array): 平均距离
    """
    X = df[x_label].values
    h = df[y_label].values

    # 建立距离矩阵
    dist_matrix = np.sqrt((X[:, np.newaxis] - X) ** 2 + (h[:, np.newaxis] - h) ** 2)
    # 排除自身距离（对角线）
    # np.fill_diagonal(dist_matrix, np.nan)

    # 排除掉最近的元素（其本身），从第二位开始，取k个元素
    k_nearest_distances = np.partition(dist_matrix, k, axis=1)[:, 1 : k + 1]
    # 在忽略无效值的前提下计算平均值
    if mean:
        result = np.nanmean(k_nearest_distances, axis=1)
    else:
        result = np.nanmax(k_nearest_distances, axis=1)

    return result


def evaluating_denoise_results(is_noise: np.ndarray) -> None:
    print(f"样本数量：{len(is_noise)}")
    print(f"有效点数量：{sum(is_noise == False)}")
    print(f"噪声点数量：{sum(is_noise)}")
    print(f"有效点比例：{sum(is_noise == False) / len(is_noise)}")
    print(f"噪声点比例：{sum(is_noise) / len(is_noise)}")


def optics_clustering_denoise(
    under_water_points: pd.DataFrame,
    x_lable: str = "Along-Track (m)",
    y_lable: str = "Height (m MSL)",
    min_samples: int = 5,
    xi: float = 0.5,
    min_cluster_size: float = 0.05,
    threshold: float = 4.0,
) -> np.ndarray:
    """_summary_

    Args:
        under_water_points (pd.DataFrame): 水下点云数据
        x_lable (str, optional): Defaults to "Along-Track (m)".
        y_lable (str, optional): Defaults to "Height (m MSL)".
        min_samples (int, optional): 核的邻域内点的数量. Defaults to 5.
        xi (float, optional): 簇的紧凑程度. Defaults to 0.5.
        min_cluster_size (float, optional): 簇的最小大小. Defaults to 0.05.
        threshold (float, optional): 预测可达距离. Defaults to 4.0.

    Returns:
        is_noise(np.ndarray):
    """
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    optics.fit(under_water_points[[x_lable, y_lable]])
    # 获取样本的可达性距离
    reachability = optics.reachability_.copy()

    # 获取样本的排序索引
    ordering = np.argsort(optics.ordering_)

    # 根据可达性距离和排序索引来判断噪声点
    noise_points = [i for i in ordering if reachability[i] > threshold]

    # 获取聚类标签
    labels = optics.labels_
    print(f"样本数量：{len(reachability)}")
    print(f"类型数量：{len(set(labels))}")
    print(f"噪声点数量：{len(noise_points)}")
    print(f"噪声点比例：{len(noise_points) / len(reachability)}")

    # 创建一个布尔数组，标记噪声点
    is_noise = np.zeros(len(under_water_points), dtype=bool)
    is_noise[noise_points] = True

    evaluating_denoise_results(is_noise)

    return is_noise


def dbscan_denoise(
    under_water_points: pd.DataFrame,
    x_lable: str = "Along-Track (m)",
    y_lable: str = "Height (m MSL)",
    eps: float = 0.3,
    min_samples: int = 5,
) -> np.ndarray:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(under_water_points[[x_lable, y_lable]])

    # 获取聚类标签
    labels = dbscan.labels_

    # 创建一个布尔数组，标记噪声点
    is_noise = labels == -1

    evaluating_denoise_results(is_noise)

    return is_noise


def adaptive_elliptical_denoise(
    under_water_points: pd.DataFrame,
    x_label: str = "Along-Track (m)",
    y_label: str = "Height (m MSL)",
    k: int = 10,  # 近邻点数
    alpha: float = 2.0,  # 椭圆缩放因子
) -> np.ndarray:
    """
    **这个算法的实现是有问题的，不要用！**
    """
    # 提取所需的点云数据
    points = under_water_points[[x_label, y_label]].values
    num_points = len(points)
    is_noise = np.ones(num_points, dtype=bool)

    # 一次性计算所有点之间的欧氏距离
    distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    # 获取每个点的k近邻点的索引
    k_nearest_indices = np.argsort(distances, axis=1)[:, 1 : k + 1]

    for i in range(num_points):
        # 获取当前点的k近邻点
        k_nearest_points = points[k_nearest_indices[i]]

        # 计算k近邻点的协方差矩阵
        cov_matrix = np.cov(k_nearest_points, rowvar=False)

        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 确保特征值为实数
        eigenvalues = np.real(eigenvalues)

        # 对特征值进行排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # 椭圆的长半轴和短半轴
        b = alpha * np.sqrt(eigenvalues[0])
        a = alpha * np.sqrt(eigenvalues[1])

        # 计算所有点到当前点的向量
        vectors = points - points[i]

        # 将所有向量转换到特征向量基下
        transformed_vectors = np.dot(vectors, eigenvectors)

        # 判断所有点是否在椭圆内
        in_ellipse = (transformed_vectors[:, 0] ** 2 / a**2) + (
            transformed_vectors[:, 1] ** 2 / b**2
        ) < 1

        # 更新当前点的噪声标记
        if all(in_ellipse):
            is_noise[i] = False

    print(f"样本数量：{num_points}")
    print(f"噪声点数量：{sum(is_noise)}")
    print(f"噪声点比例：{sum(is_noise) / num_points}")

    return is_noise


def main(path_str):
    # 检查文件是否存在
    data_path = Path(path_str)
    assert data_path.exists(), "文件不存在"

    df = None

    with open(data_path, "r", encoding="utf-8") as f:
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

    underwater_points = df[df["SignalType"] == 2]
    underwater_points["adjust_depth"] = underwater_points["Height (m MSL)"].apply(
        get_real_depth, args=(sea_level,)
    )
    underwater_points["real_height"] = sea_level - underwater_points["adjust_depth"]
    underwater_points["LocalDomainDistance"] = local_domain_distance(
        underwater_points, k=1
    )

    # # 领域搜索频率图
    # plt.clf()
    # plt.hist(underwater_points["LocalDomainDistance"], bins="auto")
    # plt.xlabel("LocalDomainDistance")
    # plt.ylabel("Frequency")
    # plt.title("LocalDomainDistance Histogram")
    # # 图例
    # plt.legend

    # fig = plt.gcf()
    # fig.show()
    # save_figure(fig, Path("local_domain_distance.png"), backup=True, target=backup_dir)

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
    # # 未分类
    # df[df["SignalType"] == 0].plot(
    #     x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="gray", ax=ax
    # )
    # # 水面点
    # df[df["SignalType"] == 1].plot(
    #     x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="blue", ax=ax
    # )
    # # 水上噪声
    # df[df["SignalType"] == -1].plot(
    #     x="Along-Track (m)", y="Height (m MSL)", kind="scatter", s=0.5, c="red", ax=ax
    # )
    # 水下点
    underwater_points[underwater_points["SignalType"] == 2].plot(
        # x="Along-Track (m)", y="real_height", kind="scatter", s=0.5, c="green", ax=ax
        x="Along-Track (m)",
        y="Height (m MSL)",
        kind="scatter",
        s=0.5,
        c="green",
        ax=ax,
    )
    # 水下噪声
    underwater_points[underwater_points["SignalType"] == -2].plot(
        x="Along-Track (m)",
        # y="real_height",
        y="Height (m MSL)",
        kind="scatter",
        s=0.5,
        c="yellow",
        ax=ax,
    )
    # 添加图例
    plt.legend(loc="best")

    fig = plt.gcf()
    plt.show()

    img_path = Path("classified.png")
    save_figure(fig, img_path, backup=True, target=backup_dir)

    file_name = data_path.stem + "_classified" + ".csv"
    df.to_csv(file_name, index=False)

    # underwater_points.to_csv(file_name + "_underwater_points.csv", index=False)


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
