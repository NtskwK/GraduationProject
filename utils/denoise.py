import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
# from sklearn.cluster import OPTICS


def get_sea_level(df: pd.DataFrame, index: str = "Height (m MSL)"):
    """
    reference:

    https://kns.cnki.net/kcms2/article/abstract?v=2F0E8PSemVMiHlQ-VB7uXDrhTbmBEHKnObtQbKj8I-0N1WBtInpwVYlgS_NzT1cMxRAHxkIYbT-ie7vfyShquZpcOvVf2oT2_inI_xfQRZxtu3JG41aneQpL0wqpInS4aM26WnU3RAhUQAbok4AjqheNlwBbeWMSQADuf4OgZ9ayMd7ha5T3K-efEskmpOWJ&uniplatform=NZKPT&language=CHS
    """

    assert index in df.columns, f"索引 {index} 不在数据中"

    heights = df[index].values

    # 拟合高斯函数
    mu, sigma = np.mean(heights), np.std(heights)
    return mu, (mu - 0.5 * sigma, mu + 0.5 * sigma)


def get_sea_points(df: pd.DataFrame, index: str = "Height (m MSL)", sea_level: float = None, sea_range: tuple = None, n=0):
    if sea_level is None or sea_range is None:
        sea_level, sea_range = get_sea_level(df, index)

    sea_points = df[(df[index] >= sea_range[0]) & (df[index] <= sea_range[1])]

    return sea_points


def local_domain_distance(df: pd.DataFrame, k: int = 5):
    """
    局部邻域距离计算

    Args:
        df (pd.DataFrame): _description_
        index (str, optional): _description_. Defaults to "Height (m MSL)".
    """
    x = df['Along-Track (m)'].values
    h = df['Height (m MSL)'].values

    # 建立距离矩阵
    dist_matrix = np.sqrt((x[:, np.newaxis] - x) **
                          2 + (h[:, np.newaxis] - h) ** 2)
    # 排除自身距离（对角线）
    # np.fill_diagonal(dist_matrix, np.nan)

    # 排除掉最近的元素（其本身），从第二位开始，取k个元素
    k_nearest_distances = np.partition(dist_matrix, k, axis=1)[:, 1:k+1]
    # 在忽略无效值的前提下计算平均值
    result = np.nanmean(k_nearest_distances, axis=1)

    return result


def main(path_str):
    # 检查文件是否存在
    data_path = Path(path_str)
    assert data_path.exists(), "文件不存在"

    df = None

    with open(data_path, 'r') as f:
        df = pd.read_csv(f)

    # 将信号类型初始化为0
    df['SignalType'] = 0

    print(df.columns.tolist())

    # 选择频率最高的深度为参考海平面
    sea_level, sea_range = get_sea_level(df)

    print(f"参考海平面: {sea_level}, 范围: {sea_range}")

    # 水上噪声
    df.loc[df['Height (m MSL)'] > sea_range[1], 'SignalType'] = -1
    
    # 水面点
    df.loc[(df['Height (m MSL)'] <= sea_range[1]) & (
        df['Height (m MSL)'] >= sea_range[0]), 'SignalType'] = 1

    # 水下点
    df.loc[df['Height (m MSL)'] < sea_range[0], 'SignalType'] = 2

    underwater_points = df[df['SignalType'] == 2]
    underwater_points['LocalDomainDistance'] = local_domain_distance(
        underwater_points)

    plt.hist(underwater_points['LocalDomainDistance'], bins="auto")
    plt.xlabel('LocalDomainDistance')
    plt.ylabel('Frequency')
    plt.title('LocalDomainDistance Histogram')
    plt.show()

    mu = np.mean(underwater_points['LocalDomainDistance'])
    sigma = np.std(underwater_points['LocalDomainDistance'])

    limits = (mu - 1 * sigma, mu + 1 * sigma)
    
    underwater_points.loc[underwater_points['LocalDomainDistance'] > limits[1], 'SignalType'] = -2
    
    
    # 创建一个图形对象
    fig, ax = plt.subplots()
    df[df['SignalType'] == 0].plot(x='Along-Track (m)', y='Height (m MSL)', kind='scatter', s=0.5, c='gray', ax=ax)
    df[df['SignalType'] == 1].plot(x='Along-Track (m)', y='Height (m MSL)', kind='scatter', s=0.5, c='blue', ax=ax)
    df[df['SignalType'] == -1].plot(x='Along-Track (m)', y='Height (m MSL)', kind='scatter', s=0.5, c='red', ax=ax)
    underwater_points[underwater_points['SignalType'] == 2].plot(x='Along-Track (m)', y='Height (m MSL)', kind='scatter', s=0.5, c='green', ax=ax)
    underwater_points[underwater_points['SignalType'] == -2].plot(x='Along-Track (m)', y='Height (m MSL)', kind='scatter', s=0.5, c='yellow', ax=ax)
    plt.title('Denoised Data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description="ICESAT2数据去噪工具")

    # 添加参数
    parser.add_argument("--FilePath", type=str, help="文件路径",
                        default="processed_ATL03_20220105120423_02171407_006_01_gt2l_cut.csv"
                        )

    args = parser.parse_args()

    main(args.FilePath)
