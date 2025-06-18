import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from typing import Optional

from utils.denoise import PointType

from .property import ICESAT2Properties


def get_plt(
    df: pd.DataFrame,
    title: str,
    x_lable: str = ICESAT2Properties.AlongTrack.value,
    y_lable: str = ICESAT2Properties.Height_MSL.value,
    x_lable_str: Optional[str] = None,
    y_lable_str: Optional[str] = None,
    type_lable: str = "point_type",
    straight: bool = False,
    curve: bool = False,
    interpolations: int = 1000,
    k: int = 3,
    bc_type: str = "clamped",
    async_plt: bool = True,
):
    plt.rcParams["font.family"] = ["Times New Roman","SimHei"]
    
    distance_range = df[y_lable].max() - df[y_lable].min()

    if x_lable_str is None:
        x_lable_str = x_lable
    if y_lable_str is None:
        y_lable_str = y_lable

    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_lable_str)
    plt.ylabel(y_lable_str)
    plt.ylim(df[y_lable].min() - distance_range * 0.1, df[y_lable].max() + distance_range * 0.1)

    def get_color(point_type: str) -> str:
        if point_type == PointType.Noise.value:
            return "orange"
        elif point_type == PointType.Submarine.value:
            return "green"
        elif point_type == PointType.WaterSurface.value:
            return "red"
        elif point_type == PointType.UnderWater.value:
            return "cyan"
        elif point_type == PointType.Valid.value:
            return "purple"
        else:
            return None

    if type_lable in df.columns:
        for point_type in df[type_lable].unique():
            ax.scatter(
                x=df[df[type_lable] == point_type][x_lable],
                y=df[df[type_lable] == point_type][y_lable],
                s=1,
                label=point_type,
                color=get_color(point_type)
            )
    else:
        ax.scatter(
            x=df[x_lable],
            y=df[y_lable],
            s=1
        )

    # 直线
    if straight:
        # 紫色
        ax.plot(df[x_lable], df[y_lable], color="purple", linewidth=0.5)

    if curve:
        l = np.linspace(df[x_lable].min(), df[x_lable].max(), interpolations)
        mode = make_interp_spline(df[x_lable], df[y_lable], k=k, bc_type=bc_type)
        y_hat = mode(l)
        # 粉色
        ax.plot(
            l,
            y_hat,
            color="black",
            linewidth=0.5,
        )
    plt.legend()
    plt.show(block = (not async_plt))
    return fig, ax
