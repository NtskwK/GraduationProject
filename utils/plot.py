import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from .property import ICESAT2Properties

def get_plt(
    df: pd.DataFrame,
    title: str,
    x_lable: str = ICESAT2Properties.AlongTrack.value,
    y_lable: str = ICESAT2Properties.Height_MSL.value,
    type_lable: str = "point_type",
    straight: bool = False,
    curve: bool = False,
    interpolations: int = 1000,
    k: int = 3,
    bc_type: str = "clamped",
    async_plt: bool = True,
):
    distance_range = df[y_lable].max() - df[y_lable].min()


    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.ylim(df[y_lable].min() - distance_range * 0.1, df[y_lable].max() + distance_range * 0.1)

    if type_lable in df.columns:
        for point_type in df[type_lable].unique():
            ax.scatter(
                x=df[df[type_lable] == point_type][x_lable],
                y=df[df[type_lable] == point_type][y_lable],
                s=1,
                label=point_type,
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
