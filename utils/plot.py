import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def get_plt(
    df: pd.DataFrame,
    x: str = "Along-Track (m)",
    y: str = "Height (m MSL)",
    title: str | None = None,
    straight: bool = False,
    curve: bool = False,
    interpolations: int = 1000,
    k: int = 3,
    bc_type: str = "clamped",
):
    fig, ax = plt.subplots()
    df.plot(
        x=x,
        y=y,
        ax=ax,
        kind="scatter",
        s=0.5,
        title=y if title is None else title,
        c="point_type",
        colormap="Set1_r",
    )

    # 直线
    if straight:
        # 紫色
        ax.plot(df[x], df[y], color="purple", linewidth=0.5)

    if curve:
        l = np.linspace(df[x].min(), df[x].max(), interpolations)
        mode = make_interp_spline(df[x], df[y], k=k, bc_type=bc_type)
        y_hat = mode(l)
        # 粉色
        ax.plot(
            l,
            y_hat,
            color="black",
            linewidth=0.5,
        )

    plt.show()
    return fig, ax
