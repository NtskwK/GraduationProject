import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_plt(
    df: pd.DataFrame,
    x: str = "Along-Track (m)",
    y: str = "Height (m MSL)",
    title: str = None,
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
    plt.show()
    return fig, ax