import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from utils.denoise import get_sea_level
from utils.icesat2_helper import get_beams, is_h5_valid

def atl03_quicklook(filepath: str, save_img=True, show_img=False) -> str:
    plt.clf()
    file = Path(filepath)

    if is_h5_valid(file):
        print(f"File {file} is valid")

    atl03 = get_beams(file)

    if len(atl03) == 0:
        return None

    fname = Path(file).stem
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    for x, y in itertools.product([1, 2, 3], ["l", "r"]):
        if not f"gt{x}{y}" in atl03.keys():
            continue

        df = pd.DataFrame(
            {
                "lat_ph": atl03[f"gt{x}{y}"]["heights"]["lat_ph"],
                "lon_ph": atl03[f"gt{x}{y}"]["heights"]["lon_ph"],
                "h_ph": atl03[f"gt{x}{y}"]["heights"]["h_ph"],
            }
        )

        sea_level, sea_range = get_sea_level(df, "h_ph")
        print(f"sea level: {sea_level}, range: {sea_range}")

        df = df[(df["h_ph"] > sea_level - 10) & (df["h_ph"] < sea_level + 5)]

        # plt.subplot(3,2,int(x)*2 + int(y=="l") - 1 )
        # plt.plot(df["lat_ph"],df["h_ph"],'.',s=0.7)

        df.plot(
            x="lat_ph",
            y="h_ph",
            kind="scatter",
            s=1,
            ax=axs[int(x) - 1, int(y == "l")],
            title=f"gt{x}{y}",
        )

        # Adjust single image
        if x == 1 and y == "r":
            pass
        elif x == 1 and y == "l":
            pass
        elif x == 2 and y == "r":
            pass
        elif x == 2 and y == "l":
            pass
        elif x == 3 and y == "r":
            pass
        elif x == 3 and y == "l":
            pass

    plt.suptitle(fname)
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5
    )

    if show_img:
        plt.show()

    if not save_img:
        return None

    fn = Path(file).stem + ".png"
    img_path = Path(file).parent.joinpath(fn)
    print(f"try to save to {img_path}")
    plt.savefig(img_path)
    print(f"save to {img_path}")

    return img_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Quicklook ICESat-2 ATL03 data")

    parser.add_argument("--File", type=str, help="Path to the ATL03 HDF5 file")
    parser.add_argument("--Show", action="store_true", help="Show the image")
    parser.add_argument("--Save", action="store_true", help="Save the image")

    args = parser.parse_args()

    atl03_quicklook(args.File, args.Show, args.Save)
