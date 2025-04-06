from pathlib import Path

from tools.quicklook import atl03_quicklook
from utils.icesat2_helper import export_h5_to_csv

dir = r"E:\Documents\课程学习\GraduationProject\program\data\icepyx\GoldenBay"
dir = Path(dir)

for file in dir.rglob("*.h5"):
    # atl03_quicklook(file)
    atl03_quicklook(file, show_img=True)
    export_h5_to_csv(file)
