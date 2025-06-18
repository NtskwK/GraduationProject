from pathlib import Path

from tqdm import tqdm
from tools.quicklook import atl03_quicklook
from utils.icesat2_helper import export_h5_to_csv
from loguru import logger

# 该脚本用于递归浏览目标目录内的所有atl03的h5文件并在文件目录生成速览图

dir_str = r".\data\icepyx\GoldenBay"
dir = Path(dir_str)

files = list(dir.rglob("*.h5"))
for file in tqdm(files, desc="Processing files"):
    logger.info(f"Processing file: {file.name}")
    atl03_quicklook(file)
    # atl03_quicklook(file, show_img=True)
    export_h5_to_csv(file)
    tqdm.wrapattr(tqdm, "set_description", f"Processing {file.name}")
