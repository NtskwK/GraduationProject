import pickle
import pprint

from pathlib import Path

import pandas
import matplotlib
import matplotlib.pyplot as plt

import h5py

while True:
    path_str = input("请输入文件路径：")
    data_path = Path(path_str)
    if not data_path.exists():
        print("文件不存在，请重新输入")

    # 读取csv文件
    with open(data_path, 'r') as f:
        df = pandas.read_csv(f)
        # 画图
        df.plot(x='Latitude (deg)', y='Height (m MSL)', kind='scatter', s=1)
        
        plt.title(data_path.name)
        plt.show()
        
        if "y" == input("是否保存图片？(yes/[no])").upper()[:1]:
            plt.savefig(data_path.name[:-4] + ".png")
            print("图片已保存")
        else:
            print("图片未保存")