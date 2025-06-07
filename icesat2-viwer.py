import argparse
import pprint
from pathlib import Path

import pandas
import matplotlib.pyplot as plt

from utils.plot import get_plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

class icesat2data:

    path_str = Path(
        r"E:\Documents\CourseStudy\GraduationProject\program\data\icepyx\GoldenBay\20221230\processed_ATL03_20221230185202_01561807_006_02_gt3r.csv")
    df = None
    xlim = None
    ylim = None

    def __init__(self, path_str=None):

        if path_str is not None:
            self.path_str = Path(path_str)

        assert Path(self.path_str).exists()
        with open(self.path_str, 'r',encoding="utf-8") as f:
            self.df = pandas.read_csv(f)

        print(f"Open File:\n{self.path_str}\n")
        print(f"Data ID:\n{self.path_str.stem}\n")

    def show_info(self):
        pprint.pprint(
            {"path": str(self.path_str), "xlim": self.xlim, "ylim": self.ylim},
            indent=4)

    def save_img(self, tag: str):
        self.show_info()
        confirm = input(f"是否保存\"{tag}\"图片？(yes/[no])").lower()[:1]
        if not "y" == confirm:
            print("图片未保存")
            return

        fn = self.path_str.stem + "_" + tag + ".png"
        plt.savefig(fn)
        print(f"图片已保存{fn}")

    def plot(self):
        # 画图
        # self.df.plot(x='Latitude (deg)', y='Height (m MSL)',
        #              kind='scatter', s=1)
        # if self.xlim is not None:
        #     plt.xlim(self.xlim)
        # if self.ylim is not None:
        #     plt.ylim(self.ylim)

        # plt.title(self.path_str.name)
        # plt.show()
        # self.save_img(tag="plot")
        get_plt(self.df, title=self.path_str.stem)


    def hist(self):
        """
        todo:
            fix it
        """
        plt.xlim(None)
        plt.ylim(None)

        df = self.df

        if self.xlim is not None:
            df = self.df[self.df['Height (m MSL)'] > self.xlim[0]
                         and self.df['Height (m MSL)'] < self.xlim[1]]
        if self.ylim is not None:
            df = self.df[self.df['Latitude (deg)'] > self.ylim[0]
                         and self.df['Latitude (deg)'] < self.ylim[1]]

        # 绘制频率直方图
        plt.hist(df['Height (m MSL)'], bins="auto", edgecolor='black')
        plt.title('Frequency Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

        self.save_img(tag="hist")
        df = None

    def cut(self):
        xlim = [float(x) for x in input("请输入x轴范围，用空格分隔：").split()]
        ylim = [float(y) for y in input("请输入y轴范围，用空格分隔：").split()]

        if len(xlim) == 2:
            self.xlim = xlim
        if len(ylim) == 2:
            self.ylim = ylim

        print("已保存修改范围")
        self.show_info()

    def save_cut(self):
        if self.xlim is None and self.ylim is None:
            print("没有修改范围")
            return

        self.show_info()
        if not "y" == input("是否保存修改范围？(yes/[no])").lower()[:1]:
            print("未保存修改范围")
            return

        if self.xlim is not None:
            self.df = self.df[(self.df['Latitude (deg)'] > self.xlim[0]) & (
                self.df['Latitude (deg)'] < self.xlim[1])]

        if self.ylim is not None:
            self.df = self.df[(self.df['Height (m MSL)'] > self.ylim[0]) & (
                self.df['Height (m MSL)'] < self.ylim[1])]
    

        fn = self.path_str.stem + "_cut.csv"

        self.df.to_csv(fn, index=True)
        print(f"已保存修改范围到{fn}")


def main(fp: str, show: bool = False):
    data = icesat2data(fp)

    if show:
        data.plot()
        exit()

    while True:
        print("""
请选择操作：
1. 画图
2. 直方图
3. 修改范围
4. 保存数据切片
5. 退出
              """)
        choice = input("请输入操作编号：")
        if choice == "1":
            data.plot()
        elif choice == "2":
            data.hist()
        elif choice == "3":
            data.cut()
            data.plot()
        elif choice == "4":
            data.save_cut()
            data.plot()
        elif choice == "5":
            break

if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description="ICESAT2数据可视化工具")

    # 添加参数
    parser.add_argument("--FilePath", type=str, help="文件路径",
                        default=None)
    parser.add_argument("--show", action="store_true", default=False, help="显示图像")

    args = parser.parse_args()

    main(args.FilePath, args.show)
