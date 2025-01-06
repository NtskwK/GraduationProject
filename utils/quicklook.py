import argparse
import re
import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from utils.denoise import get_sea_level

# https://www.cnblogs.com/sw-code/p/18161987
def read_hdf5_atl03_beam_h5py(file_path, beam, verbose=False):
    """
    ATL03 原始数据读取
    Args:
        filename (str): h5文件路径
        beam (str): 光束
        verbose (bool): 输出HDF5信息

    Returns:
        返回ATL03光子数据的heights和geolocation信息
    """

    # 打开HDF5文件进行读取
    file_id = h5py.File(file_path, 'r')

    # 输出HDF5文件信息
    if verbose:
        print(file_id.filename)
        print(list(file_id.keys()))
        print(list(file_id['METADATA'].keys()))

    # 为ICESat-2 ATL03变量和属性分配python字典
    atl03_mds = {}

    # 读取文件中每个输入光束
    beams = [k for k in file_id.keys() if bool(re.match('gt\\d[lr]', k))]
    if beam not in beams:
        print('请填入正确的光束代码')
        return

    atl03_mds['heights'] = {}
    atl03_mds['geolocation'] = {}
    atl03_mds['bckgrd_atlas'] = {}

    # -- 获取每个HDF5变量
    # -- ICESat-2 Measurement Group
    try:
        for key, val in file_id[beam]['heights'].items():
            atl03_mds['heights'][key] = val[:]

        # -- ICESat-2 Geolocation Group
        for key, val in file_id[beam]['geolocation'].items():
            atl03_mds['geolocation'][key] = val[:]

        for key, val in file_id[beam]['bckgrd_atlas'].items():
            atl03_mds['bckgrd_atlas'][key] = val[:]
    except KeyError as e:
        print(f"error in beam:{beam}")
        print(f'KeyError: {e}')
        return None

    return atl03_mds


def atl03_quicklook(filepath:str, save_img=True, show_img=False) -> str:
    plt.clf()
    plt.close()
    
    
    file = Path(filepath)
    assert file.exists(), "File does not exist"
    
    # like this: "processed_ATL03_20211126014738_09871301_006_01.h5"
    pattern = r"processed_ATL03_\d{14}_\d{8}_\d{3}_\d{2}\.h5"
    if not re.search(pattern, file.name):
        print("It isn't a atl03.h5 file!")
        return None
    
    atl03 = {}
    
    with h5py.File(file, 'r') as f:
        beams = [k for k in f.keys() if bool(re.match('gt\\d[lr]', k))]
        for beam in beams:
            data = read_hdf5_atl03_beam_h5py(file, beam, verbose=False)
            if data is None:
                beams.remove(beam)
                atl03.pop(beam, None) 
                continue
            atl03[beam] = data 

    if len(atl03) == 0:
        return None

    fname = Path(file).stem
    fig, axs = plt.subplots(3, 2, figsize=(8,12))  
    for x,y in itertools.product([1,2,3],["l","r"]):
        if not f"gt{x}{y}" in atl03.keys(): continue
        
        df = pd.DataFrame({
            "lat_ph":atl03[f"gt{x}{y}"]["heights"]["lat_ph"],
            "lon_ph":atl03[f"gt{x}{y}"]["heights"]["lon_ph"],
            "h_ph":atl03[f"gt{x}{y}"]["heights"]["h_ph"]
        })
        
        sea_level, sea_range = get_sea_level(df,"h_ph")
        print(f"sea level: {sea_level}, range: {sea_range}")
        
        df = df[(df["h_ph"] > sea_level -25) & (df["h_ph"] < sea_level + 5)]
        
        # plt.subplot(3,2,int(x)*2 + int(y=="l") - 1 )
        # plt.plot(df["lat_ph"],df["h_ph"],'.',s=0.7)

        df.plot(x="lat_ph", y="h_ph" ,kind='scatter', s=1, ax=axs[int(x)-1,int(y=="l")],title=f"gt{x}{y}")

        
        # Adjust single image
        if x==1 and y=="r":
            pass
        elif x==1 and y=="l":
            pass
        elif x==2 and y=="r":
            pass
        elif x==2 and y=="l":
            pass
        elif x==3 and y=="r":
            pass
        elif x==3 and y=="l":
            pass
        
    plt.suptitle(fname)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)
    
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

    parser = argparse.ArgumentParser(description='Quicklook ICESat-2 ATL03 data')

    parser.add_argument('--File', type=str, help='Path to the ATL03 HDF5 file')
    parser.add_argument('--Show', action='store_true', help='Show the image')
    parser.add_argument('--Save', action='store_true', help='Save the image')

    args = parser.parse_args()
    
    atl03_quicklook(args.File, args.Show, args.Save)