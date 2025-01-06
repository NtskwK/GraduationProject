from pathlib import Path

dir = r"E:\Documents\课程学习\GraduationProject\program\data\icepyx\GoldenBay"
dir = Path(dir)

for file in dir.glob("*.h5"):
    fn = file.stem
    date = fn.split("_")[2][:8]

    new_dir = dir.joinpath(date)
    new_dir.mkdir(exist_ok=True)
    file.rename(new_dir.joinpath(file.name))