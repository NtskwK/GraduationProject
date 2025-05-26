import pandas as pd


def see_list(name: str, lst: list):
    """
    Print a list with each element on a new line.
    """
    print(f"{name}中元素的个数:{len(lst)}")
    print(f"{name}中元素的值:")
    for i in lst:
        print(i)


def get_keypoint(ds: pd.DataFrame, sort_by: str = "Height (m MSL)") -> pd.Series:
    """
    Get the key points from the dataset.
    :param ds: The dataset to get the key points from.
    :param sort_by: The column to sort by.
    :return: The key points.
    """
    # Sort the dataset by the specified column
    sorted_ds = ds.sort_values(by=[sort_by])
    # Select the middle point
    if len(sorted_ds) == 1:
        return sorted_ds.iloc[0]
    elif len(sorted_ds) % 2 == 0:
        return sorted_ds.iloc[len(sorted_ds) // 2]
    else:
        return sorted_ds.iloc[len(sorted_ds) // 2 + 1]
