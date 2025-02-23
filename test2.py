import numpy as np

# x = np.array([1, 2, 3])
# h = np.array([1, 2, 3])
# print(x.shape)  # 输出: (3,)

# # 使用 np.newaxis 增加维度
# x_reshaped = x[:, np.newaxis]
# print(x_reshaped.shape)  # 输出: (3, 1)
# print(x_reshaped)

# dist_matrix = np.sqrt((x[:, np.newaxis] - x) ** 2 + (h[:, np.newaxis] - h) ** 2)
# print(dist_matrix)
# k_nearest_distances = np.partition(dist_matrix, 2, axis=1)[:, 1 : 2 + 1]
# print(k_nearest_distances)
# result = np.nanmean(k_nearest_distances, axis=1)
# print(result)

import pandas as pd
import numpy as np

df = pd.read_csv("processed_ATL03_20221230185202_01561807_006_02_gt3r.csv")
print(df.columns)