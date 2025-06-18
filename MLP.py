import pandas as pd

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from d2l import torch as d2l

# 读取数据
df = pd.read_csv("dataset_train.csv")

# 特征和标签
X_cols = [
    "Height (m MSL)",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
]
Y_col = "real_height"

batch_size=16
num_epochs = 50
lr = 0.001
acc_diff = 0.1

# 取出特征和标签
data_X = df[X_cols].values
data_Y = df[Y_col].values

# 对输入特征做归一化，但不对Height (m MSL)做归一化
scaler = StandardScaler()
X_cols_no_height = X_cols.copy()
X_cols_no_height.remove("Height (m MSL)")
data_X_scaled = df[X_cols_no_height].values
scaled_part = scaler.fit_transform(data_X_scaled)
# 拼接未归一化的Height (m MSL)和归一化后的其他特征
height_col = df[["Height (m MSL)"]].values
# 保证顺序一致
if X_cols[0] == "Height (m MSL)":
    data_X = np.concatenate([height_col, scaled_part], axis=1)
else:
    # 若Height (m MSL)不在首位，需按X_cols顺序重组
    data_X_dict = {"Height (m MSL)": height_col}
    for i, col in enumerate(X_cols_no_height):
        data_X_dict[col] = scaled_part[:, i:i+1]
    data_X = np.concatenate([data_X_dict[col] for col in X_cols], axis=1)

# 划分为训练集和测试集，确保不重合
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_Y, test_size=0.2, random_state=42, shuffle=True, stratify=None
)

# 检查GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转为Tensor并放到DEVICE
train_X = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
train_Y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
test_X = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
test_Y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# 构建数据集和数据加载器
train_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)



model = MLP(input_dim=train_X.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 用于保存每个epoch的loss和acc
loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(num_epochs):
    epoch_loss = 0
    correct_train = 0
    total_train = 0
    for batch_X, batch_Y in train_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct_train += (torch.abs(pred - batch_Y) <= acc_diff).sum().item()
        total_train += batch_Y.size(0)
    train_acc = correct_train / total_train

    # 计算测试集acc
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            pred = model(batch_X)
            correct_test += (torch.abs(pred - batch_Y) <= acc_diff).sum().item()
            total_test += batch_Y.size(0)
    test_acc = correct_test / total_test
    model.train()

    # 保存loss和acc
    loss_list.append(epoch_loss / len(train_loader))
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    if (epoch + 1) % (num_epochs / 10) == 0 or epoch == num_epochs - 1:
        loss = epoch_loss / len(train_loader)
        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

print("\n#########################\n")
# 特征重要性分析
with torch.no_grad():
    weights = model.net[0].weight.abs().mean(dim=0).cpu().numpy()
    feature_importance = list(zip(X_cols, weights))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    print("Feature importance (descending):")
    for name, score in feature_importance:
        print(f"{name}: {score:.4f}")
print("\n#########################\n")

# 测试集评估
with torch.no_grad():
    model.eval()
    test_loss = 0
    for batch_X, batch_Y in test_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    model.train()
print(f"Final: Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}")

print("\n#########################\n")

# 画出loss和acc曲线
plt.figure()
plt.title('Training Curve')

plt.plot(loss_list, label='Loss')
plt.plot(train_acc_list, label='Train Acc')
plt.plot(test_acc_list, label='Test Acc')

plt.xlabel('Epoch')
plt.legend()

plt.show()



with open("MLP.txt", "w", encoding="utf-8") as f:
    f.write("Feature importance (descending):\n")
    for name, score in feature_importance:
        f.write(f"{name}: {score:.4f}\n")
    f.write("\n#########################\n")
    f.write(f"Final: Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")