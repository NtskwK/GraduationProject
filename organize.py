import pandas as pd
import re

# 读取git日志文件
with open("gitlog.log", "r", encoding="utf-8") as f:
    lines = f.readlines()

commits = []
commit = {}
message_lines = []

for line in lines:
    if line.startswith("commit "):
        if commit:
            commit["message"] = "\n".join(message_lines).strip()
            commits.append(commit)
            commit = {}
            message_lines = []
        commit["hash"] = line.strip().split()[1]
    elif line.startswith("Author:"):
        commit["author"] = line.strip()[len("Author:"):].strip()
    elif line.startswith("Date:"):
        commit["date"] = line.strip()[len("Date:"):].strip()
    elif line.strip() == "":
        continue
    else:
        message_lines.append(line.strip())

# 最后一条commit
if commit:
    commit["message"] = "\n".join(message_lines).strip()
    commits.append(commit)

# 转为DataFrame
df = pd.DataFrame(commits)

# 简单分析
print("提交总数:", len(df))
print("作者统计:\n", df["author"].value_counts())
print("最近5条提交:\n", df.head())

# 可选：保存为csv
df.to_csv("gitlog_analysis.csv", index=False)