import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV読み込み
df = pd.read_csv('data/wine.csv').to_numpy()

# 特徴量
x = df[:, 1]

# 品質
y = df[:, 11]

# 単回帰
a, b = np.polyfit(x, y, 1)

# 図全体
fig = plt.figure(figsize=(7, 7))

# 散布図エリア
ax = fig.add_axes([0.46, 0.46, 0.46, 0.46])

ax.scatter(x, y, s=10)

# 回帰直線
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, a * x_line + b)

ax.set_xlabel("x:volatile acidity")
ax.set_ylabel("y:quality")

plt.show()