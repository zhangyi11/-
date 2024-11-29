# 图2.2八数字推盘问题
问题描述：在一块3*3的木板上，有编号为1~8的图块和一个空白区域，与空白区域相邻的图块可以被推入空白区域，我们的目标是从初始布局，通过移动图块达到指定的目标对局。

![image](https://github.com/zhangyi11/Artificial-Intelligence-Yao-Qi-Zhi/blob/main/%E5%9B%BE%E7%89%87/Chapter%202/initial_layout.png?raw=true)![image](https://github.com/zhangyi11/Artificial-Intelligence-Yao-Qi-Zhi/blob/main/%E5%9B%BE%E7%89%87/Chapter%202/final_layout.png?raw=true)

生成上述图片所用到的代码：
```
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']

# 创建一个 3x3 的数字网格
grid_initial = np.array([[8, 0, 6],
                         [5, 4, 7],
                         [2, 3, 1]])
grid_final = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# 颠倒数组，使生成的图像与书中初始布局一致
grid_initial = grid_initial[::-1]
grid_final = grid_final[::-1]

# 创建函数来绘制并保存网格
def plot_grid(grid, title, filename):
    fig, ax = plt.subplots()

    # 设置网格的范围
    ax.set_xlim(-0.5, 2.5)  # x轴从 -0.5 到 2.5
    ax.set_ylim(-0.5, 2.5)  # y轴从 -0.5 到 2.5

    # 绘制 3x3 网格
    for i in range(4):  # 4 条线（包括边界）
        ax.axhline(i - 0.5, color='black', linewidth=2)  # 绘制水平线
        ax.axvline(i - 0.5, color='black', linewidth=2)  # 绘制垂直线

    # 在每个网格格子内添加数字，数字0表示空白格，用灰色填充
    for i in range(3):
        for j in range(3):
            if grid[i, j] == 0:
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth = 0, facecolor = 'gray'))
            else:
                ax.text(j, i, str(grid[i, j]), ha='center', va='center', fontsize=16, color='black')

    # 去掉坐标轴的刻度和坐标轴上的数字
    ax.set_xticks([])  # 去掉x轴的刻度
    ax.set_yticks([])  # 去掉y轴的刻度

    # 设置标题
    ax.set_title(title)

    # 保存图像
    fig.savefig(filename)
    plt.close(fig)  # 关闭图形，防止显示多次

# 生成并保存初始布局图像
plot_grid(grid_initial, "八数字推盘问题（初始布局）", "initial_layout.png")

# 生成并保存目标布局图像
plot_grid(grid_final, "八数字推盘问题（目标布局）", "final_layout.png")

# 显示提示信息
print("两张图片已保存为 initial_layout.png 和 final_layout.png")

```
使用随机策略解决八数字推盘问题
```
import numpy as np
import random

# 创建一个 3x3 的数字网格
grid_initial = np.array([[8, 0, 6],
                         [5, 4, 7],
                         [2, 3, 1]])

grid_final = np.array([[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8]])

# 获取0的位置
rows, cols = np.where(grid_initial == 0)
current_position_0 = (rows[0], cols[0])  # 当前0的位置


# 获取当前位置周围可移动的位置
def possible_position(rows, cols):
    # 计算四个可能的移动位置
    n_position = [(rows-1, cols), (rows+1, cols), (rows, cols-1), (rows, cols+1)]
    # 过滤掉超出边界的位置
    n_position = [(r, c) for r, c in n_position if 0 <= r < 3 and 0 <= c < 3]
    return n_position


# 计数器
count = 0

while True:
    # 随机选择一个可移动的位置
    next_position_0 = random.choice(possible_position(current_position_0[0], current_position_0[1]))

    # 交换当前0位置和目标位置的元素
    grid_initial[current_position_0], grid_initial[next_position_0] = grid_initial[next_position_0], grid_initial[
        current_position_0]

    # 更新当前0的位置
    current_position_0 = next_position_0

    # 增加交换次数
    count += 1

    # 判断是否达到目标布局
    if np.array_equal(grid_initial, grid_final):
        print("总共执行次数", count)
        break
```
我大概试了十多次，最好的执行次数为81573，最坏的执行次数为2878599，该随机策略仍有优化的空间，比如第一次推盘灰色空格向下走，第二次推盘灰色空格就不能再向上走，以保证短期内没有相同的状态空间，代码编写留给读者。
