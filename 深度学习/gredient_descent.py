# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
#
#
# def derivatives(x):
#     return 2 * x
#
#
# x = np.linspace(-1, 1, 400)
# y = x ** 2
#
# iters = 10  # 可以根据需要调整迭代次数
#
# w = 1
# lr = 0.8
# ws = [w]
#
# # 创建一个画布，设置子图布局
# fig = plt.figure(figsize=(15, 6))
# gs = gridspec.GridSpec(2, 5)
#
# for i in range(iters):
#     grad = derivatives(w)
#     w = w - lr * grad
#     ws.append(w)
#
#     ax = fig.add_subplot(gs[i])
#     ax.plot(x, y, c='b')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     # 绘制梯度下降的路径
#     ax.plot(ws[:i + 2], [w_val ** 2 for w_val in ws[:i + 2]], c='r', marker='o', linestyle='-')
#     ax.set_title(f'Iteration {i + 1}')
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(0, 1)
#
#     if abs(grad) < 0.0001:
#         print(f"第{i + 1}次迭代达到最低点")
#         break
#
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def derivatives2(w1, w2):
    return 2 * w1, 2 * w2

# 初始化参数
iters = 9  # 设置展示的迭代次数
w1, w2 = 1.0, 3.0  # 初始点
lr = 0.8  # 学习率
w1s, w2s, zs = [w1], [w2], [w1**2 + w2**2]  # 记录参数和函数值

# 梯度下降迭代
for i in range(iters):
    grad1, grad2 = derivatives2(w1, w2)
    w1 -= lr * grad1
    w2 -= lr * grad2
    w1s.append(w1)
    w2s.append(w2)
    zs.append(w1**2 + w2**2)
    if abs(grad1) < 1e-1 and abs(grad2) < 1e-1:
        print(f"第{i}次迭代达到最低点")
        break

# 创建三维曲面
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
x_grid, y_grid = np.meshgrid(x, y)
z_grid = x_grid**2 + y_grid**2

# 创建画布
fig = plt.figure(figsize=(15, 10))
cmap = plt.get_cmap('viridis')  # 使用更具美感的颜色映射

# 每次迭代展示一个子图
for i in range(iters):
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, cmap=cmap, alpha=0.7, edgecolor='none')

    # 绘制连接路径的线条
    ax.plot(w1s[:i+1], w2s[:i+1], zs[:i+1], color='blue', marker='o', markersize=5, markerfacecolor='red')

    ax.set_title(f'Iteration {i+1}', fontsize=10)

    # 美化轴标签
    ax.set_xlabel('w1', fontsize=8)
    ax.set_ylabel('w2', fontsize=8)
    ax.set_zlabel('z', fontsize=8)

    # 统一视角
    ax.view_init(elev=30, azim=45)

# 调整布局以防止重叠，并增大子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
