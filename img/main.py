# -*- coding: utf-8 -*-


import math
import numpy as np
from time import sleep
from pprint import pprint
import matplotlib.pyplot as plt


def init_plt(which_plt, title):
    which_plt.set_title(title)
    which_plt.set_xlabel("x")
    which_plt.set_ylabel("y")
    which_plt.set_xlim(-2, 2)   # 横坐标的纵坐标的区间
    which_plt.set_ylim(0, 4)    # 纵坐标的纵坐标的区间
    which_plt.set_xticks(np.arange(-2, 3, step=1))
    which_plt.set_yticks(np.arange(0, 5, step=1))

    xs = [i[0] for i in raw_agents_loc]
    ys = [i[1] for i in raw_agents_loc]
    # 绘制初始点
    which_plt.scatter(xs, ys, marker='o', label='initial value')
    for i in range(len(xs)):
        which_plt.annotate('{}'.format(i + 1), xy=(xs[i], ys[i]), xytext=(xs[i] + 0.03, ys[i] + 0.05))

    avg_x = np.average(xs)
    avg_y = np.average(ys)
    # 绘制全局中心点
    print("avg(x,y) = ({}, {})".format(avg_x, avg_y))
    which_plt.scatter(avg_x, avg_y, marker='o', label='global center')
    which_plt.annotate('({:.2f},{:.2f})'.format(avg_x, avg_y), xy=(avg_x, avg_y), xytext=(avg_x - 0.3, avg_y - 0.2))
    which_plt.legend()
    return avg_x, avg_y

def calc_L(agents_loc):
    """根据当前的agents_loc求拉普拉斯矩阵"""
    L = np.zeros((num, num))  # 0阵
    for i in range(agents_loc.shape[0]):    # 第i个智能体
        for j in range(agents_loc.shape[0]):# 第j个智能体
            if i != j:
                # 计算距离
                x1 = agents_loc[i][0]
                y1 = agents_loc[i][1]
                x2 = agents_loc[j][0]
                y2 = agents_loc[j][1]
                dis = math.sqrt(math.pow((x1-x2), 2) + math.pow((y1-y2), 2))
                if dis <= r:
                    L[i][j] = 1
                    L[i][i] -= 1
    return np.mat(L)

def calc_f(Z, avg_x, avg_y):
    """计算Z和中心点距离和"""
    tmp_Z = np.array(Z)
    total_dis = 0
    for agent_index in range(Z.shape[0]):
        x1 = tmp_Z[agent_index][0]
        y1 = tmp_Z[agent_index][1]
        dis = math.sqrt(math.pow((x1 - avg_x), 2) + math.pow((y1 - avg_y), 2))
        total_dis += dis
    return total_dis

"""系统的初始值"""
num = 10  # 智能体个数
raw_agents_loc = np.array([
    [0, 0.5], [0, 1], [-1.2, 1], [1.2, 1], [-0.5, 1.5],
    [0.5, 1.5], [-0.7, 2], [0.5, 2], [0, 2.2], [0, 3]
])  # 十个智能体初始位置
gamas = np.array([0.1, 0.2, 0.21, 0.3, 0.4])
r = 1       # 感应半径
times = 20  # 测试20次

"""初始化matplotlib"""
plt.close('all')    # 清空画布
fig, axarr = plt.subplots(3, 5)
fig.set_size_inches(25, 15)

"""先把t0时刻画出来"""
axarr[0, 0].axis('off') # 其他的子图隐藏
axarr[0, 1].axis('off')
axarr[0, 3].axis('off')
axarr[0, 4].axis('off')
avg_x, avg_y = init_plt(axarr[0, 2], "initial")

# 配置统计曲线的图
axarr[2, 2].set_title("f with different gama")
axarr[2, 2].set_xlabel("t/s")
axarr[2, 2].set_ylabel("f(t)")
axarr[2, 2].set_xlim(1, times)  # 横坐标的纵坐标的区间
axarr[2, 2].set_ylim(0, 9)   # 纵坐标的纵坐标的区间
axarr[2, 2].set_xticks(np.arange(1, times+1, step=1))
axarr[2, 2].set_yticks(np.arange(0, 10, step=1))
axarr[2, 0].axis('off') # 其他的子图隐藏
axarr[2, 1].axis('off')
axarr[2, 3].axis('off')
axarr[2, 4].axis('off')

# 计算不同gama下的，t∈[0, 11]的汇集情况
for i in range(len(gamas)):      # 每一个gama
    # 使用矩阵的形式更新，速度快 且 计算方便
    gama = gamas[i]
    Zs = []
    fs = []
    ts = []
    Zs.append(raw_agents_loc)
    Z = np.mat(raw_agents_loc)  # Z矩阵
    agents_loc = raw_agents_loc
    # 迭代times次
    for time in range(times):
        L = calc_L(agents_loc)  # L矩阵
        Z = Z + gama * L * Z

        agents_loc = np.array(Z)

        print("*** Gama: {}, time: {}/{}".format(gama, time+1, times))
        Zs.append(np.array(Z))
        # 统计评估的折线
        f = calc_f(Z, avg_x, avg_y)
        fs.append(f)
        ts.append(time+1)

    init_plt(axarr[1, i], "gama: {}".format(gama))

    # 绘制每一个智能体的运动轨迹
    for agent_index in range(num):
        axarr[1, i].plot(
            [Z[agent_index][0] for Z in Zs],    # 该智能体的x的轨迹
            [Z[agent_index][1] for Z in Zs],    # 该智能体的y的轨迹
            label="{} agent".format(agent_index+1),
            marker='o', markersize=4
        )

    # 绘制统计评估的折线
    axarr[2, 2].plot(
        ts, fs,
        label="gama = {}".format(gama),
        marker='o'
    )


axarr[2, 2].legend()

fig.tight_layout()
plt.savefig("kkk.png", format="png", dpi=500)
plt.show()
