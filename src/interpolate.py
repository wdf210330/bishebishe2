#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Cubic Spline library on python

author Atsushi Sakai

usage: see test codes as below

license: MIT
"""

import math
import numpy as np
import bisect


class Spline:
    """
    三次样条插值类（Cubic Spline Interpolation）

    给定一组离散的x-y数据点，使用三次样条插值方法拟合出一条平滑的曲线。
    样条插值的特点是在每个数据点处连续且二阶导数也连续，曲线光滑自然。

    数学原理：
    对于每个区间 [x_i, x_{i+1}]，三次多项式形式为：
    S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3

    通过边界条件（自然边界或夹紧边界）求解系数 a, b, c, d
    """

    def __init__(self, x, y):
        """
        初始化三次样条插值器

        参数:
            x: list/np.array, x坐标数据点（必须单调递增）
            y: list/np.array, 对应的y坐标数据点
        """
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x          # 存储x坐标
        self.y = y          # 存储y坐标

        self.nx = len(x)    # 数据点数量

        # 计算相邻x点之间的距离（步长）
        h = np.diff(x)

        # 检查x是否有序
        if h.any() == np.nan:
            print("x is not sorted")

        # a系数等于原始y值
        self.a = [iy for iy in y]

        # 构建线性系统 Ax = B，求解c系数
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # 根据c系数，计算b和d系数
        for i in range(self.nx - 1):
            # d_i = (c_{i+1} - c_i) / (3 * h_i)
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))

            # b_i = (y_{i+1} - y_i) / h_i - h_i * (c_{i+1} + 2*c_i) / 3
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        计算给定t处的插值结果（位置/y值）

        参数:
            t: float, 要计算的x坐标位置

        返回:
            float: 插值得到的y值
            None: 如果t超出输入x的范围
        """

        # 边界检查：t必须在x的范围内
        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        # 找到t所在的区间索引
        i = self.__search_index(t)

        # 计算相对于区间起点的距离
        dx = t - self.x[i]

        # 三次多项式: a + b*dx + c*dx^2 + d*dx^3
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        计算给定t处的一阶导数（切线斜率/速度）

        参数:
            t: float, 要计算的x坐标位置

        返回:
            float: 一阶导数值
            None: 如果t超出输入x的范围
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]

        # 一阶导数: b + 2*c*dx + 3*d*dx^2
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        计算给定t处的二阶导数（曲率变化率）

        参数:
            t: float, 要计算的x坐标位置

        返回:
            float: 二阶导数值
            None: 如果t超出输入x的范围
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]

        # 二阶导数: 2*c + 6*d*dx
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        使用二分查找找到x所在的区间索引

        参数:
            x: float, 要查找的x值

        返回:
            int: x所在的区间索引（满足 x[i] <= x < x[i+1]）
        """
        # bisect.bisect 返回x应该插入的位置减1，即为左区间索引
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        构建求解c系数的矩阵A（系数矩阵）

        使用自然边界条件（自然样条）：
        - 首末端点的二阶导数为0

        参数:
            h: np.array, 相邻x点的距离数组

        返回:
            np.array: n x n 的三对角矩阵A
        """
        A = np.zeros((self.nx, self.nx))

        # 自然边界条件：起点处
        A[0, 0] = 1.0

        # 构建三对角矩阵
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                # 对角线元素: 2*(h_i + h_{i+1})
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])

            # 下对角线元素: h_i
            A[i + 1, i] = h[i]

            # 上对角线元素: h_{i+1}
            A[i, i + 1] = h[i]

        # 自然边界条件：终点处
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0

        return A

    def __calc_B(self, h):
        """
        构建求解c系数的向量B（常数项向量）

        参数:
            h: np.array, 相邻x点的距离数组

        返回:
            np.array: 长度为n的常数项向量B
        """
        B = np.zeros(self.nx)

        # B[i] = 3*(a_{i+2} - a_{i+1})/h_{i+1} - 3*(a_{i+1} - a_i)/h_i
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]

        return B


class Spline2D:
    """
    二维三次样条插值类（2D Cubic Spline）

    用于对二维平面上的路径进行参数化样条插值。
    将x和y都表示为弧长s的函数：x(s), y(s)

    适用场景：
    - 路径平滑与重采样
    - 获取路径上任意点的位置、航向角、曲率
    """

    def __init__(self, x, y):
        """
        初始化二维样条插值器

        参数:
            x: list/np.array, x坐标序列
            y: list/np.array, y坐标序列
        """
        # 计算弧长累计量s，作为参数化变量
        self.s = self.__calc_s(x, y)

        # 分别对x和y进行基于弧长s的一次样条插值
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        """
        计算沿着路径的累计弧长

        参数:
            x, y: 路径点坐标

        返回:
            list: 从起点到每个点的累计弧长
        """
        dx = np.diff(x)
        dy = np.diff(y)

        # 检查数据有效性
        if dx.any() == np.nan or dy.any() == np.nan:
            print("x or y is not sorted")

        # 计算每段直线距离
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]

        # 累计求和，首个点弧长为0
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        根据弧长s计算对应的x, y坐标

        参数:
            s: float, 弧长参数（从路径起点开始的累计距离）

        返回:
            tuple: (x, y) 坐标值
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        计算路径在弧长s处的曲率

        曲率k = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
        对于二维平面，曲率表示曲线弯曲的程度

        参数:
            s: float, 弧长参数

        返回:
            float: 曲率值 [1/m]
        """
        # 一阶导数（速度/切向量）
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)

        # 曲率公式
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        """
        计算路径在弧长s处的航向角（yaw angle）

        航向角是路径切线与x轴正方向的夹角

        参数:
            s: float, 弧长参数

        返回:
            float: 航向角 [弧度], 范围 [-π, π]
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)

        # atan2(dy, dx) 给出相对于x轴的角度
        yaw = math.atan2(dy, dx)

        return yaw

    def calc_yaw_carla(self, s, ref_yaw):
        """
        计算航向角，并对角度突变进行平滑处理（适用于CARLA仿真）

        当计算得到的yaw与参考yaw差异过大时（超过π），说明存在角度跳变，
        此时进行插值平滑，避免车辆产生180度转向

        参数:
            s: float, 弧长参数
            ref_yaw: float, 参考航向角（上一时刻的yaw）

        返回:
            float: 平滑后的航向角 [弧度]
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)

        # 如果yaw接近0，直接返回参考yaw（避免微小抖动）
        if yaw < 0.0001:
            return ref_yaw

        ratio = 0.1  # 平滑系数，值越小越保守

        # 检测角度跳变：两角差超过π说明存在绕圈情况
        if abs(ref_yaw - yaw) > np.pi:
            if yaw < 0:
                # yaw为负，从正方向接近，需要正向插值
                yaw = ref_yaw + (yaw + np.pi) * ratio
            elif yaw > 0:
                # yaw为正，从负方向接近，需要负向插值
                yaw = ref_yaw - (yaw - np.pi) * ratio
        else:
            # 正常情况，渐近调整yaw
            yaw = ref_yaw - (ref_yaw - yaw) * ratio

        return yaw


def calc_spline_course(x, y, ds=0.1):
    """
    根据给定的路径点，计算平滑后的样条路径

    参数:
        x: list/np.array, 原始x坐标点
        y: list/np.array, 原始y坐标点
        ds: float, 采样步长（弧长间隔），默认0.1m

    返回:
        rx, ry: 平滑后的x, y坐标
        ryaw: 各点航向角 [弧度]
        rk: 各点曲率 [1/m]
        s: 弧长参数数组
    """
    # 创建二维样条插值器
    sp = Spline2D(x, y)

    # 从0到路径总长，按ds间隔采样
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []

    # 沿弧长采样，获取路径上的位置、航向、曲率
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def calc_spline_course_carla(x, y, yaw, ds=0.1):
    """
    计算样条路径，并使用CARLA专用航向角平滑处理

    与calc_spline_course类似，但航向角计算使用calc_yaw_carla
    进行平滑处理，适用于自动驾驶仿真中避免角度突变

    参数:
        x, y: 原始路径点坐标
        yaw: 初始参考航向角
        ds: 采样步长，默认0.1m

    返回:
        rx, ry, ryaw, rk, s: 与上面函数相同
    """
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)

        # 使用CARLA专用航向角计算（带平滑）
        ryaw.append(sp.calc_yaw_carla(i_s, yaw))
        yaw = ryaw[-1]  # 更新参考yaw用于下一时刻

        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def calc_spline_course_carla_lti(x, y, yaw, ds=0.1):
    """
    计算样条路径（LT变体，可能有bug，calc_yaw参数个数不匹配）

    注意：此函数中 calc_yaw(i_s, yaw) 调用方式有误，
    正确的Spline2D.calc_yaw只接受一个参数s
    """
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s, yaw))  # 注意：这里参数个数可能有问题
        yaw = ryaw[-1]
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def test_spline2d():
    """
    二维样条插值测试函数

    创建一个带有环路（loop）的路径进行样条插值，
    并绘制位置、航向角、曲率的可视化图表
    """
    print("Spline 2D test")

    import matplotlib.pyplot as plt

    # 定义一个非单调的路径（包含环路）
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]

    # 创建二维样条
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.1)

    # 采样得到平滑路径
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    # 绘图1：路径对比（原始点 vs 样条曲线）
    flg, ax = plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    # 绘图2：航向角沿路径的变化
    flg, ax = plt.subplots(1)
    plt.plot(s, [math.degrees(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    # 绘图3：曲率沿路径的变化
    flg, ax = plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()


def test_spline():
    """
    一维样条插值测试函数

    对一组一维数据点进行三次样条插值，并绘图对比
    """
    print("Spline test")
    import matplotlib.pyplot as plt

    x = [-0.5, 0.0, 0.5, 1.0, 1.5]
    y = [3.2, 2.7, 6, 5, 6.5]

    # 创建样条插值器
    spline = Spline(x, y)

    # 在更大范围内采样绘图
    rx = np.arange(-2.0, 4, 0.01)
    ry = [spline.calc(i) for i in rx]

    plt.plot(x, y, "xb")
    plt.plot(rx, ry, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def pi_2_pi(angle):
    """
    将角度归一化到 [-π, π] 范围内

    用于处理角度的周期性，避免角度计算中出现大的跳变

    参数:
        angle: float, 输入角度 [弧度]

    返回:
        float: 归一化后的角度 [弧度], 范围 [-π, π]
    """
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    根据路径设计合适的速度策略

    核心逻辑：
    - 检测路径中的大角度转向点
    - 对于需要掉头或后退的路段，标记为反向行驶（负速度）

    参数:
        cx: list, 参考路径的x坐标序列
        cy: list, 参考路径的y坐标序列
        cyaw: list, 参考路径各点的航向角 [弧度]
        target_speed: float, 目标前进速度 [m/s]

    返回:
        list: 速度剖面，正值为前进，负值为后退
    """

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # 默认前进方向

    # 遍历路径上的每个点（除了最后一个）
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        # 计算从当前点到下一个点的运动方向角
        move_direction = math.atan2(dy, dx)

        # 检查是否存在大角度转向（≥ 45度）
        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))

            # 如果转向角度超过45度，认为需要掉头
            if dangle >= math.pi / 4.0:
                direction = -1.0  # 标记为后退
            else:
                direction = 1.0   # 继续前进

        # 根据行驶方向设置速度
        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

    # 路径终点速度设为0（完全停止）
    speed_profile[-1] = 0.0

    return speed_profile


class PATH:
    """
    路径类：封装参考路径并提供查询功能

    用于MPC（模型预测控制）等控制算法中，
    快速查找车辆在路径上的最近点和横向偏差
    """

    def __init__(self, cx, cy, cyaw, ck):
        """
        初始化路径对象

        参数:
            cx: list, 路径x坐标
            cy: list, 路径y坐标
            cyaw: list, 路径各点航向角
            ck: list, 路径各点曲率
        """
        self.cx = cx          # 路径x坐标
        self.cy = cy          # 路径y坐标
        self.cyaw = cyaw      # 路径航向角
        self.ck = ck          # 路径曲率
        self.length = len(cx)  # 路径点数量
        self.ind_old = 0      # 上一次查询的索引（用于加速搜索）

    def nearest_index(self, node, N_IND=10):
        """
        计算路径上距离车辆最近的点索引

        只在当前索引附近N_IND个点的范围内搜索，
        避免全局搜索的计算开销

        参数:
            node: list/tuple, 车辆当前状态 [x, y, yaw]
            N_IND: int, 向前搜索的点数，默认10

        返回:
            ind: int, 最近点的路径索引
            er: float, 横向偏差（正值为左，负值为右）
        """

        # 提取车辆位置
        dx = [node[0] - x for x in self.cx[self.ind_old: (self.ind_old + N_IND)]]
        dy = [node[1] - y for y in self.cy[self.ind_old: (self.ind_old + N_IND)]]

        # 计算到各点的欧氏距离
        dist = np.hypot(dx, dy)

        # 找到最小距离对应的索引
        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind  # 更新索引缓存

        # 计算横向偏差
        # 构建后轴坐标系（车身后方90度）的单位向量
        rear_axle_vec_rot_90 = np.array([[math.cos(node[2] + math.pi / 2.0)],
                                         [math.sin(node[2] + math.pi / 2.0)]])

        # 从目标点指向后轴的向量
        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        # 点积得到横向偏差（正=目标在车左侧，负=目标在车右侧）
        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er


if __name__ == '__main__':
    # 运行一维样条测试
    test_spline()

    # 运行二维样条测试
    test_spline2d()
