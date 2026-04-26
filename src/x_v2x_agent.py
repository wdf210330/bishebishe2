"""
x_v2x_agent.py - MPC轨迹跟踪Agent

功能说明：
    基于MPC控制器的自动驾驶轨迹跟踪Agent
    继承自BasicAgent，负责协调全局路径规划、局部轨迹生成和MPC控制求解
    是test_main.py与MPC控制器之间的"桥梁"

核心流程：
    1. 从全局路径中提取参考路径点
    2. 三次样条插值生成平滑路径
    3. 基于当前速度和预测时域计算参考轨迹
    4. 调用MPC控制器求解最优控制
    5. 返回控制指令给test_main.py执行

作者：[待填写]
日期：[待填写]
"""

#!/usr/bin/env python

import os
import sys

# 添加路径以导入official和utils模块
try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'official'))
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'utils'))
except IndexError:
    pass

import copy
import carla
import math
import numpy as np
import interpolate as itp  # 三次样条插值模块
import carla_utils as ca_u  # CARLA工具函数
from enum import Enum
from collections import deque
from basic_agent import BasicAgent
from global_route_planner import GlobalRoutePlanner

import matplotlib.pyplot as plt
import time


class RoadOption(Enum):
    """
    道路拓扑选项枚举
    
    表示车辆从当前车道到下一车道的拓扑关系：
    - VOID: 无效/空
    - LEFT: 左转
    - RIGHT: 右转
    - STRAIGHT: 直行
    - LANEFOLLOW: 车道保持（默认）
    - CHANGELANELEFT: 换道至左侧
    - CHANGELANERIGHT: 换道至右侧
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class Xagent(BasicAgent):
    """
    MPC轨迹跟踪Agent
    
    继承自BasicAgent，负责：
    - 管理全局路径队列
    - 生成局部参考轨迹
    - 调用MPC控制器求解
    - 处理坐标转换
    
    属性：
        _env: 仿真环境对象
        _vehicle: CARLA车辆
        _model: MPC控制器（Vehicle对象）
        _waypoints_queue: 全局路径点队列（deque结构）
        _a_opt: 上一时刻求解的加速度序列
        _delta_opt: 上一时刻求解的转向角序列
    """
    
    def __init__(self, env, model, dt=0.1) -> None:
        """
        初始化Xagent
        
        参数：
            env: Env环境对象
            model: Vehicle（MPC控制器）对象
            dt: 控制时间步长（秒）
        """
        self._env = env
        self._vehicle = env.ego_vehicle  # 被控车辆
        self._model = model
        
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        # -------------------------------------------
        # 误差滤波参数
        # -------------------------------------------
        self._dist_error_old = 0.0  # 上一时刻横向误差
        self._dist_error_filtered = 0.0  # 滤波后横向误差
        self._filter_alpha = 0.8  # 滤波系数（越大越保留新值，推荐0.7~0.9）

        # -------------------------------------------
        # 障碍物避让参数
        # -------------------------------------------
        self._obs_offset_dir = None  # 避让方向锁定（-1=右绕, 1=左绕, None=无锁定）
        self._obs_offset_mag = None  # 避让偏移量大小

        # -------------------------------------------
        # 路径跟踪参数
        # -------------------------------------------
        self._base_min_distance = 2.0  # 基础最小跟车距离（米）
        self._waypoints_queue = deque(maxlen=100000)  # 路径点队列
        self._d_dist = 0.2  # 路径采样距离（米），减小以增加急弯处平滑度
        self._sample_resolution = 2.0  # 全局路径采样分辨率（米）
        
        # -------------------------------------------
        # MPC求解状态（用于热启动）
        # -------------------------------------------
        self._a_opt = np.array([0.0] * self._model.horizon)      # 最优加速度序列初始化，array([0., 0., 0., 0., 0.])
        self._delta_opt = np.array([0.0] * self._model.horizon)  # 最优转向角序列
        self._dt = dt
        
        # -------------------------------------------
        # 状态跟踪
        # -------------------------------------------
        self._next_states = None          # 预测的下一状态序列
        self._last_traffic_light = None   # 上一个交通信号灯
        self._last_traffic_waypoint = None  # 上一个交通灯位置
        self._obstacles = None           # 障碍物位置列表 (N x 2)
        
        # -------------------------------------------
        # MPC代价函数权重（可调整）
        # -------------------------------------------
        # Q: 状态跟踪权重，[x, y, yaw, vx]的重要性
        # R: 控制代价权重，[acc, steer]的重要性
        # Rd: 控制平滑权重，抑制控制抖动
        # 避障调整：适当降低速度权重，让绕行比刹车更划算
        # 【修复振荡问题】降低Q权重，增大Rd转向平滑权重
        # self._model.solver_basis(Q=np.diag([7, 7, 7, 7, 7]), R=np.diag([2, 2]), Rd=np.diag([2.0, 1000]))
        # self._model.solver_basis(Q=np.diag([7, 7, 7, 7, 0]), R=np.diag([2, 1]), Rd=np.diag([1, 10]))
        self._model.solver_basis(Q=np.diag([7, 7, 7, 7, 0]), R=np.diag([1, 2]), Rd=np.diag([0.5, 50]))
        self.Q_origin = copy.deepcopy(self._model.Q)  # 保存原始权重，创建Q副本，方便后续调整
        self._log_data = []  # 数据记录（用于后续分析）
        self._simu_time = 0  # 累计仿真时间
        
        # -------------------------------------------
        # 全局路径规划器
        # -------------------------------------------
        self._global_planner = GlobalRoutePlanner(self._map, self._sample_resolution)
        
        # 路径点移动距离计算参数
        self.dist_move = 0.2   # 初始移动距离
        self.dist_step = 1.5  # 距离步长系数
    
    def plan_route(self, start_location, end_location):
        """
        规划全局路径
        
        从起点到终点计算全局路径，并将路径点加入队列，（有长度限制）
        
        参数：
            start_location: 起点CARLA Location
            end_location: 终点CARLA Location
        """
        self._route = self.trace_route(start_location.location, end_location.location)
        # 将路径点加入队列
        for i in self._route:
            self._waypoints_queue.append(i)
    
    def set_start_end_transforms(self, start_idx, end_idx):
        """
        设置起点和终点的Transform（通过spawn point索引）
        
        参数：
            start_idx: 起点spawn point索引
            end_idx: 终点spawn point索引
        
        异常：
            IndexError: 索引超出范围时抛出
        """
        spawn_points = self._map.get_spawn_points()  # 获取所有spawn points
        if start_idx < len(spawn_points) and end_idx < len(spawn_points):
            self._start_transform = spawn_points[start_idx]
            self._end_transform = spawn_points[end_idx]
        else:
            raise IndexError("Start or end index out of bounds!")
    
    def set_obstacles(self, obstacles):
        """
        设置障碍物位置
        
        参数：
            obstacles: N x 2 数组，每行是 [x, y] 障碍物坐标
        """
        self._obstacles = obstacles

    def apply_obs_avoidance_offset(self, waypoints, ego_x, ego_y, ego_yaw):
        """
        不施加任何轨迹偏移，完全依靠势场法避障
        
        参数：
            waypoints: np.array [4, N]，行分别是 [x, y, v, yaw]
            ego_x, ego_y: 车辆当前位置
            ego_yaw: 车辆当前航向角（弧度）
            
        返回：
            原始waypoints数组（不做修改）
        """
        return waypoints


    def calc_ref_trajectory_in_T_step(self, node, ref_path, sp):
        """
        计算T步预测时域内的参考轨迹
        
        根据当前速度和预测时域，在参考路径上采样T+1个点作为MPC的参考轨迹
        
        参数：
            node: 当前状态 [x, y, v, yaw]
            ref_path: 参考路径对象（含cx, cy, cyaw等）
            sp: 速度剖面（速度随路程变化的曲线）
        
        返回：
            z_ref: 参考轨迹 [4, T+1]，包含[x, y, v, yaw]
            ind: 当前路径点索引
        """
        T = self._model.horizon  # 预测时域长度
        z_ref = np.zeros((4, T + 1))  # 初始化参考轨迹，4行T+1列的零矩阵
        
        # 找到参考路径上距离当前状态最近的点，length轨迹点总数
        length = ref_path.length
        #找到参考路径上离车辆最近的点。
        #输出含义ind 最近路径点在cx,cy 数组中的索引
        #er 横向距离偏差（车辆偏左 / 偏右多少米）
        ind, _ = ref_path.nearest_index(node)
        
        # 第一个点：最近邻点，4个参数放置在第一列
        z_ref[0, 0] = ref_path.cx[ind]
        z_ref[1, 0] = ref_path.cy[ind]
        z_ref[2, 0] = sp[ind]  # 速度
        z_ref[3, 0] = ref_path.cyaw[ind]  # 航向角
        
        # -------------------------------------------
        # 沿路径前向采样T个点，# self.dist_move初始移动距离，self.dist_step距离步长系数
        # -------------------------------------------
        #copy.copy(self.dist_move) 就是"我需要用这个值，但我不想改掉原来的基准值"。
        dist_move = copy.copy(self.dist_move)
        
        for i in range(1, T + 1):
            # 根据当前速度计算移动距离
            # 速度越大，采样点越远（自适应采样）
            dist_move += self.dist_step * abs(self._model.get_v()) * self._dt   #从当前点前看了多少米
            ind_move = int(round(dist_move / self._d_dist))                     #从当前点前看了多少米，轨迹采样间隔（ds=0.5m）
            index = min(ind + ind_move, length - 1)  # 防止索引超出轨迹末尾
            
            # 提取参考状态
            z_ref[0, i] = ref_path.cx[index]
            z_ref[1, i] = ref_path.cy[index]
            z_ref[2, i] = sp[index]
            z_ref[3, i] = ref_path.cyaw[index]

        return z_ref, ind
    
    def rotate(self, x, y, theta, ratio=1.75):
        """
        坐标旋转（绕原点旋转theta角度）
        
        用于计算车辆相对路径点的横向距离
        
        参数：
            x, y: 原始坐标
            theta: 旋转角度（弧度）
            ratio: 缩放比例
        
        返回：
            旋转后的坐标 [x', y']
        """
        return np.array([
            (x * np.cos(theta) - y * np.sin(theta)) * ratio,
            (x * np.sin(theta) + y * np.cos(theta)) * ratio
        ])
    
    def lat_dis_wp_ev(self, wp, ev):
        """
        计算路径点与车辆之间的横向距离
        
        用于车道保持等功能的误差计算
        
        参数：
            wp: 路径点Waypoint
            ev: 车辆Actor
        
        返回：
            横向距离（米）
        """
        # 提取位置
        wp_loc = np.array([wp.transform.location.x, wp.transform.location.y])
        ev_loc = np.array([ev.get_location().x, ev.get_location().y])
        
        # 旋转到路径点局部坐标系
        wp_yaw = wp.transform.rotation.yaw
        wp_loc = self.rotate(wp_loc[0], wp_loc[1], np.deg2rad(wp_yaw))
        ev_loc = self.rotate(ev_loc[0], ev_loc[1], np.deg2rad(wp_yaw))
        
        # 返回y方向的距离（横向）
        return np.abs(wp_loc[1] - ev_loc[1])
    
    def run_step(self, lv=None):
        """
        执行一步控制（核心方法，每个控制周期调用一次）
        
        流程：
            1. 获取当前车辆状态
            2. 更新路径点队列（移除已通过的）
            3. 插值生成平滑参考路径
            4. 计算T步参考轨迹
            5. 调用MPC求解最优控制
            6. 绘制预测轨迹（可视化）
            7. 返回控制指令
        
        返回：
            (a_opt, delta_opt, next_state): 加速度、转
            向角、下一状态
        """
        # 累计仿真时间，dt=0.05s
        self._simu_time += self._dt


        # 日志功能



        import os

        log_dir = r"C:\Users\Administrator\Desktop\carla_MPC-main2\carla_MPC-main2\debug_logs"

        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "run_debug.log")

        def log(msg):

            with open(log_file, "a", encoding="utf-8") as f:

                f.write(msg + "\n")
        
        # -------------------------------------------
        # 1. 获取当前车辆状态，
        # state: [x, y, yaw, vx, vy, omega]，return [self.x, self.y, self.yaw, self.vx, self.vy, self.omega], self.z
        # -------------------------------------------
        state, height = self._model.get_state_carla()  # 从CARLA获取状态
        # 坐标转换：CARLA坐标系（左手） -> 右手坐标系，yaw取弧度， vy, omega取相反方向
        current_state = np.array(ca_u.carla_vector_to_rh_vector(
            state[0:2], state[2], state[3:]
        ))
        
        # -------------------------------------------
        # 2. 动态计算最小跟车距离
        # -------------------------------------------
        veh_location = self._vehicle.get_location()         #当前车的坐标
        vehicle_speed = self._model.get_v()                 #当前车速
        # 速度越大，最小跟车距离越大（安全考虑）
        self._min_distance = self._base_min_distance + 0.5 * vehicle_speed


        # -------------------------------------------
        # 3. 获取参考路径点,_waypoints_queue先进先出
        # -------------------------------------------
        if len(self._waypoints_queue) == 0:
            raise Exception("No waypoints to follow")
        else:
            carla_wp, _ = np.array(self._waypoints_queue).T   #这行代码就是从队列里把所有路点对象"倒出来"，方便后续逐个处理。【】
            waypoints = []
            v = math.sqrt(current_state[3]**2 + current_state[4]**2)  # 合成速度
            waypoints.append([current_state[0], current_state[1], v, current_state[2]])#x，y，v，yaw
            cnt = 0


            # -------------------------------------------
            # 简化路径点（避免插值NaN）
            # 只保留前30个点，减少计算量
            # 同时去除距离过近的重复点
            # -------------------------------------------
            last_state = None
            for wp in carla_wp:
                if cnt > 30:
                    break
                cnt += 1
                t = wp.transform
                #参考轨迹（x，y，yaw）， last_state = None，不考虑v=0，omega=0
                ref_state = ca_u.carla_vector_to_rh_vector(
                    [t.location.x, t.location.y], t.rotation.yaw
                )
                if last_state is not None:
                    # 距离太近的路径点会导致三次样条插值出现NaN
                    if np.sqrt(ref_state[0]**2 + ref_state[1]**2) - last_state < 0.005:
                        continue
                waypoints.append([
                    ref_state[0], ref_state[1],
                    self._model.target_v, ref_state[2]
                ])
                last_state = np.sqrt(ref_state[0]**2 + ref_state[1]**2)

            #把从carla_wp里得到的前30个路径点信息进行判断，距离过近的点取其一，得到新的路点列表，cnt行4列
            #waypoints=[[current_state[0], current_state[1], v, current_state[2]]，
            #   cnt=1   [ref_state[0], ref_state[1],self._model.target_v, ref_state[2]]，
            #   cnt=2   [ref_state[0], ref_state[1],self._model.target_v, ref_state[2]]
            waypoints = np.array(waypoints).T
            #T后：(第一列是当前状态，后面是参考点的状态)
            #├── 0: [x_0, x_1, x_2, ..., x_(N - 1)]     ← x
            #坐标
            #├── 1: [y_0, y_1, y_2, ..., y_(N - 1)]     ← y
            #坐标
            #├── 2: [v_0, v_1, v_2, ..., v_(N - 1)]      ← 速度（第一个是当前速度，后面是target_v）
            #└── 3: [yaw_0, yaw_1, yaw_2, ..., yaw_(N - 1)] ← yaw弧度角



        # -------------------------------------------
        # 4. 移除已通过的路径点
        # -------------------------------------------
        num_waypoint_removed = 0        # 移除路径点计数器
        for waypoint, _ in self._waypoints_queue:
            # 最后一个点要特殊处理（快到了才移除）
            if len(self._waypoints_queue) - num_waypoint_removed == 1:     # 只剩最后一个点未被移除
                min_distance = 1                                           #缩小移除距离
            else:
                min_distance = self._min_distance
            
            # 如果车辆已经到达该路径点，移除它
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break
        
        # 从队列左侧（队首）移除已通过的路径点
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()


        # -------------------------------------------
        # 4.5 障碍物避让：直接偏移参考轨迹
        # -------------------------------------------
        # 使用车辆当前航向角（从current_state获取）
        ego_yaw_for_offset = current_state[2]
        waypoints = self.apply_obs_avoidance_offset(
            waypoints, current_state[0], current_state[1], ego_yaw_for_offset
        )


        # -------------------------------------------
        # 5. 三次样条插值生成平滑路径
        # -------------------------------------------

        #cx，cy，cyaw为插值后的密集的x，y坐标，yaw；
        #ck插值后所有点的曲率。ck ≈ 1/转弯半径，ck 很小（趋近 0）→ 直线行驶，ck很大→ 急弯；曲率在 MPC 控制器里很有用，可以提前知道前方弯道有多大，据此调整速度。
        #ds采样间隔，比如 0.5 表示每隔 0.5 米输出一个点
        #插值后每个点的 累计路程（从起点到该点的弧长）
        cx, cy, cyaw, ck, s = itp.calc_spline_course_carla(
            waypoints[0], waypoints[1], waypoints[3][0], ds=self._d_dist
        )
        # 根据路径曲率生成速度剖面，这行代码的作用是根据参考轨迹的曲率，生成一个速度规划数组。
        #sp 是一个与 cx, cy 等长的数组，每个元素对应轨迹上每个点的推荐速度：
        #sp = [5.0, 5.0, 5.0, 4.8, 4.5, 3.2, 2.0, 1.5, 0.5, 0.0, ...]
        #↑       ↑        ↑      ↑         ↑
        #
        #直线    弯道前   弯道中  快到终点   停止

        sp = itp.calc_speed_profile(cx, cy, cyaw, self._model.target_v)
        
        # 封装为路径对象
        ref_path = itp.PATH(cx, cy, cyaw, ck)



        # -------------------------------------------
        # 6. 计算T步参考轨迹
        # -------------------------------------------
        #输出含义ind，当前路径点索引，最近路径点在cx, cy 数组中的索引
        # z_ref: 参考轨迹[4, T + 1]，包含[x, y, v, yaw]
        # [x_0,    x_1,     x_2,     x_3,    ...,   x_T]
        # [y_0,    y_1,     y_2,     y_3,    ...,   y_T]
        # [v_0,    v_1,     v_2,     v_3,    ...,   v_T]
        # [yaw_0, yaw_1,  yaw_2,  yaw_3,  ...,  yaw_T]


        z_ref, target_ind = self.calc_ref_trajectory_in_T_step(
            [current_state[0], current_state[1], v, current_state[2]],
            ref_path, sp
        )

        
        # 转换为MPC需要的状态格式 [x, y, yaw, vx, vy, omega]，6维状态向量，[:, :self._model.horizon]切片，切掉最后一列
        ref_traj = np.array([
            z_ref[0], z_ref[1], z_ref[3], z_ref[2],  # x, y, yaw, v
            [0] * len(z_ref[0]),  # vy = 0
            [0] * len(z_ref[0])   # omega = 0
        ])[:, :self._model.horizon]
        
        # -------------------------------------------
        # 7. MPC求解
        # -------------------------------------------
        # 11行×6列，每一行 [x, y, yaw, vx, vy, omega]是一个时间步的状态：
        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon + 1)
            ).T
        
        # 更新当前速度
        cur_v = self._model.get_v()
        #把预测序列中每一行的vx（纵向速度）都填成当前速度cur_v，作为MPC求解的初始猜测。
        self._next_states[:, 3] = cur_v
        #把current_state中的速度分量vx,vy,omega更新为预测序列第 0 行（当前时刻）的值，保持一致。
        current_state[3:] = self._next_states[0][3:]
        
        # 使用上一时刻的控制序列作为热启动，u0为输入
        u0 = np.array([self._a_opt, self._delta_opt]).reshape(-1, 2).T
        #u0形状: (2, T)时间步0 到T：
        #加速度: [a0, a1, a2, ..., aT]
        #转向角: [δ0, δ1, δ2, ..., δT]

        # -------------------------------------------
        # 7.5 障碍物接近检测：决定是否用冷启动
        # -------------------------------------------
        # 如果障碍物在势场范围内，使用冷启动帮助跳出局部最优
        # (暂时禁用，恢复热启动)

        # 添加势场代价（障碍物、道路边界等，目前为0）
        apf_obs = apf_nc_road = apf_c_road = 0
        
        # 构建MPC优化问题
        self._model.solver_add_cost()      # 定义代价函数
        
        # 调试：打印障碍物信息

        
        if self._obstacles is not None and len(self._obstacles) > 0:
            log(f"[DEBUG] 障碍物数量: {len(self._obstacles)}, 位置: {self._obstacles}")
        else:
            log("[DEBUG] 无障碍物")
        
        self._model.solver_add_soft_obs(self._obstacles)  # 添加软障碍物避让约束
        self._model.solver_add_bounds()  # 添加约束条件
        
        # 求解MPC
        state = self._model.solve_MPC(
            ref_traj.T, current_state, self._next_states, u0
        )
        
        # 调试：打印求解返回
        log(f"[DEBUG solve_MPC返回] len(state)={len(state)}, type={type(state)}")
        
        # -------------------------------------------
        # 8. 可视化预测轨迹
        # -------------------------------------------
        # 绘制参考轨迹红色 (255，0，0)，[:2, :]前2行
        ca_u.draw_planned_trj(
            self._world, state[2][:, :2], height + 0.5, color=(255,0,0)
        )


        # 保存预测状态序列（用于下一时刻热启动）
        self._next_states = state[2]
        
        # -------------------------------------------
        # 9. 准备返回数据
        # -------------------------------------------
        next_state = state[2][1]
        self._a_opt = state[0]     # 更新最优加速度序列
        self._delta_opt = state[1]  # 更新最优转向角序列
        
        # 调试：打印控制输出
        log(f"[DEBUG] a_opt: {self._a_opt[0]:.3f}, delta_opt: {self._delta_opt[0]:.3f}")
        log(f"[DEBUG] 下一状态: x={next_state[0]:.1f}, y={next_state[1]:.1f}, v={next_state[3]:.1f}")
        
        # 用动力学模型预测实际下一状态
        next_state = self._model.predict(
            current_state, (self._a_opt[0], self._delta_opt[0])
        )
        self._model.set_state(next_state)  # 更新模型内部状态
        
        # -------------------------------------------
        # 10. 计算跟踪误差（用于日志/调试）
        # -------------------------------------------
        # 横向误差：使用PATH类的nearest_index方法获取真正的横向偏差
        # ind: 最近路径点索引, er: 横向偏差（正=车辆偏左，负=车辆偏右）
        _, dist_error = ref_path.nearest_index(
            [current_state[0], current_state[1], current_state[2]]
        )
        
        # 指数平滑滤波：消减突变尖峰
        # y_filtered = alpha * y_raw + (1-alpha) * y_old
        dist_error_filtered = self._filter_alpha * dist_error + (1 - self._filter_alpha) * self._dist_error_old
        self._dist_error_old = dist_error_filtered  # 保存供下一帧使用
        
        yaw_error = abs(next_state[2] - ref_traj[2, 1])
        vel_error = abs(state[2][0][3] - ref_traj[3, 0])
        
        # 当前控制输出
        acc = self._a_opt[0]
        steer = self._delta_opt[0]
        
        # 求解耗时time_2
        cost_time = state[-1]

        return self._a_opt[0], self._delta_opt[0], (next_state, height + 0.05), cost_time * 1000, dist_error
    
    def trace_route(self, start_location, end_location):
        """
        调用全局路径规划器计算路径
        
        参数：
            start_location: 起点Location
            end_location: 终点Location
        
        返回：
            路径（Waypoint, RoadOption）列表
        """
        return self._global_planner.trace_route(start_location, end_location)
