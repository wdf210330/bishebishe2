"""
pure_pursuit_controller.py - 纯跟踪控制器

功能说明：
    基于Pure Pursuit算法的轨迹跟踪控制器
    作为轨迹跟踪的底层转向控制器，稳定可靠

特点：
    - 计算简单，实时性好
    - 对参数不敏感，鲁棒性强
    - 适合作为底层控制器
"""

import numpy as np
import math


class PurePursuitController:
    """
    Pure Pursuit纯跟踪控制器
    
    原理：
        在参考路径上寻找一个前瞻点，计算转向角使车辆朝向该点行驶
    """
    
    def __init__(self, wheelbase=2.89, lookahead_gain=0.5, min_lookahead=3.0, max_lookahead=15.0):
        """
        初始化Pure Pursuit控制器
        
        参数：
            wheelbase: 轴距（米）
            lookahead_gain: 前瞻距离增益
            min_lookahead: 最小前瞻距离（米）
            max_lookahead: 最大前瞻距离（米）
        """
        self.wheelbase = wheelbase
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
    
    def compute_steering(self, current_state, ref_path, target_speed):
        """
        计算转向角
        
        参数：
            current_state: [x, y, yaw, vx, vy, omega]
            ref_path: PATH对象，包含cx, cy, cyaw
            target_speed: 目标速度（m/s）
        
        返回：
            steering: 转向角（弧度）
            lookahead_idx: 前瞻点索引
        """
        x, y, yaw, vx, vy, omega = current_state
        speed = max(abs(vx), 0.1)  # 速度下限
        
        # 确保路径数据是numpy数组
        cx = np.array(ref_path.cx)
        cy = np.array(ref_path.cy)
        
        # 计算自适应前瞻距离
        lookahead_dist = self.lookahead_gain * speed
        lookahead_dist = np.clip(lookahead_dist, self.min_lookahead, self.max_lookahead)
        
        # 找到最近点
        dx = cx - x
        dy = cy - y
        dists = np.sqrt(dx**2 + dy**2)
        nearest_idx = np.argmin(dists)
        
        # 找到前瞻点
        lookahead_idx = nearest_idx
        accumulated_dist = 0.0
        for i in range(nearest_idx, len(cx) - 1):
            segment_dist = np.sqrt(
                (cx[i+1] - cx[i])**2 + 
                (cy[i+1] - cy[i])**2
            )
            accumulated_dist += segment_dist
            if accumulated_dist >= lookahead_dist:
                lookahead_idx = i + 1
                break
        
        # 确保索引不越界
        lookahead_idx = min(lookahead_idx, len(cx) - 1)
        
        # 前瞻点坐标
        lookahead_x = cx[lookahead_idx]
        lookahead_y = cy[lookahead_idx]
        
        # 计算转向角（Pure Pursuit公式）
        dx_la = lookahead_x - x
        dy_la = lookahead_y - y
        
        # 转换到车辆坐标系
        local_x = dx_la * np.cos(yaw) + dy_la * np.sin(yaw)
        local_y = -dx_la * np.sin(yaw) + dy_la * np.cos(yaw)
        
        # 计算曲率（避免除零）
        if lookahead_dist > 0.1:
            curvature = 2 * local_y / (lookahead_dist**2)
        else:
            curvature = 0.0
        
        # 计算转向角
        steering = np.arctan(curvature * self.wheelbase)
        
        # 限幅
        steering = np.clip(steering, -0.7, 0.7)
        
        return steering, lookahead_idx
    
    def compute_speed_control(self, current_speed, target_speed, dt):
        """
        计算速度控制（简单PID）
        
        参数：
            current_speed: 当前速度（m/s）
            target_speed: 目标速度（m/s）
            dt: 时间步长
        
        返回：
            acc: 加速度（m/s²）
        """
        speed_error = target_speed - current_speed
        
        # 简单P控制
        kp = 2.0
        acc = kp * speed_error
        
        # 限幅
        acc = np.clip(acc, -3.0, 3.0)
        
        return acc
