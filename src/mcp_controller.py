"""
mcp_controller.py - MPC模型预测控制器

功能说明：
    基于模型预测控制（MPC）的自动驾驶轨迹跟踪控制器
    采用自行车模型（Bicycle Model）作为车辆动力学模型
    使用CasADi符号计算库和IPOPT求解器进行在线优化求解

参考文献：
    - Carla中车辆避障与轨迹跟踪的MPC原理：https://zhuanlan.zhihu.com/p/525523586
    - gustavomoers/CollisionAvoidance-Carla-DRL-MPC（Simple版本参考）
    - henryhcliu/udmc_carla（Strong版本参考）

注意：
    本控制器包含两种模式：
    - solve_MPC_wo：无前次解热启动，适用于开环测试
    - solve_MPC：带前次解热启动，适用于闭环控制（推荐）
"""

import carla
import math
import os
import time
import copy
import numpy as np
import casadi as ca
from scipy.optimize import minimize

try:
    from .project_paths import ensure_debug_log_dir
except ImportError:
    from project_paths import ensure_debug_log_dir

# ============================================
# 车辆动力学参数（通过系统辨识得到）
# ============================================
kf = -102129.8307648713   # 前轮侧偏刚度（N/rad）
kr = -89999.99208226385   # 后轮侧偏刚度（N/rad）
lf = 1.2868463998073212   # 前轴到质心的距离（m）
lr = 1.6031536001974758   # 后轴到质心的距离（m）
m = 1845.9998612883219    # 车辆质量（kg）
Iz = 2699.993738447328    # 车辆绕z轴转动惯量（kg·m²）

dt = None                 # 仿真步长（秒），运行时由Vehicle类设置
Lk = lf*kf - lr*kr        # 前后轮侧偏刚度加权和（中间变量）

n_input = 2  # 控制输入维度：acc（加速度）、steering（转向角）


class Vehicle:
    """
    车辆类：封装车辆动力学模型和MPC控制器
    
    属性：
        - 车辆状态：位置(x, y)、航向角(yaw)、速度(vx, vy)、横摆角速度(omega)
        - MPC参数：预测时域(horizon)、最大迭代次数(maxiter)
        - 约束限制：转向角边界、加速度边界
    """
    
    def __init__(self, state=None, actor=None, horizon=10, target_v=18, carla=True, delta_t=0.05, max_iter=150):
        """
        初始化车辆对象和MPC控制器
        
        参数说明：
            state: 车辆初始状态 [x, y, yaw, v]，仅在carla=False时使用
            actor: CARLA中的车辆Actor对象，carla=True时必填
            horizon: MPC预测时域长度（步数），决定"看多远"
            target_v: 目标纵向速度（km/h），会自动转换为m/s
            carla: 是否在CARLA环境中运行，True=CARLA仿真，False=纯仿真
            delta_t: 控制时间步长（秒），默认0.05s=20Hz
            max_iter: IPOPT优化器最大迭代次数
        """
        self.carla = carla
        global dt
        dt = delta_t  # 设置全局时间步长，供其他方法使用
        
        # -------------------------------------------
        # 从CARLA获取初始状态
        # -------------------------------------------
        if carla:
            self.actor = actor
            self.c_loc = self.actor.get_location()
            self.transform = self.actor.get_transform()
            yaw = self.transform.rotation.yaw * np.pi / 180  # 角度转弧度
            self.x = self.c_loc.x
            self.y = self.c_loc.y
            self.yaw = yaw
            self.vx = 0
            self.vy = 0
            self.direct = 1  # 行驶方向标志，1=前进
        else:
            # 纯仿真模式：直接使用传入的状态
            assert not type(state) is None
            assert len(state) == 4
            self.x, self.y, self.yaw, v = state
            self.vx = v * math.cos(self.yaw)
            self.vy = v * math.sin(self.yaw)
            self.direct = 1

        # -------------------------------------------
        # MPC控制器参数
        # -------------------------------------------
        self.horizon = horizon          # 预测时域
        self.maxiter = max_iter        # 优化器最大迭代次数
        
        # -------------------------------------------
        # 车辆动力学参数
        # -------------------------------------------
        self.omega = 0                  # 横摆角速度（rad/s）
        
        # 转向角约束（弧度）
        # 标定模式用1.22，实时控制模式用0.8（更保守）
        self.steer_bound = 0.7
        
        # 加速度约束（m/s²）
        self.acc_lbound = -5.0  # 最小加速度（制动）
        self.acc_ubound = 10.0  # 最大加速度（加速）
        
        # 目标速度（m/s），由km/h转换
        self.target_v = target_v / 3.6

        if self.carla and actor is not None:
            ego_extent = actor.bounding_box.extent
            self.ego_footprint_radius = max(float(ego_extent.y) + 0.2, 1.0)
            self.ego_footprint_half_length = max(float(ego_extent.x), 1.5)
        else:
            self.ego_footprint_radius = 1.0
            self.ego_footprint_half_length = 2.4
        self.obstacle_min_radius = 0.8
        self.obstacle_longitudinal_min_half = 2.2
        self.obstacle_influence_dist = 2.0
        self.obstacle_safe_dist = 0.65
        self.obstacle_cost_weight = 20.0
        self.obstacle_violation_weight = 850.0
        self.obstacle_side_cost_weight = 0.0
        self.obstacle_side_clearance = 3.2
        self.obstacle_bypass_start_distance = 10.0
        self.obstacle_bypass_full_distance = 1.5
        self.obstacle_return_front_clearance_lengths = 0.5
        self.obstacle_return_distance = 7.0
        self.obstacle_cluster_gap = 36.0
        self.obstacle_cluster_yaw_tolerance = 0.7
        self.obstacle_cluster_lateral_tolerance = 6.0
        self.obstacle_static_speed_threshold = 0.5
        self.obstacle_parallel_yaw_weight = 0.0
        self.obstacle_yaw_release_distance = 2.0
        self.lane_boundary_margin = 0.20
        self.lane_boundary_weight = 900.0
        self.lane_bound_inactive = 1.0e6
        # The imported CILQR narrow-corridor scenario spawns 12 static vehicles.
        # Keep the default MPC parameter slots large enough to cover that layout
        # without requiring per-run environment overrides.
        self.max_obstacles = int(os.environ.get("MPC_MAX_OBSTACLES", "16"))
        self.max_bypass_clusters = int(os.environ.get("MPC_MAX_BYPASS_CLUSTERS", "12"))
        self.obstacle_param_dim = 10
        self.bypass_cluster_param_dim = 7
        self.lane_bound_param_dim = 2
        self.dynamic_lane_guard_weight = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_WEIGHT", "140.0")
        )
        self.dynamic_lane_guard_hold_margin = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_HOLD_MARGIN", "0.45")
        )
        self.dynamic_lane_guard_same_lane_tolerance = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_SAME_LANE_TOL", "1.75")
        )
        self.dynamic_lane_guard_target_lane_tolerance = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_TARGET_LANE_TOL", "1.6")
        )
        self.dynamic_lane_guard_front_gap = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_FRONT_GAP", "14.0")
        )
        self.dynamic_lane_guard_rear_gap = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_REAR_GAP", "6.0")
        )
        self.dynamic_lane_guard_entry_start_distance = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_ENTRY_START", "22.0")
        )
        self.dynamic_lane_guard_entry_full_distance = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_ENTRY_FULL", "8.0")
        )
        self.dynamic_lane_guard_front_release_distance = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_FRONT_RELEASE", "2.0")
        )
        self.dynamic_lane_guard_release_distance = float(
            os.environ.get("MPC_DYNAMIC_LANE_GUARD_RELEASE", "8.0")
        )
        self.dynamic_overtake_lateral_weight = float(
            os.environ.get("MPC_DYNAMIC_OVERTAKE_LATERAL_WEIGHT", "26.0")
        )
        self.dynamic_obstacle_safe_margin = float(
            os.environ.get("MPC_DYNAMIC_OBSTACLE_SAFE_MARGIN", "0.0")
        )
        self.dynamic_obstacle_influence_margin = float(
            os.environ.get("MPC_DYNAMIC_OBSTACLE_INFLUENCE_MARGIN", "0.0")
        )
        self.dynamic_obstacle_distance_weight_scale = float(
            os.environ.get("MPC_DYNAMIC_OBSTACLE_DISTANCE_WEIGHT_SCALE", "1.0")
        )
        self.dynamic_obstacle_violation_weight_scale = float(
            os.environ.get("MPC_DYNAMIC_OBSTACLE_VIOLATION_WEIGHT_SCALE", "1.0")
        )
        self.previous_acc_cost_weight = 0.5
        self.previous_steer_cost_weight = 100.0
        self.steer_delta_bound = float(os.environ.get("MPC_STEER_DELTA_BOUND", "0.12"))
        self.first_step_steer_delta_bound = float(
            os.environ.get("MPC_FIRST_STEER_DELTA_BOUND", "0.18")
        )
        self.ipopt_max_cpu_time = float(os.environ.get("MPC_IPOPT_MAX_CPU_TIME", "0.08"))
        self.ipopt_tol = float(os.environ.get("MPC_IPOPT_TOL", "1e-5"))
        self.ipopt_acceptable_tol = float(os.environ.get("MPC_IPOPT_ACCEPTABLE_TOL", "5e-4"))
        self.ipopt_acceptable_obj_change_tol = float(
            os.environ.get("MPC_IPOPT_ACCEPTABLE_OBJ_CHANGE_TOL", "5e-4")
        )
        self.solver_initialized = False
        self.fallback_maxiter = max(
            self.maxiter,
            int(os.environ.get("MPC_FALLBACK_MAX_ITER", str(self.maxiter))),
        )
        self.solver_fallback = None
        self.last_min_obstacle_clearance = float("inf")

    def get_state_carla(self):
        """
        从CARLA仿真器获取当前车辆状态
        
        返回：
            state: [x, y, yaw, vx, vy, omega] 车辆状态向量
            z: 车辆高度（主要用于调试）
        """
        self.transform = self.actor.get_transform()
        self.x = self.transform.location.x
        self.y = self.transform.location.y
        self.z = self.transform.location.z
        yaw = self.transform.rotation.yaw
        self.yaw = yaw  # 注意：这里保持角度制，与弧度制混用需注意！

        # 获取速度
        self.velocity = self.actor.get_velocity()
        self.vx = np.sqrt(self.velocity.x**2 + self.velocity.y**2)
        self.vy = 0   # 二维简化，忽略侧向速度分量
        self.omega = self.actor.get_angular_velocity().z  # 绕z轴角速度

        return [self.x, self.y, self.yaw, self.vx, self.vy, self.omega], self.z

    def get_v(self):
        """
        获取车辆实际速度（考虑行驶方向）
        
        为什么要判断方向？
            因为CARLA中速度是向量，但在倒车时纵向速度可能为负
            这个函数用于确定车辆是前进还是后退
        
        返回：
            速度标量值（m/s），带符号表示方向
        """
        if self.carla:
            self.get_state_carla()
            return self.get_direction() * np.sqrt(self.vx**2+self.vy**2)
        else:
            return np.sqrt(self.vx**2+self.vy**2)

    def get_direction(self):
        """
        判断车辆行驶方向（前进还是后退）
        
        原理：
            比较速度向量方向和航向角方向
            如果夹角大于90度，说明在倒车
        
        返回：
            1=前进，-1=后退
        """
        yaw = np.radians(self.yaw)  # 角度转弧度
        v_yaw = math.atan2(self.velocity.y, self.velocity.x)  # 速度向量方向
        error = v_yaw - yaw
        
        # 角度归一化到[-π, π]
        if error < -math.pi:
            error += 2*math.pi
        elif error > math.pi:
            error -= 2*math.pi
        error = abs(error)
        
        # 夹角大于90度认为是倒车
        if error > math.pi/2:
            return -1
        return 1

    def set_state(self, next_state):
        """
        更新车辆状态（闭环控制时调用）
        
        参数：
            next_state: 新状态 [x, y, yaw, vx, vy, omega]
        """
        _, _, _, self.vx, self.vy, self.omega = next_state
        self.vy = -self.vy        # 符号调整（坐标系相关）
        self.omega = -self.omega

    def set_target_velocity(self, target_v):
        """
        设置目标速度
        
        参数：
            target_v: 目标速度（km/h）
        """
        self.target_v = target_v / 3.6  # km/h -> m/s

    def set_location_carla(self, location):
        """
        设置车辆在CARLA中的位置（用于重置车辆）
        
        参数：
            location: 位置信息，可以是 [x, y, z, roll, pitch, yaw] 或 [x, y, yaw]
        """
        if len(location) == 6:
            x, y, z, roll, pitch, yaw = location
        elif len(location) == 3:
            x, y, yaw = location
            z = roll = pitch = 0
        # 创建CARLA Transform并应用
        self.actor.set_transform(carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        ))

    def update(self, acc, steer):
        """
        基于自行车模型更新车辆状态（离散时间）
        
        自行车模型原理：
            将四轮车辆简化为两轮（自行车）
            前轮负责转向，后轮直行
        
        参数：
            acc: 控制输入，加速度（m/s²）
            steer: 控制输入，转向角（弧度）
        """
        # 位置更新（欧拉法）
        self.x += (self.vx*math.cos(self.yaw) - self.vy*math.sin(self.yaw)) * dt
        self.y += (self.vy*math.cos(self.yaw) + self.vx*math.sin(self.yaw)) * dt
        self.yaw += self.omega * dt
        
        # 纵向速度更新
        self.vx += dt * acc
        
        # 侧向速度更新（轮胎侧偏动力学）
        self.vy = (m*self.vx*self.vy + dt*Lk*self.omega 
                   - dt*kf*steer*self.vx - dt*m*(self.vx**2)*self.omega) / \
                  (m*self.vx - dt*(kf+kr))
        
        # 横摆角速度更新
        self.omega = (Iz*self.vx*self.omega + dt*Lk*self.vy 
                      - dt*lf*kf*steer*self.vx) / \
                     (Iz*self.vx - dt*(lf**2*kf + lr**2*kr))

    def predict(self, prev_state, u):
        """
        预测未来状态（用于MPC的轨迹预测）
        
        参数：
            prev_state: 当前状态 [x, y, yaw, vx, vy, omega]
            u: 控制输入 [acc, steer]
        
        返回：
            预测的下一状态
        """
        x, y, yaw, vx, vy, omega = prev_state
        acc = u[0]
        steer = u[1]

        # 状态预测（与update()相同）
        x += (vx*math.cos(yaw) - vy*math.sin(yaw)) * dt
        y += (vy*math.cos(yaw) + vx*math.sin(yaw)) * dt
        yaw += omega * dt
        vx += dt * acc
        vy = (m*vx*vy + dt*Lk*omega - dt*kf*steer*vx 
              - dt*m*(vx**2)*omega) / (m*vx - dt*(kf+kr))
        omega = (Iz*vx*omega + dt*Lk*vy - dt*lf*kf*steer*vx) / \
                (Iz*vx - dt*(lf*lf*kf + lr*lr*kr))

        return (x, y, yaw, vx, vy, omega)

    def solver_basis(self, Q=np.diag([3, 3, 1, 1, 0]), R=np.diag([1, 1]), Rd=np.diag([1, 10.0])):
        """
        构建MPC优化问题的数学基础（CasADi符号表达式）
        
        这个函数建立了MPC的优化问题框架：
            状态变量 X = [x, y, yaw, vx, vy, omega]   omega横摆角速度
            控制变量 U = [acc, steer]
        
        参数：
            Q: 状态跟踪权重矩阵，对角线元素越大越重视该状态的跟踪
            R: 控制代价权重矩阵，元素越大越抑制大的控制输入
            Rd: 控制增量权重矩阵，元素越大越追求控制平滑（抑制加加速度）
        
        优化目标：
            min Σ (state_errorᵀ Q state_error + uᵀ R u + Δuᵀ Rd Δu)
        
        默认权重设置：
            Q = diag([3, 3, 1, 1, 0])：位置跟踪最重要(3)，航向次之(1)，速度不太重要
            R = diag([1, 1])：加速度和转向同等对待
            Rd = diag([1, 10.0])：转向变化率惩罚更大（更追求平滑）
        """
        # -------------------------------------------
        # 定义状态变量（CasADi符号变量）
        # -------------------------------------------
        cx = ca.SX.sym('cx')      # x坐标，是创建一个叫 cx 的符号占位符
        cy = ca.SX.sym('cy')      # y坐标
        cyaw = ca.SX.sym('cyaw')  # 航向角
        cvx = ca.SX.sym('cvx')    # 纵向速度
        cvy = ca.SX.sym('cvy')    # 侧向速度
        comega = ca.SX.sym('comega')  # 横摆角速度

        # 拼接成状态向量，states 6行1列的列向量
        states = ca.vertcat(cx, cy, cyaw, cvx, cvy, comega)
        n_states = states.size()[0] # 从状态符号向量里取出它的行数（=6）
        self.n_states = n_states  # 状态维度 = 6

        # -------------------------------------------
        # 定义控制变量（CasADi符号变量）
        # -------------------------------------------
        cacc = ca.SX.sym('cacc')      # 加速度，
        csteer = ca.SX.sym('csteer')  # 转向角
        controls = ca.vertcat(cacc, csteer)
        n_controls = controls.size()[0]
        self.n_controls = n_controls  # 控制维度 = 2



        # -------------------------------------------
        # 定义系统动力学（状态更新方程）
        # -------------------------------------------
        #rhs是自行车模型的核心： 把车身坐标系的运动vx, vy, ω转换到世界坐标系，更新x, y, yaw，vx，vy，omega 6个状态。
        rhs = ca.vertcat(
            (cx + (cvx*ca.cos(cyaw) - cvy*ca.sin(cyaw))*dt),  # x更新
            (cy + (cvy*ca.cos(cyaw) + cvx*ca.sin(cyaw))*dt),        # y更新
            cyaw + comega*dt,                                       # yaw更新
        )
        
        if self.carla:
            # CARLA模式的动力学（含侧偏力）
            rhs = ca.vertcat(rhs, cvx + dt*cacc)  # vx更新
            rhs = ca.vertcat(rhs, (m*cvx*cvy + dt*Lk*comega - dt*kf*csteer*cvx 
                                   - dt*m*(cvx*cvx)*comega) / (m*cvx - dt*(kf+kr)))  # vy更新
            rhs = ca.vertcat(rhs, (Iz*cvx*comega + dt*Lk*cvy - dt*lf*kf*csteer*cvx) 
                             / (Iz*cvx - dt*(lf*lf*kf + lr*lr*kr)))  # omega更新
        else:
            # 纯仿真模式（相同）
            rhs = ca.vertcat(rhs, cvx + dt*cacc)
            rhs = ca.vertcat(rhs, (m*cvx*cvy + dt*Lk*comega - dt*kf*csteer*cvx 
                                   - dt*m*(cvx*cvx)*comega) / (m*cvx - dt*(kf+kr)))
            rhs = ca.vertcat(rhs, (Iz*cvx*comega + dt*Lk*cvy - dt*lf*kf*csteer*cvx) 
                             / (Iz*cvx - dt*(lf*lf*kf + lr*lr*kr)))



        # 创建CasADi函数：输入(状态, 控制) -> 输出(下一状态)
        #CasADi创建可微分的函数对象的核心步骤，把自行车模型的状态更新方程打包成一个 "黑盒" 函数。 f（[states, controls]）=[rhs]
        #self.f(...)可以直接调用；self.f(x0, u0) 能算出具体数值
        self.f = ca.Function('f', [states, controls], [rhs],
                             ['input_state', 'control_input'], ['rhs'])




        # -------------------------------------------
        # 定义MPC优化变量
        # -------------------------------------------
        # U: 控制序列 (n_controls × horizon)  2*10
        self.U = ca.SX.sym('U', n_controls, self.horizon)
        # X: 状态序列 (n_states × (horizon+1))，多一步初始状态
        self.X = ca.SX.sym('X', n_states, self.horizon+1)
        # P: 参考轨迹 (n_states × (horizon+1))
        self.P = ca.SX.sym('P', n_states, self.horizon+1)
        self.previous_control_param = ca.SX.sym('previous_control', n_controls)
        self.Obstacles = ca.SX.sym('Obstacles', self.obstacle_param_dim, self.max_obstacles)
        self.BypassClusters = ca.SX.sym(
            'BypassClusters',
            self.bypass_cluster_param_dim,
            self.max_bypass_clusters,
        )
        self.LaneBounds = ca.SX.sym(
            'LaneBounds',
            self.lane_bound_param_dim,
            self.horizon + 1,
        )
        self.solver_parameters = ca.vertcat(
            ca.reshape(self.P, -1, 1),
            self.previous_control_param,
            ca.reshape(self.Obstacles, -1, 1),
            ca.reshape(self.BypassClusters, -1, 1),
            ca.reshape(self.LaneBounds, -1, 1),
        )

        # 权重矩阵
        self.Q = Q
        self.Q_o = copy.deepcopy(Q)  # 保存原始Q用于动态调整
        self.R = R
        self.Rd = Rd

        # 合并所有优化变量：[U, X] 拉成一维向量. IPOPT 要求的标准格式
        #  (2T+6(T+1), 1)  [a0,,,aT δ0,,,δT x0,,,xT y,, yaw,, vx,, vy,, ω,,]
        self.opt_variables = ca.vertcat(
            ca.reshape(self.U, -1, 1),
            ca.reshape(self.X, -1, 1)
        )
        
        # 障碍物相关参数（用于扩展）
        #self.Da = 1
        #self.ac_r = 10      # 可通过区域势场权重
        #self.anc_r = 100    # 不可通过区域势场权重

    def get_obs_centers(self, ob_, radius=1.68, carla=False):
        """
        计算障碍物的两个等效圆心（用于圆碰撞检测）
        
        原理：将车辆简化为两个圆，方便做碰撞检测
        
        参数：
            ob_: 障碍物状态 [x, y, yaw]
            radius: 等效圆半径（车辆半长）
            carla: 是否CARLA模式
        
        返回：
            obc1: 前圆圆心 [x, y]
            obc2: 后圆圆心 [x, y]
        """
        if carla:
            # 在航向方向前后各一个圆心
            obc1 = [ob_[0] + radius*np.cos(ob_[2]), ob_[1] + radius*np.sin(ob_[2])]
            obc2 = [ob_[0] - radius*np.cos(ob_[2]), ob_[1] - radius*np.sin(ob_[2])]
        else:
            # 纯仿真模式：前圆+质心
            obc1 = [ob_[0] + radius*np.cos(ob_[2]), ob_[1] + radius*np.sin(ob_[2])]
            obc2 = [ob_[0], ob_[1]]

        return obc1, obc2

    def solver_add_c_road_pf(self, roads_pos, yaw=0, carla=False):
        """
        添加可穿越车道边界势场代价（可选功能）
        
        用于让车辆保持在车道内行驶
        
        参数：
            roads_pos: 车道边界y坐标列表 [y1, y2, y3, ...]
            yaw: 车辆航向角（用于坐标系转换）
            carla: 是否CARLA模式
        """
        for i in range(self.horizon):
            for road_pos in roads_pos:
                for selfc in self.get_obs_centers(self.X[:, i], carla):
                    # 转换到车辆局部坐标系
                    selfc = [0, ca.sin(-yaw)*selfc[0] + ca.cos(-yaw)*selfc[1]]
                    dist = ca.fabs(selfc[1] - road_pos)  # 横向距离

                # 势场代价：距离边界越近惩罚越大
                self.obj = self.obj + ca.if_else(
                    dist < 0.5,
                    self.ac_r * (dist-1)**2,
                    0)

    def c_road_pf(self, roads_pos, ref_traj, yaw=0, carla=False):
        """
        计算可穿越车道势场代价（数值版本，用于仿真）
        
        参数：
            roads_pos: 车道边界位置
            ref_traj: 参考轨迹
            yaw: 车辆航向角
            carla: 是否CARLA模式
        
        返回：
            road_pf: 势场代价标量
        """
        road_pf = 0
        for road_pos in roads_pos:
            for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                selfc = [0, np.sin(-yaw)*selfc[0] + np.cos(-yaw)*selfc[1]]
                dist = np.fabs(selfc[1] - road_pos)

            if dist < 1.5:
                road_pf += self.ac_r * (dist-1)**2

        return road_pf

    def solver_add_nc_road_pf(self, roads_pos, yaw=0, carla=False):
        """
        添加不可穿越障碍物势场代价（用于护栏、路边石等）
        
        与可穿越势场的区别：惩罚更重，使用倒数势场避免碰撞
        
        参数：
            roads_pos: 障碍物边界 [(y, dir), ...]，dir=1上方，dir=-1下方
        """
        for i in range(self.horizon):
            for road_pos, dir in roads_pos:
                for selfc in self.get_obs_centers(self.X[:, i], carla):
                    selfc = [0, (ca.sin(-yaw)*selfc[0] + ca.cos(-yaw)*selfc[1])]
                    dist = (selfc[1] - road_pos)**2

                # 倒数势场：距离越近惩罚越大（指数增长）
                self.obj = self.obj + \
                    ca.if_else(
                        dist < 2.25,
                        ca.if_else(dir == 1,
                            ca.if_else(selfc[1] > road_pos-0.2, 
                                    2501,  # 越过边界大惩罚
                                    self.anc_r * (1/(selfc[1] - road_pos))**2),
                            ca.if_else(selfc[1] < road_pos+0.2, 
                                    2501,
                                    self.anc_r * (1/(selfc[1] - road_pos))**2)
                        ),
                        0
                    )

    def nc_road_pf(self, roads_pos, ref_traj, yaw=0, carla=False):
        """
        计算不可穿越障碍物势场代价（数值版本）
        """
        nc_road_pf = 0
        for road_pos, dir in roads_pos:
            for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                selfc = [0, (np.sin(-yaw)*selfc[0] + np.cos(-yaw)*selfc[1])]
                dist = np.fabs(selfc[1] - road_pos)

                if dist < 1.5:
                    if dir == 1:
                        if selfc[1] > road_pos-0.2:
                            nc_road_pf += 1000
                        else:
                            nc_road_pf += self.anc_r * (1/np.fabs(selfc[1] - road_pos))**2
                    else:
                        if selfc[1] < road_pos+0.2:
                            nc_road_pf += 1000
                        else:
                            nc_road_pf += self.anc_r * (1/np.fabs(selfc[1] - road_pos))**2
        
        return nc_road_pf

    def solver_add_cost(self):
        """
        Build the fixed MPC objective and dynamics constraints.
        """
        self.obj = 0
        self.g = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []

        lane_bound_inactive_threshold = 0.5 * float(self.lane_bound_inactive)
        lane_boundary_weight = float(self.lane_boundary_weight)

        def lane_boundary_cost(state_slice, ref_slice, lane_bounds_slice):
            left_bound = lane_bounds_slice[0]
            right_bound = lane_bounds_slice[1]
            bounds_active = ca.if_else(
                ca.fmin(left_bound, right_bound) < lane_bound_inactive_threshold,
                1.0,
                0.0,
            )
            route_lateral = (
                (state_slice[0] - ref_slice[0]) * (-ca.sin(ref_slice[2]))
                + (state_slice[1] - ref_slice[1]) * ca.cos(ref_slice[2])
            )
            left_violation = ca.if_else(route_lateral > left_bound, route_lateral - left_bound, 0.0)
            right_limit = -right_bound
            right_violation = ca.if_else(route_lateral < right_limit, right_limit - route_lateral, 0.0)
            return bounds_active * lane_boundary_weight * (left_violation ** 2 + right_violation ** 2)

        self.g.append(self.X[:, 0] - self.P[:, 0])

        for i in range(self.horizon):
            yaw_error = ca.atan2(
                ca.sin(self.X[2, i] - self.P[2, i]),
                ca.cos(self.X[2, i] - self.P[2, i]),
            )
            state_error = ca.vertcat(
                self.X[0, i] - self.P[0, i],
                self.X[1, i] - self.P[1, i],
                yaw_error,
                self.X[3, i] - self.P[3, i],
                self.X[4, i] - self.P[4, i],
            )
            self.obj = self.obj + ca.mtimes([state_error.T, self.Q, state_error])
            self.obj = self.obj + ca.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
            if i < (self.horizon - 1):
                control_diff = self.U[:, i] - self.U[:, i + 1]
                self.obj = self.obj + ca.mtimes([control_diff.T, self.Rd, control_diff])
            self.obj = self.obj + lane_boundary_cost(
                self.X[:, i],
                self.P[:, i],
                self.LaneBounds[:, i],
            )
            x_next_ = self.f(self.X[:, i], self.U[:, i])
            self.g.append(self.X[:, i + 1] - x_next_)

        self.obj = self.obj + lane_boundary_cost(
            self.X[:, self.horizon],
            self.P[:, self.horizon],
            self.LaneBounds[:, self.horizon],
        )

    def get_self_centers(self, state_slice, radius=1.5):
        """
        获取车辆的两个圆心位置（前后各一个）
        state_slice: [x, y, yaw, vx, vy, omega] 当前时刻状态
        """
        x = state_slice[0]
        y = state_slice[1]
        yaw = state_slice[2]
        # 前圆心
        cx1 = x + radius * ca.cos(yaw)
        cy1 = y + radius * ca.sin(yaw)
        # 后圆心
        cx2 = x - radius * ca.cos(yaw)
        cy2 = y - radius * ca.sin(yaw)
        return (cx1, cy1), (cx2, cy2)
    
    def get_obs_centers_simple(self, obs_x, obs_y, obs_yaw, radius=1.68):
        """
        获取障碍物的两个圆心位置（前后各一个）
        对于静态障碍物，使用给定的yaw
        """
        # 前圆心
        cx1 = obs_x + radius * ca.cos(obs_yaw)
        cy1 = obs_y + radius * ca.sin(obs_yaw)
        # 后圆心
        cx2 = obs_x - radius * ca.cos(obs_yaw)
        cy2 = obs_y - radius * ca.sin(obs_yaw)
        return (cx1, cy1), (cx2, cy2)

    def _min_obstacle_clearance(self, x_seq, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return float("inf")

        obstacle_rows = np.asarray(obstacles, dtype=float)
        if obstacle_rows.ndim == 1:
            obstacle_rows = obstacle_rows.reshape(1, -1)

        state_rows = np.asarray(x_seq, dtype=float)
        if state_rows.ndim == 1:
            state_rows = state_rows.reshape(1, -1)

        min_clearance = float("inf")
        ego_radius = float(self.ego_footprint_radius)
        ego_half_length = float(self.ego_footprint_half_length)

        for step_index, state in enumerate(state_rows):
            ego_x = float(state[0])
            ego_y = float(state[1])
            ego_yaw = float(state[2])
            ego_centers = (
                (
                    ego_x + ego_half_length * math.cos(ego_yaw),
                    ego_y + ego_half_length * math.sin(ego_yaw),
                ),
                (
                    ego_x - ego_half_length * math.cos(ego_yaw),
                    ego_y - ego_half_length * math.sin(ego_yaw),
                ),
            )
            for obstacle in obstacle_rows:
                if obstacle.shape[0] < 5:
                    continue
                obs_vx = float(obstacle[8]) if obstacle.shape[0] > 8 else 0.0
                obs_vy = float(obstacle[9]) if obstacle.shape[0] > 9 else 0.0
                obs_x = float(obstacle[0]) + obs_vx * float(step_index) * float(dt)
                obs_y = float(obstacle[1]) + obs_vy * float(step_index) * float(dt)
                obs_yaw = float(obstacle[2]) if obstacle.shape[0] > 2 else 0.0
                obs_radius = max(
                    float(obstacle[3]) if obstacle.shape[0] > 3 else self.obstacle_min_radius,
                    float(self.obstacle_min_radius),
                )
                obs_half_length = max(
                    float(obstacle[4]) if obstacle.shape[0] > 4 else self.obstacle_longitudinal_min_half,
                    float(self.obstacle_longitudinal_min_half),
                )
                obs_centers = (
                    (
                        obs_x + obs_half_length * math.cos(obs_yaw),
                        obs_y + obs_half_length * math.sin(obs_yaw),
                    ),
                    (
                        obs_x - obs_half_length * math.cos(obs_yaw),
                        obs_y - obs_half_length * math.sin(obs_yaw),
                    ),
                )
                for ego_center in ego_centers:
                    for obs_center in obs_centers:
                        clearance = (
                            math.hypot(ego_center[0] - obs_center[0], ego_center[1] - obs_center[1])
                            - ego_radius
                            - obs_radius
                        )
                        min_clearance = min(min_clearance, clearance)
        return min_clearance
    
    def _pack_obstacle_parameters(self, obs):
        params = np.zeros((self.obstacle_param_dim, self.max_obstacles), dtype=float)
        if obs is None or len(obs) == 0:
            return params

        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[0] > self.max_obstacles:
            raise ValueError(
                f"Obstacle count {obs.shape[0]} exceeds MPC_MAX_OBSTACLES={self.max_obstacles}."
            )

        for index, row in enumerate(obs):
            params[0, index] = 1.0
            params[1, index] = float(row[0])
            params[2, index] = float(row[1])
            params[3, index] = float(row[2]) if row.shape[0] > 2 else 0.0
            params[4, index] = max(
                float(row[3]) if row.shape[0] > 3 else self.obstacle_min_radius,
                float(self.obstacle_min_radius),
            )
            params[5, index] = max(
                float(row[4]) if row.shape[0] > 4 else self.obstacle_longitudinal_min_half,
                float(self.obstacle_longitudinal_min_half),
            )
            params[6, index] = float(row[8]) if row.shape[0] > 8 else 0.0
            params[7, index] = float(row[9]) if row.shape[0] > 9 else 0.0
            params[8, index] = float(row[6]) if row.shape[0] > 6 else 1.0
            params[9, index] = float(row[7]) if row.shape[0] > 7 else 0.0
        return params

    def _build_bypass_clusters(self, obs):
        if obs is None or len(obs) == 0:
            return []

        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        def obstacle_value(row, index, default):
            return float(row[index]) if row.shape[0] > index else float(default)

        def angle_diff(a, b):
            return math.atan2(math.sin(a - b), math.cos(a - b))

        candidates = []
        for row in obs:
            obs_speed = obstacle_value(row, 5, 0.0)
            is_vehicle = obstacle_value(row, 6, 1.0)
            pass_lateral = obstacle_value(row, 7, 0.0)
            if is_vehicle < 0.5 or obs_speed > self.obstacle_static_speed_threshold:
                continue
            candidates.append(
                {
                    "x": float(row[0]),
                    "y": float(row[1]),
                    "yaw": obstacle_value(row, 2, 0.0),
                    "half_length": max(
                        obstacle_value(row, 4, self.obstacle_longitudinal_min_half),
                        float(self.obstacle_longitudinal_min_half),
                    ),
                    "pass_lateral": pass_lateral,
                }
            )

        clusters = []
        for candidate in candidates:
            merged = False
            for cluster in clusters:
                if abs(angle_diff(candidate["yaw"], cluster["yaw"])) > self.obstacle_cluster_yaw_tolerance:
                    continue
                if (
                    abs(candidate["pass_lateral"]) > 1e-6
                    and abs(cluster["pass_lateral"]) > 1e-6
                    and candidate["pass_lateral"] * cluster["pass_lateral"] <= 0.0
                ):
                    continue
                forward = cluster["forward"]
                left = cluster["left"]
                dx = candidate["x"] - cluster["origin_x"]
                dy = candidate["y"] - cluster["origin_y"]
                center_s = dx * forward[0] + dy * forward[1]
                center_l = dx * left[0] + dy * left[1]
                rear_s = center_s - candidate["half_length"]
                front_s = center_s + candidate["half_length"]
                gap = max(rear_s - cluster["front_s"], cluster["rear_s"] - front_s, 0.0)
                if abs(center_l) > self.obstacle_cluster_lateral_tolerance or gap > self.obstacle_cluster_gap:
                    continue
                cluster["rear_s"] = min(cluster["rear_s"], rear_s)
                cluster["front_s"] = max(cluster["front_s"], front_s)
                if abs(candidate["pass_lateral"]) > 1e-6:
                    if abs(cluster["pass_lateral"]) <= 1e-6:
                        cluster["pass_lateral"] = candidate["pass_lateral"]
                    else:
                        cluster["pass_lateral"] = math.copysign(
                            max(abs(cluster["pass_lateral"]), abs(candidate["pass_lateral"])),
                            cluster["pass_lateral"],
                        )
                merged = True
                break

            if not merged:
                forward = (math.cos(candidate["yaw"]), math.sin(candidate["yaw"]))
                left = (-math.sin(candidate["yaw"]), math.cos(candidate["yaw"]))
                clusters.append(
                    {
                        "origin_x": candidate["x"],
                        "origin_y": candidate["y"],
                        "yaw": candidate["yaw"],
                        "forward": forward,
                        "left": left,
                        "rear_s": -candidate["half_length"],
                        "front_s": candidate["half_length"],
                        "pass_lateral": candidate["pass_lateral"],
                    }
                )
        return [cluster for cluster in clusters if abs(cluster["pass_lateral"]) > 1e-6]

    def _pack_bypass_cluster_parameters(self, obs):
        clusters = self._build_bypass_clusters(obs)
        if len(clusters) > self.max_bypass_clusters:
            raise ValueError(
                f"Bypass cluster count {len(clusters)} exceeds "
                f"MPC_MAX_BYPASS_CLUSTERS={self.max_bypass_clusters}."
            )

        params = np.zeros((self.bypass_cluster_param_dim, self.max_bypass_clusters), dtype=float)
        for index, cluster in enumerate(clusters):
            params[0, index] = 1.0
            params[1, index] = float(cluster["origin_x"])
            params[2, index] = float(cluster["origin_y"])
            params[3, index] = float(cluster["yaw"])
            params[4, index] = float(cluster["rear_s"])
            params[5, index] = float(cluster["front_s"])
            params[6, index] = float(cluster["pass_lateral"])
        return params

    def _pack_lane_bound_parameters(self, lane_bounds):
        params = np.full(
            (self.lane_bound_param_dim, self.horizon + 1),
            float(self.lane_bound_inactive),
            dtype=float,
        )
        if lane_bounds is None:
            return params

        bounds_arr = np.asarray(lane_bounds, dtype=float)
        if bounds_arr.ndim == 1:
            if bounds_arr.size != self.lane_bound_param_dim:
                raise ValueError(
                    "Lane bounds must contain two values per step: [left_bound, right_bound]."
                )
            bounds_arr = np.tile(bounds_arr.reshape(1, -1), (self.horizon + 1, 1))
        if bounds_arr.shape == (self.lane_bound_param_dim, self.horizon + 1):
            bounds_arr = bounds_arr.T
        if bounds_arr.shape[1] != self.lane_bound_param_dim:
            raise ValueError(
                "Lane bounds must have shape (horizon+1, 2) or (2, horizon+1)."
            )
        if bounds_arr.shape[0] < self.horizon + 1:
            pad_count = self.horizon + 1 - bounds_arr.shape[0]
            bounds_arr = np.vstack((bounds_arr, np.repeat(bounds_arr[-1:, :], pad_count, axis=0)))
        else:
            bounds_arr = bounds_arr[: self.horizon + 1]

        for column in range(self.lane_bound_param_dim):
            values = bounds_arr[:, column]
            values = np.where(np.isfinite(values), values, float(self.lane_bound_inactive))
            values = np.where(values >= 0.0, values, float(self.lane_bound_inactive))
            params[column, :] = values
        return params

    def _runtime_parameters(self, z_ref, z0, previous_control, obstacles, lane_bounds):
        reference_params = np.vstack((z0.T, z_ref)).T
        obstacle_params = self._pack_obstacle_parameters(obstacles)
        cluster_params = self._pack_bypass_cluster_parameters(obstacles)
        lane_bound_params = self._pack_lane_bound_parameters(lane_bounds)
        return np.concatenate(
            (
                reference_params.reshape(-1, 1, order="F"),
                np.asarray(previous_control, dtype=float).reshape(-1, 1),
                obstacle_params.reshape(-1, 1, order="F"),
                cluster_params.reshape(-1, 1, order="F"),
                lane_bound_params.reshape(-1, 1, order="F"),
            ),
            axis=0,
        )

    def solver_add_parametric_soft_obs(self):
        ego_radius = float(self.ego_footprint_radius)
        ego_half_length = float(self.ego_footprint_half_length)
        influence_dist = float(self.obstacle_influence_dist)
        safe_dist = float(self.obstacle_safe_dist)
        distance_weight = float(self.obstacle_cost_weight)
        violation_weight = float(self.obstacle_violation_weight)
        side_weight = float(self.obstacle_side_cost_weight)
        side_clearance = float(self.obstacle_side_clearance)
        bypass_start_distance = float(self.obstacle_bypass_start_distance)
        bypass_full_distance = float(self.obstacle_bypass_full_distance)
        return_front_clearance_lengths = float(self.obstacle_return_front_clearance_lengths)
        return_distance = float(self.obstacle_return_distance)
        yaw_weight = float(self.obstacle_parallel_yaw_weight)
        yaw_release_distance = float(self.obstacle_yaw_release_distance)
        dynamic_lane_guard_weight = float(self.dynamic_lane_guard_weight)
        dynamic_lane_guard_hold_margin = float(self.dynamic_lane_guard_hold_margin)
        dynamic_lane_guard_same_lane_tolerance = float(
            self.dynamic_lane_guard_same_lane_tolerance
        )
        dynamic_lane_guard_target_lane_tolerance = float(
            self.dynamic_lane_guard_target_lane_tolerance
        )
        dynamic_lane_guard_front_gap = float(self.dynamic_lane_guard_front_gap)
        dynamic_lane_guard_rear_gap = float(self.dynamic_lane_guard_rear_gap)
        dynamic_lane_guard_entry_start_distance = float(
            self.dynamic_lane_guard_entry_start_distance
        )
        dynamic_lane_guard_entry_full_distance = float(
            self.dynamic_lane_guard_entry_full_distance
        )
        dynamic_lane_guard_front_release_distance = float(
            self.dynamic_lane_guard_front_release_distance
        )
        dynamic_lane_guard_release_distance = float(
            self.dynamic_lane_guard_release_distance
        )
        dynamic_overtake_lateral_weight = float(
            self.dynamic_overtake_lateral_weight
        )
        dynamic_obstacle_safe_margin = float(self.dynamic_obstacle_safe_margin)
        dynamic_obstacle_influence_margin = float(self.dynamic_obstacle_influence_margin)
        dynamic_obstacle_distance_weight_scale = float(
            self.dynamic_obstacle_distance_weight_scale
        )
        dynamic_obstacle_violation_weight_scale = float(
            self.dynamic_obstacle_violation_weight_scale
        )
        dynamic_lane_guard_speed_threshold = float(self.obstacle_static_speed_threshold)

        def smoothstep(value):
            return value * value * (3.0 - 2.0 * value)

        def bypass_profiles(rel_long, rear_s, front_s):
            entry_start = rear_s - ego_half_length - bypass_start_distance
            entry_full = rear_s - ego_half_length - bypass_full_distance
            return_start = front_s + max((2.0 * return_front_clearance_lengths - 1.0) * ego_half_length, 0.0)
            return_end = return_start + return_distance
            yaw_return_start = front_s
            yaw_return_end = yaw_return_start + yaw_release_distance
            entry_ratio = (rel_long - entry_start) / max(
                bypass_start_distance - bypass_full_distance,
                1e-6,
            )
            return_ratio = (rel_long - return_start) / max(return_distance, 1e-6)
            yaw_return_ratio = (rel_long - yaw_return_start) / max(yaw_release_distance, 1e-6)
            entry_profile = ca.if_else(
                rel_long <= entry_start,
                0.0,
                ca.if_else(
                    rel_long < entry_full,
                    smoothstep(entry_ratio),
                    ca.if_else(
                        rel_long <= return_start,
                        1.0,
                        ca.if_else(rel_long < return_end, 1.0 - smoothstep(return_ratio), 0.0),
                    ),
                ),
            )
            yaw_profile = ca.if_else(
                rel_long <= entry_start,
                0.0,
                ca.if_else(
                    rel_long < entry_full,
                    smoothstep(entry_ratio),
                    ca.if_else(
                        rel_long <= yaw_return_start,
                        1.0,
                        ca.if_else(rel_long < yaw_return_end, 1.0 - smoothstep(yaw_return_ratio), 0.0),
                    ),
                ),
            )
            return entry_profile, yaw_profile

        def dynamic_lane_guard_profile(rel_long, obs_half_length):
            entry_start = (
                -obs_half_length
                - ego_half_length
                - dynamic_lane_guard_entry_start_distance
            )
            entry_full = (
                -obs_half_length
                - ego_half_length
                - dynamic_lane_guard_entry_full_distance
            )
            release_start = (
                obs_half_length
                + ego_half_length
                + dynamic_lane_guard_front_release_distance
            )
            release_end = release_start + dynamic_lane_guard_release_distance
            entry_ratio = (rel_long - entry_start) / max(
                dynamic_lane_guard_entry_start_distance
                - dynamic_lane_guard_entry_full_distance,
                1e-6,
            )
            release_ratio = (rel_long - release_start) / max(
                dynamic_lane_guard_release_distance,
                1e-6,
            )
            return ca.if_else(
                rel_long <= entry_start,
                0.0,
                ca.if_else(
                    rel_long < entry_full,
                    smoothstep(entry_ratio),
                    ca.if_else(
                        rel_long <= release_start,
                        1.0,
                        ca.if_else(
                            rel_long < release_end,
                            1.0 - smoothstep(release_ratio),
                            0.0,
                        ),
                    ),
                ),
            )

        for i in range(self.horizon):
            state_i = self.X[:, i]
            ego_centers = self.get_self_centers(state_i, ego_half_length)

            for j in range(self.max_obstacles):
                active = self.Obstacles[0, j]
                obs_x0 = self.Obstacles[1, j]
                obs_y0 = self.Obstacles[2, j]
                obs_yaw = self.Obstacles[3, j]
                obs_radius = self.Obstacles[4, j]
                obs_half_length = self.Obstacles[5, j]
                obs_vx = self.Obstacles[6, j]
                obs_vy = self.Obstacles[7, j]
                obs_is_vehicle = self.Obstacles[8, j]
                obs_pass_lateral = self.Obstacles[9, j]
                obs_x = obs_x0 + obs_vx * (i * dt)
                obs_y = obs_y0 + obs_vy * (i * dt)
                obs_speed = ca.sqrt(obs_vx ** 2 + obs_vy ** 2 + 1e-6)
                dynamic_obstacle_factor = ca.if_else(
                    obs_speed > dynamic_lane_guard_speed_threshold,
                    1.0,
                    0.0,
                )
                influence_dist_eff = influence_dist + dynamic_obstacle_factor * dynamic_obstacle_influence_margin
                safe_dist_eff = safe_dist + dynamic_obstacle_factor * dynamic_obstacle_safe_margin
                distance_weight_eff = distance_weight * (
                    1.0
                    + dynamic_obstacle_factor
                    * max(dynamic_obstacle_distance_weight_scale - 1.0, 0.0)
                )
                violation_weight_eff = violation_weight * (
                    1.0
                    + dynamic_obstacle_factor
                    * max(dynamic_obstacle_violation_weight_scale - 1.0, 0.0)
                )
                obs_forward_x = ca.cos(obs_yaw)
                obs_forward_y = ca.sin(obs_yaw)
                obs_left_x = -obs_forward_y
                obs_left_y = obs_forward_x
                obs_centers = self.get_obs_centers_simple(obs_x, obs_y, obs_yaw, obs_half_length)

                for ego_center in ego_centers:
                    for obs_center in obs_centers:
                        dist = ca.sqrt(
                            (ego_center[0] - obs_center[0]) ** 2
                            + (ego_center[1] - obs_center[1]) ** 2
                            + 1e-6
                        )
                        clearance = dist - ego_radius - obs_radius
                        distance_cost = ca.if_else(
                            clearance < influence_dist_eff,
                            distance_weight_eff * (influence_dist_eff - clearance) ** 2,
                            0.0,
                        )
                        violation_cost = ca.if_else(
                            clearance < safe_dist_eff,
                            violation_weight_eff * (safe_dist_eff - clearance) ** 2,
                            0.0,
                        )
                        self.obj += active * (distance_cost + violation_cost) / (i + 1)

                ego_obs_dx = state_i[0] - obs_x
                ego_obs_dy = state_i[1] - obs_y
                ego_obs_long = (
                    ego_obs_dx * obs_forward_x + ego_obs_dy * obs_forward_y
                )
                ego_obs_lateral = (
                    ego_obs_dx * obs_left_x + ego_obs_dy * obs_left_y
                )
                same_lane_ratio = ca.fmax(
                    0.0,
                    1.0 - ca.fabs(ego_obs_lateral)
                    / max(dynamic_lane_guard_same_lane_tolerance, 1e-6),
                )
                same_lane_factor = smoothstep(same_lane_ratio)
                target_lane_defined = ca.if_else(ca.fabs(obs_pass_lateral) > 1e-6, 1.0, 0.0)
                dynamic_vehicle_active = active * obs_is_vehicle * ca.if_else(
                    obs_speed > dynamic_lane_guard_speed_threshold,
                    1.0,
                    0.0,
                )
                lane_guard_profile = dynamic_lane_guard_profile(
                    ego_obs_long,
                    obs_half_length,
                )
                lateral_sign = ca.if_else(obs_pass_lateral >= 0.0, 1.0, -1.0)
                move_toward_target = ca.fmax(
                    0.0,
                    lateral_sign * ego_obs_lateral - dynamic_lane_guard_hold_margin,
                )
                candidate_rear = -obs_half_length
                candidate_front = obs_half_length
                corridor_start = ego_obs_long - ego_half_length - dynamic_lane_guard_rear_gap
                corridor_end = candidate_front + dynamic_lane_guard_front_gap
                corridor_norm = max(
                    dynamic_lane_guard_rear_gap
                    + dynamic_lane_guard_front_gap
                    + 2.0 * ego_half_length,
                    1e-6,
                )
                target_lane_block_score = 0.0

                for m in range(self.max_obstacles):
                    if m == j:
                        continue
                    other_active = self.Obstacles[0, m]
                    other_x0 = self.Obstacles[1, m]
                    other_y0 = self.Obstacles[2, m]
                    other_half_length = self.Obstacles[5, m]
                    other_vx = self.Obstacles[6, m]
                    other_vy = self.Obstacles[7, m]
                    other_is_vehicle = self.Obstacles[8, m]
                    other_x = other_x0 + other_vx * (i * dt)
                    other_y = other_y0 + other_vy * (i * dt)
                    other_speed = ca.sqrt(other_vx ** 2 + other_vy ** 2 + 1e-6)
                    other_obs_dx = other_x - obs_x
                    other_obs_dy = other_y - obs_y
                    other_obs_long = (
                        other_obs_dx * obs_forward_x
                        + other_obs_dy * obs_forward_y
                    )
                    other_obs_lateral = (
                        other_obs_dx * obs_left_x
                        + other_obs_dy * obs_left_y
                    )
                    lane_match_ratio = ca.fmax(
                        0.0,
                        1.0 - ca.fabs(other_obs_lateral - obs_pass_lateral)
                        / max(dynamic_lane_guard_target_lane_tolerance, 1e-6),
                    )
                    lane_match_factor = smoothstep(lane_match_ratio)
                    other_rear = other_obs_long - other_half_length
                    other_front = other_obs_long + other_half_length
                    overlap_start = ca.fmax(corridor_start, other_rear)
                    overlap_end = ca.fmin(corridor_end, other_front)
                    overlap = ca.fmax(0.0, overlap_end - overlap_start)
                    target_lane_block_score += (
                        other_active
                        * other_is_vehicle
                        * ca.if_else(
                            other_speed > dynamic_lane_guard_speed_threshold,
                            1.0,
                            0.0,
                        )
                        * lane_match_factor
                        * (overlap / corridor_norm)
                    )

                self.obj += (
                    dynamic_vehicle_active
                    * target_lane_defined
                    * same_lane_factor
                    * lane_guard_profile
                    * dynamic_lane_guard_weight
                    * target_lane_block_score
                    * move_toward_target ** 2
                ) / (i + 1)
                lane_open_ratio = ca.fmax(0.0, 1.0 - target_lane_block_score)
                lane_open_factor = smoothstep(lane_open_ratio)
                desired_dynamic_lateral = obs_pass_lateral
                lateral_progress_error = ego_obs_lateral - lane_guard_profile * desired_dynamic_lateral
                self.obj += (
                    dynamic_vehicle_active
                    * target_lane_defined
                    * same_lane_factor
                    * lane_open_factor
                    * dynamic_overtake_lateral_weight
                    * lateral_progress_error ** 2
                ) / (i + 1)

            for j in range(self.max_bypass_clusters):
                active = self.BypassClusters[0, j]
                origin_x = self.BypassClusters[1, j]
                origin_y = self.BypassClusters[2, j]
                cluster_yaw = self.BypassClusters[3, j]
                rear_s = self.BypassClusters[4, j]
                front_s = self.BypassClusters[5, j]
                pass_lateral = self.BypassClusters[6, j]
                forward_x = ca.cos(cluster_yaw)
                forward_y = ca.sin(cluster_yaw)
                left_x = -ca.sin(cluster_yaw)
                left_y = ca.cos(cluster_yaw)
                dx = state_i[0] - origin_x
                dy = state_i[1] - origin_y
                rel_long = dx * forward_x + dy * forward_y
                rel_lat = dx * left_x + dy * left_y
                lateral_sign = ca.if_else(pass_lateral >= 0.0, 1.0, -1.0)
                desired_lateral = lateral_sign * ca.if_else(
                    ca.fabs(pass_lateral) > side_clearance,
                    side_clearance,
                    ca.fabs(pass_lateral),
                )
                lateral_error = rel_lat - desired_lateral
                yaw_error = ca.atan2(
                    ca.sin(state_i[2] - cluster_yaw),
                    ca.cos(state_i[2] - cluster_yaw),
                )
                lateral_profile, yaw_profile = bypass_profiles(rel_long, rear_s, front_s)
                self.obj += active * lateral_profile * side_weight * lateral_error ** 2
                self.obj += active * yaw_profile * yaw_weight * yaw_error ** 2

    def initialize_solver(self):
        if self.solver_initialized:
            return
        self.solver_add_cost()
        self.solver_add_parametric_soft_obs()
        self.solver_add_bounds_fixed()
        self.solver_initialized = True

    def solver_add_bounds_fixed(self):
        previous_control_diff = self.U[:, 0] - self.previous_control_param
        self.obj += self.previous_acc_cost_weight * previous_control_diff[0] ** 2
        self.obj += self.previous_steer_cost_weight * previous_control_diff[1] ** 2

        self.g.append(self.U[:, 0] - self.previous_control_param)

        for i in range(self.horizon - 1):
            self.g.append(self.U[0, i + 1] - self.U[0, i])
            self.g.append(self.U[1, i + 1] - self.U[1, i])

        nlp_prob = {
            'f': self.obj,
            'x': self.opt_variables,
            'p': self.solver_parameters,
            'g': ca.vertcat(*self.g),
        }
        opts_setting = {
            'ipopt.max_iter': self.maxiter,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': self.ipopt_tol,
            'ipopt.acceptable_tol': self.ipopt_acceptable_tol,
            'ipopt.acceptable_obj_change_tol': self.ipopt_acceptable_obj_change_tol,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.max_cpu_time': self.ipopt_max_cpu_time,
            'ipopt.sb': 'yes',
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
        if self.fallback_maxiter > self.maxiter:
            fallback_opts = dict(opts_setting)
            fallback_opts['ipopt.max_iter'] = self.fallback_maxiter
            self.solver_fallback = ca.nlpsol('solver_fallback', 'ipopt', nlp_prob, fallback_opts)
        else:
            self.solver_fallback = self.solver

        for _ in range(self.horizon + 1):
            for _ in range(self.n_states):
                self.lbg.append(0.0)
                self.ubg.append(0.0)

        self.lbg.extend([-2.0, -self.first_step_steer_delta_bound])
        self.ubg.extend([2.0, self.first_step_steer_delta_bound])
        for _ in range(self.horizon - 1):
            self.lbg.extend([-0.9, -self.steer_delta_bound])
            self.ubg.extend([0.9, self.steer_delta_bound])

        acc_lb = getattr(self, '_temp_acc_lbound', self.acc_lbound)
        for _ in range(self.horizon):
            self.lbx.extend([acc_lb, -self.steer_bound])
            self.ubx.extend([self.acc_ubound, self.steer_bound])

        if hasattr(self, '_temp_acc_lbound'):
            delattr(self, '_temp_acc_lbound')

        for _ in range(self.horizon + 1):
            self.lbx.extend([-np.inf, -np.inf, -np.inf, -self.target_v, -np.inf, -np.inf])
            self.ubx.extend([np.inf, np.inf, np.inf, self.target_v, np.inf, np.inf])

    def solver_add_soft_obs(self, obs=None, ratio=500, expn=1):
        """
        添加软障碍物避让代价。

        obs 每行格式：
            [x, y, yaw_rad, radius, half_length, speed, is_vehicle, pass_lateral]
        """
        import os
        log_dir = ensure_debug_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "run_debug.log")
        
        def log(msg):
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        
        if obs is None or (isinstance(obs, list) and len(obs) == 0):
            log("[DEBUG solver_add_soft_obs] obs为空，跳过")
            return
        
        obs = np.array(obs)  # 转为numpy数组
        n_obs = obs.shape[0]
        pass_laterals = obs[:, 7].tolist() if obs.shape[1] > 7 else []
        log(f"[DEBUG solver_add_soft_obs] n_obs={n_obs}, ratio={ratio}, pass_laterals={pass_laterals}")

        ego_radius = float(self.ego_footprint_radius)
        ego_half_length = float(self.ego_footprint_half_length)
        influence_dist = float(self.obstacle_influence_dist)
        safe_dist = float(self.obstacle_safe_dist)
        distance_weight = float(self.obstacle_cost_weight)
        violation_weight = float(self.obstacle_violation_weight)
        side_weight = float(self.obstacle_side_cost_weight)
        side_clearance = float(self.obstacle_side_clearance)
        bypass_start_distance = float(self.obstacle_bypass_start_distance)
        bypass_full_distance = float(self.obstacle_bypass_full_distance)
        return_front_clearance_lengths = float(self.obstacle_return_front_clearance_lengths)
        return_distance = float(self.obstacle_return_distance)
        cluster_gap = float(self.obstacle_cluster_gap)
        cluster_yaw_tolerance = float(self.obstacle_cluster_yaw_tolerance)
        cluster_lateral_tolerance = float(self.obstacle_cluster_lateral_tolerance)
        static_speed_threshold = float(self.obstacle_static_speed_threshold)
        yaw_weight = float(self.obstacle_parallel_yaw_weight)
        yaw_release_distance = float(self.obstacle_yaw_release_distance)

        def smoothstep(value):
            return value * value * (3.0 - 2.0 * value)

        def angle_diff(a, b):
            return math.atan2(math.sin(a - b), math.cos(a - b))

        def obstacle_value(row, index, default):
            return float(row[index]) if obs.shape[1] > index else float(default)

        def build_bypass_clusters():
            candidates = []
            for obs_index in range(n_obs):
                row = obs[obs_index]
                obs_speed = obstacle_value(row, 5, 0.0)
                is_vehicle = obstacle_value(row, 6, 1.0)
                pass_lateral = obstacle_value(row, 7, 0.0)
                if is_vehicle < 0.5 or obs_speed > static_speed_threshold:
                    continue
                candidates.append(
                    {
                        "x": float(row[0]),
                        "y": float(row[1]),
                        "yaw": obstacle_value(row, 2, 0.0),
                        "half_length": max(
                            obstacle_value(row, 4, self.obstacle_longitudinal_min_half),
                            float(self.obstacle_longitudinal_min_half),
                        ),
                        "pass_lateral": pass_lateral,
                    }
                )

            clusters = []
            for candidate in candidates:
                merged = False
                for cluster in clusters:
                    if abs(angle_diff(candidate["yaw"], cluster["yaw"])) > cluster_yaw_tolerance:
                        continue
                    if (
                        abs(candidate["pass_lateral"]) > 1e-6
                        and abs(cluster["pass_lateral"]) > 1e-6
                        and candidate["pass_lateral"] * cluster["pass_lateral"] <= 0.0
                    ):
                        continue
                    forward = cluster["forward"]
                    left = cluster["left"]
                    dx = candidate["x"] - cluster["origin_x"]
                    dy = candidate["y"] - cluster["origin_y"]
                    center_s = dx * forward[0] + dy * forward[1]
                    center_l = dx * left[0] + dy * left[1]
                    rear_s = center_s - candidate["half_length"]
                    front_s = center_s + candidate["half_length"]
                    gap = max(rear_s - cluster["front_s"], cluster["rear_s"] - front_s, 0.0)
                    if abs(center_l) > cluster_lateral_tolerance or gap > cluster_gap:
                        continue
                    cluster["rear_s"] = min(cluster["rear_s"], rear_s)
                    cluster["front_s"] = max(cluster["front_s"], front_s)
                    if abs(candidate["pass_lateral"]) > 1e-6:
                        if abs(cluster["pass_lateral"]) <= 1e-6:
                            cluster["pass_lateral"] = candidate["pass_lateral"]
                        else:
                            cluster["pass_lateral"] = math.copysign(
                                max(abs(cluster["pass_lateral"]), abs(candidate["pass_lateral"])),
                                cluster["pass_lateral"],
                            )
                    merged = True
                    break

                if not merged:
                    forward = (math.cos(candidate["yaw"]), math.sin(candidate["yaw"]))
                    left = (-math.sin(candidate["yaw"]), math.cos(candidate["yaw"]))
                    clusters.append(
                        {
                            "origin_x": candidate["x"],
                            "origin_y": candidate["y"],
                            "yaw": candidate["yaw"],
                            "forward": forward,
                            "left": left,
                            "rear_s": -candidate["half_length"],
                            "front_s": candidate["half_length"],
                            "pass_lateral": candidate["pass_lateral"],
                        }
                    )
            return [cluster for cluster in clusters if abs(cluster["pass_lateral"]) > 1e-6]

        bypass_clusters = build_bypass_clusters()
        log(f"[DEBUG solver_add_soft_obs] bypass_clusters={len(bypass_clusters)}")

        def bypass_profiles(rel_long, rear_s, front_s):
            entry_start = rear_s - ego_half_length - bypass_start_distance
            entry_full = rear_s - ego_half_length - bypass_full_distance
            return_start = front_s + max((2.0 * return_front_clearance_lengths - 1.0) * ego_half_length, 0.0)
            return_end = return_start + return_distance
            yaw_return_start = front_s
            yaw_return_end = yaw_return_start + yaw_release_distance
            entry_ratio = (rel_long - entry_start) / max(entry_full - entry_start, 1e-6)
            return_ratio = (rel_long - return_start) / max(return_end - return_start, 1e-6)
            yaw_return_ratio = (rel_long - yaw_return_start) / max(yaw_return_end - yaw_return_start, 1e-6)
            entry_shape = smoothstep(entry_ratio)
            return_shape = 1.0 - smoothstep(return_ratio)
            lateral_profile = ca.if_else(
                rel_long <= entry_start,
                0.0,
                ca.if_else(
                    rel_long < entry_full,
                    entry_shape,
                    ca.if_else(
                        rel_long <= return_start,
                        1.0,
                        ca.if_else(rel_long < return_end, return_shape, 0.0),
                    ),
                ),
            )
            yaw_profile = ca.if_else(
                rel_long <= entry_start,
                0.0,
                ca.if_else(
                    rel_long < entry_full,
                    entry_shape,
                    ca.if_else(
                        rel_long <= yaw_return_start,
                        1.0,
                        ca.if_else(rel_long < yaw_return_end, 1.0 - smoothstep(yaw_return_ratio), 0.0),
                    ),
                ),
            )
            return lateral_profile, yaw_profile
        
        for i in range(self.horizon):  # 每个预测步
            state_i = self.X[:, i]
            ego_centers = self.get_self_centers(state_i, ego_half_length)
            
            for j in range(n_obs):  # 每个障碍物
                obs_x = float(obs[j, 0])
                obs_y = float(obs[j, 1])
                obs_yaw = float(obs[j, 2]) if obs.shape[1] > 2 else 0.0
                obs_radius = max(
                    float(obs[j, 3]) if obs.shape[1] > 3 else self.obstacle_min_radius,
                    float(self.obstacle_min_radius),
                )
                obs_half_length = max(
                    float(obs[j, 4]) if obs.shape[1] > 4 else self.obstacle_longitudinal_min_half,
                    float(self.obstacle_longitudinal_min_half),
                )

                obs_centers = self.get_obs_centers_simple(obs_x, obs_y, obs_yaw, obs_half_length)
                for ego_center in ego_centers:
                    for obs_center in obs_centers:
                        dist = ca.sqrt(
                            (ego_center[0] - obs_center[0]) ** 2
                            + (ego_center[1] - obs_center[1]) ** 2
                            + 1e-6
                        )
                        clearance = dist - ego_radius - obs_radius
                        distance_cost = ca.if_else(
                            clearance < influence_dist,
                            distance_weight * (influence_dist - clearance) ** 2,
                            0.0,
                        )
                        violation_cost = ca.if_else(
                            clearance < safe_dist,
                            violation_weight * (safe_dist - clearance) ** 2,
                            0.0,
                        )
                        self.obj += (distance_cost + violation_cost) / (i + 1)

            for cluster in bypass_clusters:
                dx = state_i[0] - cluster["origin_x"]
                dy = state_i[1] - cluster["origin_y"]
                rel_long = dx * cluster["forward"][0] + dy * cluster["forward"][1]
                rel_lat = dx * cluster["left"][0] + dy * cluster["left"][1]
                lateral_profile, yaw_profile = bypass_profiles(rel_long, cluster["rear_s"], cluster["front_s"])
                required_lateral = side_clearance
                desired_lateral = math.copysign(
                    min(abs(cluster["pass_lateral"]), required_lateral),
                    cluster["pass_lateral"],
                )
                lateral_error = rel_lat - desired_lateral
                yaw_error = ca.atan2(
                    ca.sin(state_i[2] - cluster["yaw"]),
                    ca.cos(state_i[2] - cluster["yaw"]),
                )
                self.obj += lateral_profile * side_weight * lateral_error ** 2
                self.obj += yaw_profile * yaw_weight * yaw_error ** 2
        
        log("[DEBUG solver_add_soft_obs] safety+bypass obstacle cost added")

    def solver_add_bounds(self, u00=None):
        """
        设置优化变量的边界约束
        
        包含：
            - 初始状态与动力学约束（等式，松弛后不等式）
            - 控制输入约束（不等式）
            - 状态约束（不等式）
            - jerk约束/控制增量约束（不等式）
        
        参数：
            u00: 上一时刻的控制输入（用于控制平滑约束）
        """
        # ---- 添加 jerk约束（控制增量约束）----
        return self.solver_add_bounds_fixed()
        if False:
            # 第一个控制增量约束
            self.g.append(self.U[:, 0] - u00)
        
        self.g.append(self.U[:, 0] - self.previous_control_param)

        for i in range(self.horizon-1):
            # 加速度增量约束（-0.9 ~ 0.9 m/s³）
            self.g.append((self.U[0, i+1] - self.U[0, i]))
            # 转向角增量约束（-0.2 ~ 0.2 rad）
            self.g.append((self.U[1, i+1] - self.U[1, i]))

        # ---- 构建NLP问题 ----
        nlp_prob = {
            'f': self.obj,           # 目标函数
            'x': self.opt_variables, # 优化变量 [U; X]
            'p': self.P,             # 参数（参考轨迹）
            'g': ca.vertcat(*self.g)  # 约束
        }

        # ---- IPOPT求解器设置 ----
        opts_setting = {
            'ipopt.max_iter': self.maxiter,          # 最大迭代次数
            'ipopt.print_level': 0,                  # 输出级别（0=静默）
            'print_time': 0,                         # 打印求解时间
            'ipopt.acceptable_tol': 1e-8,            # 可接受容忍度
            'ipopt.acceptable_obj_change_tol': 1e-6  # 目标变化容忍度
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        # ---- 设置约束边界 ----
        
        # 动力学约束边界（等式约束，0=0）
        for _ in range(self.horizon+1):
            for _ in range(self.n_states):
                self.lbg.append(0.0)
                self.ubg.append(0.0)

        # 第一个控制增量约束（与上一时刻的连续性）
        # 【修复振荡】减小转向角增量约束
        if False:
            self.lbg.append(-2.0)   # 加速度增量下界
            self.lbg.append(-0.08)  # 转向角增量下界
            self.ubg.append(2.0)    # 加速度增量上界
            self.ubg.append(0.08)   # 转向角增量上界
        
        # jerk约束（控制增量）
        # 【修复振荡】减小转向角增量约束，从±0.2改为±0.1 rad
        for _ in range(self.horizon-1):
            self.lbg.append(-0.9)  # 加速度增量
            self.lbg.append(-0.08)  # 转向角增量
            self.ubg.append(0.9)
            self.ubg.append(0.08)

        # ---- 控制输入约束 ----
        # 如果有临时刹车约束（避障时），使用宽松的刹车约束
        temp_acc_lbound = getattr(self, '_temp_acc_lbound', None)
        acc_lb = temp_acc_lbound if temp_acc_lbound is not None else self.acc_lbound
        
        for _ in range(self.horizon):
            self.lbx.append(acc_lb)  # 动态刹车约束
            self.lbx.append(-self.steer_bound)  # -0.7 rad
            self.ubx.append(self.acc_ubound)   # 3 m/s²
            self.ubx.append(self.steer_bound)  # 0.7 rad
        
        # 用完后清除临时约束
        if hasattr(self, '_temp_acc_lbound'):
            delattr(self, '_temp_acc_lbound')

        # ---- 状态约束（速度下界实现向前行驶）----
        for _ in range(self.horizon+1):
            self.lbx.append(-np.inf)  # x
            self.lbx.append(-np.inf)  # y
            self.lbx.append(-np.inf)  # yaw
            self.lbx.append(-self.target_v)  # vx >= -target_v (允许倒车但限速)
            self.lbx.append(-np.inf)  # vy
            self.lbx.append(-np.inf)  # omega

            self.ubx.append(np.inf)
            self.ubx.append(np.inf)
            self.ubx.append(np.inf)
            self.ubx.append(self.target_v)  # vx <= target_v
            self.ubx.append(np.inf)
            self.ubx.append(np.inf)

    def solve_MPC_wo(self, z_ref, z0, a_opt, delta_opt):
        """
        MPC求解（无热启动版本）
        
        每次求解都从零开始初始化，适用于开环测试
        
        参数：
            z_ref: 参考轨迹 (n_states × (horizon+1))
            z0: 当前状态 (n_states,)
            a_opt: 初始加速度假设值
            delta_opt: 初始转向角假设值
        
        返回：
            u0[0, 0]: 本周期最优加速度
            u0[0, 1]: 本周期最优转向角
            x_m: 预测的状态序列
        """
        xs = z_ref
        x0 = z0
        
        # 初始化控制序列（重复当前猜测值）
        u0 = np.array([a_opt, delta_opt] * self.horizon).reshape(-1, self.n_controls).T
        # 初始化状态序列
        x_m = np.array(x0 * (self.horizon+1)).reshape(-1, self.n_states).T

        # 构建参数向量
        c_p = np.vstack((x0, xs)).T
        
        # 初始化变量
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

        # 调用求解器求解
        res = self.solver(x0=init_control, p=c_p,
                         lbg=self.lbg, lbx=self.lbx,
                         ubg=self.ubg, ubx=self.ubx)
        
        # 调试：检查求解状态
        import os
        log_dir = ensure_debug_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "run_debug.log")
        
        def log(msg):
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        
        if 'success' in res:
            log(f"[DEBUG solve_MPC] 求解状态: {res['success']}")
        elif hasattr(res, 'stats'):
            log(f"[DEBUG solve_MPC] 求解状态: {res.stats()}")
        estimated_opt = res['x'].full()

        # 提取结果
        size_u0 = self.horizon * self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        if self.carla:
            # 返回第一个控制输入
            return u0[0, 0], u0[0, 1], x_m
        else:
            return u0, x_m

    def solve_MPC(self, z_ref, z0, n_states, u0, previous_control=None, obstacles=None, lane_bounds=None):
        """
        MPC求解（热启动版本）【推荐使用】
        
        利用上一时刻的计算结果作为初始猜测，加速求解
        
        参数：
            z_ref: 参考轨迹 (n_states × (horizon+1))
            z0: 当前状态 (n_states,)
            n_states: 预测状态序列（来自上一时刻）
            u0: 上一时刻求解的控制序列
        
        返回：
            u0[:, 0]: 预测的控制序列（完整时域）
            u0[:, 1]: 预测的转向角序列
            x_m: 预测的状态序列
            cost_time: 求解耗时（秒）
        """
        xs = z_ref
        x_m = n_states

        # 构建参数
        if not self.solver_initialized:
            self.initialize_solver()
        if previous_control is None:
            previous_control = np.array([u0[0, 0], u0[0, 1]], dtype=float)
        c_p = self._runtime_parameters(z_ref, z0, previous_control, obstacles, lane_bounds)
        # c_p = [z0,    xs[:,0], xs[:,1], ..., xs[:,10]]，形状：(n_states, horizon+2) = (6, 12)
        #前 20 个 = 控制序列 U（拉成一列），后 66 个 = 状态序列 X（拉成一列）
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

        # 计时并求解，把初始猜测、参考轨迹、约束边界全部打包扔给 IPOPT，求解器通过迭代优化，返回一组最优的控制序列和状态序列
        start_time = time.time()
        res = self.solver(
            x0=init_control,
            p=c_p,
            lbg=self.lbg,
            lbx=self.lbx,
            ubg=self.ubg,
            ubx=self.ubx,
        )
        stats = self.solver.stats()
        if (
            not bool(stats.get("success", False))
            and self.solver_fallback is not None
            and self.solver_fallback is not self.solver
        ):
            res = self.solver_fallback(
                x0=init_control,
                p=c_p,
                lbg=self.lbg,
                lbx=self.lbx,
                ubg=self.ubg,
                ubx=self.ubx,
            )
            stats = self.solver_fallback.stats()
        cost_time = time.time() - start_time

        #res['x'] 是求解器返回的原始解（CasADi 格式），.full() 把它转成 NumPy 密集数组，
        estimated_opt = res['x'].full()

        # 提取结果，从 86 维的优化向量里，切出前 20 个变回 (10, 2) 控制矩阵，后 66 个变回 (11, 6) 状态矩阵，供下一步控制使用
        size_u0 = self.horizon * self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        #返回4个东西：加速度序列、转向角序列、预测状态序列、求解时间。其中第一个控制 u0[0] 就是要在这一拍实际执行的指令，
        #u0[:, 0]第0列，u0[:, 1]第1列
        self.last_min_obstacle_clearance = self._min_obstacle_clearance(x_m, obstacles)
        if self.carla:
            return u0[:, 0], u0[:, 1], x_m, cost_time
        else:
            return u0, x_m, cost_time
