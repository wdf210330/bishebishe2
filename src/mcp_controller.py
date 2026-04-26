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
import time
import copy
import numpy as np
import casadi as ca
from scipy.optimize import minimize

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
        self.acc_lbound = -6.0  # 最小加速度（制动）
        self.acc_ubound = 3.0   # 最大加速度（加速）
        
        # 目标速度（m/s），由km/h转换
        self.target_v = target_v / 3.6

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
        构建MPC优化问题的目标函数和等式约束（动力学约束）
        
        目标函数包含三部分：
            1. 状态跟踪代价：Σ (state_errorᵀ Q state_error)
            2. 控制代价：Σ (uᵀ R u)
            3. 控制平滑代价：Σ (Δuᵀ Rd Δu)
        
        等式约束：
            X[:, i+1] = f(X[:, i], U[:, i])  即动力学模型
        
        注意：调用此函数前必须先调用 solver_basis()
        """
        self.obj = 0  # 目标函数初始化
        self.g = []   # 等式约束列表。2项
        self.lbg = [] # 约束下界
        self.ubg = [] # 约束上界
        self.lbx = [] # 变量下界
        self.ubx = [] # 变量上界

        # 等式约束  初始状态约束：X[:, 0] = 当前状态P[:, 0]，预测的第0步状态 = 参考的第0步状态，预测必须从当前实际状态出发
        self.g.append(self.X[:, 0] - self.P[:, 0])

        # 遍历预测时域，构建目标函数和约束
        for i in range(self.horizon):
            # 状态误差 = 预测状态 - 参考状态
            state_error = (self.X[0:5, i] - self.P[0:5, i])

            # ---- 目标函数三项 ----
            # 1. 状态跟踪代价，ca.mtimes矩阵乘法
            self.obj = self.obj + ca.mtimes([state_error.T, self.Q, state_error])
            # 2. 控制代价
            self.obj = self.obj + ca.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
            # 3. 控制平滑代价（仅在相邻两步之间）control_diff，控制增量
            if i < (self.horizon-1):
                control_diff = self.U[:, i] - self.U[:, i+1]
                self.obj = self.obj + ca.mtimes([control_diff.T, self.Rd, control_diff])
            
            # ---- 动力学约束 ----
            # 预测下一步状态
            x_next_ = self.f(self.X[:, i], self.U[:, i])
            # 等式约束：预测状态 = 动力学模型计算状态
            self.g.append(self.X[:, i+1] - x_next_)

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
    
    def solver_add_soft_obs(self, obs=None, ratio=500, expn=1):
        """
        添加软障碍物避让约束（势场法增强版）
        
        包含两部分：
        1. 斥力场：距离障碍物越近惩罚越大（避免碰撞）
        2. 吸引力：引导车辆向障碍物左侧绕行（引导方向）
        
        参数：
            obs: 障碍物位置列表，每个元素 [x, y, yaw]
            ratio: 惩罚系数
            expn: 惩罚指数
        """
        import os
        log_dir = r"C:\Users\Administrator\Desktop\carla_MPC-main2\carla_MPC-main2\debug_logs"
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
        log(f"[DEBUG solver_add_soft_obs] n_obs={n_obs}, ratio={ratio}")
        
        # 车辆半径（米）
        vehicle_radius = 1.5
        # 障碍物半径（米）
        obst_radius = 1.68
        
        # ========== 经典人工势场法参数 ==========
        R_max = 15.0  # 障碍物影响范围（米）
        k_u = 10000.0   # 斥力增益系数
        
        for i in range(self.horizon):  # 每个预测步
            # 获取车辆的两个圆心
            state_i = self.X[:, i]
            (vcx1, vcy1), (vcx2, vcy2) = self.get_self_centers(state_i, vehicle_radius)
            
            for j in range(n_obs):  # 每个障碍物
                obs_x = obs[j, 0]
                obs_y = obs[j, 1]
                obs_yaw = obs[j, 2] if obs.shape[1] > 2 else 0
                
                # ========== 1. 经典斥力场（避免碰撞）==========
                (ocx1, ocy1), (ocx2, ocy2) = self.get_obs_centers_simple(obs_x, obs_y, obs_yaw, obst_radius)
                
                d1 = ca.sqrt((vcx1 - ocx1)**2 + (vcy1 - ocy1)**2)
                d2 = ca.sqrt((vcx1 - ocx2)**2 + (vcy1 - ocy2)**2)
                d3 = ca.sqrt((vcx2 - ocx1)**2 + (vcy1 - ocy1)**2)
                d4 = ca.sqrt((vcx2 - ocx2)**2 + (vcy2 - ocy2)**2)
                
                dist = ca.mmin(ca.vertcat(d1, d2, d3, d4))
                
                # 经典斥力势函数（自动具有截止特性）
                rep_term = 1/(dist + 1) - 1/(R_max + 1)
                rep_potential = 0.5 * k_u * rep_term**2
                
                # 应用到目标函数
                self.obj += rep_potential / (i + 1)
                
                # ========== 2. 吸引力（暂时禁用）==========
                # 吸引力容易把车拉向错误方向，暂时只用斥力
                # 后续可以基于参考路径方向来决定绕行方向
        
        log(f"[DEBUG solver_add_soft_obs] 势场法增强版惩罚项添加完成")

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
        if u00 is not None:
            # 第一个控制增量约束
            self.g.append(self.U[:, 0] - u00)
        
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
        if u00 is not None:
            self.lbg.append(-3)   # 加速度增量下界
            self.lbg.append(-0.3)  # 转向角增量下界（减小）
            self.ubg.append(3)    # 上界
            self.ubg.append(0.3)   # 转向角增量上界（减小）
        
        # jerk约束（控制增量）
        # 【修复振荡】减小转向角增量约束，从±0.2改为±0.1 rad
        for _ in range(self.horizon-1):
            self.lbg.append(-0.9)  # 加速度增量
            self.lbg.append(-0.1)  # 转向角增量（减小以抑制振荡）
            self.ubg.append(0.9)
            self.ubg.append(0.1)

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
        log_dir = r"C:\Users\Administrator\Desktop\carla_MPC-main2\carla_MPC-main2\debug_logs"
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

    def solve_MPC(self, z_ref, z0, n_states, u0):
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
        c_p = np.vstack((z0.T, xs)).T
        # c_p = [z0,    xs[:,0], xs[:,1], ..., xs[:,10]]，形状：(n_states, horizon+2) = (6, 12)
        #前 20 个 = 控制序列 U（拉成一列），后 66 个 = 状态序列 X（拉成一列）
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

        # 计时并求解，把初始猜测、参考轨迹、约束边界全部打包扔给 IPOPT，求解器通过迭代优化，返回一组最优的控制序列和状态序列
        start_time = time.time()
        res = self.solver(x0=init_control, p=c_p,
                         lbg=self.lbg, lbx=self.lbx,
                         ubg=self.ubg, ubx=self.ubx)
        cost_time = time.time() - start_time

        #res['x'] 是求解器返回的原始解（CasADi 格式），.full() 把它转成 NumPy 密集数组，
        estimated_opt = res['x'].full()

        # 提取结果，从 86 维的优化向量里，切出前 20 个变回 (10, 2) 控制矩阵，后 66 个变回 (11, 6) 状态矩阵，供下一步控制使用
        size_u0 = self.horizon * self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        #返回4个东西：加速度序列、转向角序列、预测状态序列、求解时间。其中第一个控制 u0[0] 就是要在这一拍实际执行的指令，
        #u0[:, 0]第0列，u0[:, 1]第1列
        if self.carla:
            return u0[:, 0], u0[:, 1], x_m, cost_time
        else:
            return u0, x_m, cost_time
