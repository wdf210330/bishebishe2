"""
test_main.py - MPC轨迹跟踪控制主测试脚本

功能说明：
    基于模型预测控制（MPC）的自动驾驶轨迹跟踪测试程序
    在CARLA仿真环境中运行，记录并可视化车辆轨迹跟踪效果

作者：[待填写]
日期：[待填写]
"""

import numpy as np
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))

from src.mcp_controller import Vehicle
from env import Env, draw_waypoints
from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame
import carla

# ============================================
# 仿真参数配置
# ============================================
simu_step = 0.05      # 仿真步长（秒），即控制周期 50ms，对应控制频率 20Hz
target_v = 40         # 目标车速（km/h）
sample_res = 2.0      # 全局路径规划的采样分辨率（米）
display_mode = "spec" # 显示模式："spec"=无渲染模式 或 "pygame"=图形界面

# ============================================
# 环境初始化
# ============================================
env = Env(display_method=display_mode, dt=simu_step)

# 清理所有之前遗留的车辆
for actor in env.world.get_actors().filter('vehicle.*'):
    actor.destroy()

spawn_points = env.map.get_spawn_points()

# ============================================
# 全局路径规划
# ============================================
start_idx, end_idx = 87, 70
# GlobalRoutePlanner初始化
grp = GlobalRoutePlanner(env.map, sample_res)

# spawn_points里面有多种信息，.location是把里面的坐标提取出来（x，y）
#返回：
#route_trace: [(carla.Waypoint, RoadOption), ...] 路径点及拓扑选项列表
route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)

# [wp for wp, _ in route]取出route里面的waypoints形成新的数组，该函数用于用于可视化参考轨迹
draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))

# 在起点位置生成新的车辆
env.reset(spawn_point=spawn_points[start_idx])

# ============================================
# 生成2个静态NPC车辆作为障碍物（在路线上）
# ============================================
# 保存障碍物位置用于绘图
obs_positions = []

# 在路线第50和第100个路径点生成障碍物
if len(route) > 50:
    obs_wp50, _ = route[50]
    spawn_loc50 = carla.Location(
        x=obs_wp50.transform.location.x,
        y=obs_wp50.transform.location.y,
        z=obs_wp50.transform.location.z + 0.5
    )
    obs_transform50 = carla.Transform(location=spawn_loc50, rotation=obs_wp50.transform.rotation)
    env.spawn_obstacle_vehicle(obs_transform50)
    # 添加 [x, y, yaw]，yaw转弧度
    yaw_deg = obs_wp50.transform.rotation.yaw
    yaw_rad = yaw_deg * np.pi / 180
    obs_positions.append([spawn_loc50.x, spawn_loc50.y, yaw_rad])

if len(route) > 130:
    obs_wp130, _ = route[130]
    spawn_loc130 = carla.Location(
        x=obs_wp130.transform.location.x,
        y=obs_wp130.transform.location.y,
        z=obs_wp130.transform.location.z + 0.5
    )
    obs_transform130 = carla.Transform(location=spawn_loc130, rotation=obs_wp130.transform.rotation)
    env.spawn_obstacle_vehicle(obs_transform130)
    # 添加 [x, y, yaw]，yaw转弧度
    yaw_deg = obs_wp130.transform.rotation.yaw
    yaw_rad = yaw_deg * np.pi / 180
    obs_positions.append([spawn_loc130.x, spawn_loc130.y, yaw_rad])

# 第三个障碍物：直道末端
if len(route) > 310:
    obs_wp310, _ = route[310]
    spawn_loc310 = carla.Location(
        x=obs_wp310.transform.location.x,
        y=obs_wp310.transform.location.y,
        z=obs_wp310.transform.location.z + 0.5
    )
    obs_transform310 = carla.Transform(location=spawn_loc310, rotation=obs_wp310.transform.rotation)
    env.spawn_obstacle_vehicle(obs_transform310)
    yaw_deg = obs_wp310.transform.rotation.yaw
    yaw_rad = yaw_deg * np.pi / 180
    obs_positions.append([spawn_loc310.x, spawn_loc310.y, yaw_rad])

obs_points = np.array(obs_positions) if obs_positions else None

# ============================================
# 构建参考路径点数组
# ============================================

route_points = np.array([[wp.transform.location.x, wp.transform.location.y]
                         for wp, _ in route])

# ============================================
# MPC控制器与Agent初始化
# ============================================
dynamic_model = Vehicle(
    actor=env.ego_vehicle,
    horizon=10,
    target_v=target_v,
    delta_t=simu_step,
    max_iter=30
)

# Xagent初始化
agent = Xagent(env, dynamic_model, dt=simu_step)
#获取起点、终点坐标
agent.set_start_end_transforms(start_idx, end_idx)
#从起点到终点计算全局路径，并将路径点加入队列，（有长度限制），上边那个用于形成绿色参考轨迹，这个用于计算
agent.plan_route(agent._start_transform, agent._end_transform)
# 设置障碍物位置
if obs_points is not None:
    agent.set_obstacles(obs_points)

# ============================================
# 仿真参数
# ============================================
max_sim_steps = 2000

# ============================================
# 数据记录变量，路径、v、acc、转向角、计算时间、横向误差
# ============================================
trajectory = []
velocities = []
accelerations = []
steerings = []
times = []
solve_times = []
lateral_errors = []


# ============================================
# 初始化显示（仅pygame模式需要），spec
# =========================================== =
#if env.display_method == "pygame":
#   env.init_display()



# ============================================
# 清空日志文件（每次仿真开始时创建新日志）
# ============================================
import os
log_dir = r"/debug_logs"
os.makedirs(log_dir, exist_ok=True)
with open(os.path.join(log_dir, "run_debug.log"), "w", encoding="utf-8") as f:
    f.write("")  # 清空日志

# ============================================
# 主仿真循环，总仿真时长100s， simu_step = 0.05， 0.05*2000=100s
# ============================================
try:
    for step in range(max_sim_steps):
        try:
            # MPC求解，加速度、转向角，下一状态，求解时间, 滤波后横向误差
            a_opt, delta_opt, next_state, solve_time_ms, lateral_error = agent.run_step()

            # 发送控制指令
            env.step([a_opt, delta_opt])

            # 提取状态信息
            x, y, yaw, vx, vy, omega = next_state[0]

            # 记录数据
            trajectory.append([x, y])
            velocities.append(vx)
            accelerations.append(a_opt)
            steerings.append(delta_opt)
            times.append(step * simu_step)
            solve_times.append(solve_time_ms)
            lateral_errors.append(lateral_error)

            # 发送控制指令
            #env.step([a_opt, delta_opt])


            # 到达终点检测
            dist_to_goal = np.linalg.norm([
                next_state[0][0] - agent._end_transform.location.x,
                next_state[0][1] - agent._end_transform.location.y
            ])

            if dist_to_goal < 1.0:
                # Destination reached
                if env.display_method == "pygame":
                    pygame.quit()
                sys.exit()
                break

            if env.display_method == "pygame":
                time.sleep(simu_step)

        except Exception as e:
            import traceback
            import os
            log_dir = r"/debug_logs"
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "error.log"), "w", encoding="utf-8") as f:
                f.write(f"Error at step {step}: {e}\n")
                f.write(traceback.format_exc())
            print(f"Error at step {step}: {e}")
            break

except KeyboardInterrupt:
    pass

# ============================================
# 数据后处理
# ============================================
trajectory = np.array(trajectory)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
steerings = np.array(steerings)
times = np.array(times)
solve_times = np.array(solve_times)
lateral_errors = np.array(lateral_errors)

# ============================================
# 结果可视化：六宫格图（3×2布局）
# ============================================
fig, axs = plt.subplots(3, 2, figsize=(16, 14))

# 左上：轨迹对比图
if len(trajectory) > 0:
    trajectory = np.array(trajectory)
    axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1],
                   label="Vehicle Path", color='darkorange', linewidth=2)
else:
    axs[0, 0].text(0.5, 0.5, "No trajectory data (simulation failed)", 
                    transform=axs[0, 0].transAxes, ha='center')
axs[0, 0].scatter(agent._start_transform.location.x,
                  agent._start_transform.location.y,
                  color='green', label="Start", zorder=5)
axs[0, 0].scatter(agent._end_transform.location.x,
                  agent._end_transform.location.y,
                  color='red', label="End", zorder=5)
axs[0, 0].plot(route_points[:, 0], route_points[:, 1],
               '--', color='blue', label="Planned Route", alpha=0.6)
# 标注障碍物位置
if obs_points is not None:
    axs[0, 0].scatter(obs_points[:, 0], obs_points[:, 1],
                      color='red', s=200, marker='X', label="Obstacles", zorder=6)
axs[0, 0].set_title("Vehicle Path and Planned Route", fontsize=14)
axs[0, 0].set_xlabel("X Position", fontsize=12)
axs[0, 0].set_ylabel("Y Position", fontsize=12)
axs[0, 0].legend(loc='upper left', fontsize=10)
axs[0, 0].grid(True)

# 右上：速度曲线
axs[0, 1].plot(times, velocities,
               label="Velocity (m/s)", color='royalblue', linewidth=2)
axs[0, 1].set_title("Velocity over Time", fontsize=14)
axs[0, 1].set_xlabel("Time (s)", fontsize=12)
axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
axs[0, 1].legend(loc='upper right', fontsize=10)
axs[0, 1].grid(True)

# 中左：加速度曲线
axs[1, 0].plot(times, accelerations,
               label="Acceleration (m/s²)", color='orange', linewidth=2)
axs[1, 0].set_title("Acceleration over Time", fontsize=14)
axs[1, 0].set_xlabel("Time (s)", fontsize=12)
axs[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=12)
axs[1, 0].legend(loc='upper right', fontsize=10)
axs[1, 0].grid(True)

# 中右：转向角曲线
axs[1, 1].plot(times, steerings,
               label="Steering Angle (rad)", color='green', linewidth=2)
axs[1, 1].set_title("Steering Angle over Time", fontsize=14)
axs[1, 1].set_xlabel("Time (s)", fontsize=12)
axs[1, 1].set_ylabel("Steering Angle (rad)", fontsize=12)
axs[1, 1].legend(loc='upper right', fontsize=10)
axs[1, 1].grid(True)

# 左下：MPC求解时间
axs[2, 0].plot(times, solve_times,
               label="Solve Time (ms)", color='purple', linewidth=1.5)
axs[2, 0].axhline(y=50, color='red', linestyle='--', label="Real-time Limit (50ms)", linewidth=2)
avg_solve_time = np.mean(solve_times) if len(solve_times) > 0 else 0
axs[2, 0].axhline(y=avg_solve_time, color='orange', linestyle=':', label=f"Avg ({avg_solve_time:.1f}ms)", linewidth=2)
axs[2, 0].set_title("MPC Solve Time over Time", fontsize=14)
axs[2, 0].set_xlabel("Time (s)", fontsize=12)
axs[2, 0].set_ylabel("Solve Time (ms)", fontsize=12)
axs[2, 0].legend(loc='upper right', fontsize=10)
axs[2, 0].grid(True)

# 右下：横向跟踪误差随时间变化曲线
axs[2, 1].plot(times, lateral_errors, 
               label="Lateral Error", color='crimson', linewidth=1.5)
axs[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)  # 零基准线
axs[2, 1].fill_between(times, lateral_errors, 0, alpha=0.3, color='crimson')  # 误差区域填充
max_lateral_error = np.max(np.abs(lateral_errors)) if len(lateral_errors) > 0 else 0
avg_lateral_error = np.mean(lateral_errors) if len(lateral_errors) > 0 else 0
std_lateral_error = np.std(lateral_errors) if len(lateral_errors) > 0 else 0
axs[2, 1].axhline(y=max_lateral_error, color='darkred', linestyle='--', alpha=0.8, label=f"Max ({max_lateral_error:.3f}m)", linewidth=2)
axs[2, 1].axhline(y=avg_lateral_error, color='orange', linestyle=':', alpha=0.8, label=f"Avg ({avg_lateral_error:.3f}m)", linewidth=2)
# 添加标准差范围
axs[2, 1].axhline(y=avg_lateral_error+std_lateral_error, color='gray', linestyle='-.', alpha=0.5, label=f"±σ ({std_lateral_error:.3f}m)")
axs[2, 1].axhline(y=avg_lateral_error-std_lateral_error, color='gray', linestyle='-.', alpha=0.5)
axs[2, 1].set_title("Lateral Tracking Error over Time", fontsize=14)
axs[2, 1].set_xlabel("Time (s)", fontsize=12)
axs[2, 1].set_ylabel("Lateral Error (m)", fontsize=12)
axs[2, 1].legend(loc='upper right', fontsize=10)
axs[2, 1].grid(True)
axs[2, 1].set_ylim([-max_lateral_error*1.5 if max_lateral_error > 0 else -0.5, 
                               max_lateral_error*1.5 if max_lateral_error > 0 else 0.5])

plt.subplots_adjust(hspace=0.45, wspace=0.3)
plt.show()
