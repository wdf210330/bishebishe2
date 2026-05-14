import math
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pygame

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import Env
from src.global_route_planner import GlobalRoutePlanner
from src.mcp_controller import Vehicle
from src.project_paths import ensure_debug_log_dir
from src.scenario_utils import (
    append_route_endpoint,
    build_manual_route,
    build_result_save_path,
    calculate_lateral_error,
    calculate_trajectory_plot_limits,
    draw_route_points,
    extend_route_tail_along_lane,
    load_map_config,
    spawn_pedestrian_crossings,
    spawn_scripted_vehicle_obstacles,
    spawn_static_obstacles,
    straighten_route_tail,
    trace_route_through_locations,
    update_scripted_pedestrian_crossings,
    update_scripted_vehicle_obstacles,
)
from src.x_v2x_agent import Xagent

# ====================== 地图/场景选择开关 ======================
# 想切换测试地图，只改下面这一行。
# 可选名称来自 maps 文件夹里的 json 文件名，例如：
# "high_speed_straight_tracking"、"urban_curve_tracking"、
# "static_obstacle_avoidance"、"occluded_pedestrian_crossing"、
# "dynamic_vehicle_interaction"、"narrow_corridor_passage"、

# "double_intersection_u_turn"。不要了opportunistic_lane_change_blocked

selected_map = os.environ.get("MPC_MAP", "opportunistic_lane_change_blocked_town05")


# ====================== 全局显示/调试开关 ======================
# 绿色粗路径开关：1=在 CARLA 里绘制绿色粗路线，0=不绘制。
# 这个开关覆盖 maps/*.json 里的 draw_route_debug，所有场景都按这里决定。
draw_green_route_debug = os.environ.get("MPC_DRAW_GREEN_ROUTE", "0") == "0"


# CARLA 渲染开关：CILQR_RENDER=1 表示打开窗口渲染；其他值表示无渲染加速运行。
fast_no_rendering = os.environ.get("MPC_RENDER", "0") == "1"

map_config = load_map_config(selected_map)

simu_step = 0.1
target_v = float(map_config["target_v_kmh"])
sample_res = float(map_config["sample_res"])
display_mode = "spec"
town_name = str(map_config["carla_map"])
start_idx = int(map_config["start_idx"])
end_idx = int(map_config["end_idx"])
steer_limit_rad = 0.7
mpc_horizon = int(os.environ.get("MPC_HORIZON", "25"))
mpc_max_iter = int(os.environ.get("MPC_MAX_ITER", "30"))
debug_interval = 25
trajectory_xlim = tuple(float(value) for value in map_config["trajectory_xlim"])
trajectory_ylim = tuple(float(value) for value in map_config["trajectory_ylim"])
force_reload_world = bool(map_config["force_reload_world"])
route_draw_life_time = float(map_config["route_draw_life_time"])
draw_route_debug = draw_green_route_debug and bool(map_config["draw_route_debug"])

enable_obstacle_avoidance = bool(map_config["enable_obstacle_avoidance"])
enable_obstacle_reference_shaping = bool(map_config["enable_obstacle_reference_shaping"])
spawn_static_obstacle_demo = bool(map_config["spawn_static_obstacles"])
arrival_distance_m = float(map_config.get("arrival_distance_m", 3.0))
route_completion_distance_m = float(map_config.get("route_completion_distance_m", 0.8))
route_completion_speed_mps = float(map_config.get("route_completion_speed_mps", 0.8))
static_obstacle_route_groups = list(map_config["static_obstacle_route_groups"])
static_obstacles = list(map_config["static_obstacles"])
pedestrian_crossings = list(map_config["pedestrian_crossings"])
scripted_vehicle_obstacles = list(map_config["scripted_vehicle_obstacles"])
acc_log_upper_bound = 10.0
acc_log_lower_bound = -5.0

AGENT_DYNAMIC_PARAM_OVERRIDES = {
    "dynamic_vehicle_follow_buffer": "_dynamic_vehicle_follow_buffer",
    "dynamic_vehicle_follow_time_headway": "_dynamic_vehicle_follow_time_headway",
    "dynamic_vehicle_follow_clearance_margin": "_dynamic_vehicle_follow_clearance_margin",
    "dynamic_vehicle_follow_speed_deficit_gain": "_dynamic_vehicle_follow_speed_deficit_gain",
    "dynamic_vehicle_follow_max_slowdown": "_dynamic_vehicle_follow_max_slowdown",
    "dynamic_follow_gate_clearance": "_dynamic_follow_gate_clearance",
    "dynamic_lane_change_front_gap": "_dynamic_lane_change_front_gap",
    "dynamic_lane_change_rear_gap": "_dynamic_lane_change_rear_gap",
    "dynamic_lane_change_target_lane_tolerance": "_dynamic_lane_change_target_lane_tolerance",
    "dynamic_overtake_start_distance": "_dynamic_overtake_start_distance",
    "dynamic_overtake_full_distance": "_dynamic_overtake_full_distance",
    "dynamic_overtake_return_front_clearance_lengths": "_dynamic_overtake_return_front_clearance_lengths",
    "dynamic_overtake_return_distance": "_dynamic_overtake_return_distance",
}

MODEL_DYNAMIC_PARAM_OVERRIDES = {
    "dynamic_obstacle_safe_margin": "dynamic_obstacle_safe_margin",
    "dynamic_obstacle_influence_margin": "dynamic_obstacle_influence_margin",
    "dynamic_obstacle_distance_weight_scale": "dynamic_obstacle_distance_weight_scale",
    "dynamic_obstacle_violation_weight_scale": "dynamic_obstacle_violation_weight_scale",
}


def main():
    print("=" * 50)
    print("MPC Trajectory Tracking - CARLA Simulation")
    print("=" * 50)
    print(f"Selected map config: {selected_map} ({map_config['display_name']})")

    log_dir = ensure_debug_log_dir()
    with open(log_dir / "run_debug.log", "w", encoding="utf-8") as f:
        f.write("")
    env = None
    fig = None
    try:
        print("\n[1/5] Initializing CARLA environment...")
        env = Env(
            display_method=display_mode,
            dt=simu_step,
            max_steer_rad=steer_limit_rad,
            no_rendering=fast_no_rendering,
            town_name=town_name,
            force_reload_world=force_reload_world,
        )
        env.clean()
        spawn_points = env.map.get_spawn_points()
        loaded_map_name = env.map.name.split("/")[-1]
        print(f"    Map: {env.map.name}")
        print(f"    Requested map: {town_name}")
        print(f"    Spawn points: {len(spawn_points)}")
        print(f"    CARLA no rendering: {fast_no_rendering}")
        print(f"    Draw green route debug: {draw_route_debug}")
        print(f"    Force reload world: {force_reload_world}")

        if start_idx >= len(spawn_points) or end_idx >= len(spawn_points):
            raise IndexError(
                f"Invalid start/end index: start={start_idx}, end={end_idx}, "
                f"available={len(spawn_points)} on map {loaded_map_name}"
            )

        print(f"\n[2/5] Spawning ego vehicle (start: {start_idx}, end: {end_idx})...")
        env.reset(spawn_point=spawn_points[start_idx])

        print("\n[3/5] Initializing MPC controller...")
        print(f"    Horizon: {mpc_horizon} steps")
        print(f"    Max MPC iterations: {mpc_max_iter}")
        print(f"    Target speed: {target_v} km/h")
        print(f"    Time step: {simu_step} s")

        dynamic_model = Vehicle(
            actor=env.ego_vehicle,
            horizon=mpc_horizon,
            target_v=target_v,
            delta_t=simu_step,
            max_iter=mpc_max_iter,
        )
        applied_model_dynamic_overrides = {}
        for config_key, model_attr in MODEL_DYNAMIC_PARAM_OVERRIDES.items():
            override_value = map_config.get(config_key)
            if override_value is None:
                continue
            override_value = float(override_value)
            setattr(dynamic_model, model_attr, override_value)
            applied_model_dynamic_overrides[config_key] = override_value
        if applied_model_dynamic_overrides:
            print("    Dynamic obstacle model overrides:")
            for config_key, override_value in applied_model_dynamic_overrides.items():
                print(f"      {config_key} = {override_value}")

        initial_state, _ = dynamic_model.get_state_carla()
        init_x, init_y, init_yaw, init_speed = initial_state[:4]
        start_location = env.ego_vehicle.get_location()

        print("\n[4/5] Planning route from actual ego position...")
        grp = GlobalRoutePlanner(env.map, sample_res)
        manual_route_points = map_config.get("manual_route_points")
        route_via_indices = []
        if manual_route_points:
            route = build_manual_route(
                manual_route_points,
                repeat_count=int(map_config.get("route_repeat_count", 1)),
            )
            print("    Manual route points enabled.")
        else:
            route_via_indices = [int(value) for value in map_config.get("route_via_indices", [])]
            for route_via_index in route_via_indices:
                if route_via_index < 0 or route_via_index >= len(spawn_points):
                    raise IndexError(
                        f"Invalid route_via_indices entry: {route_via_index}, "
                        f"available={len(spawn_points)} on map {loaded_map_name}"
                    )
            route_locations = [start_location]
            route_locations.extend(spawn_points[index].location for index in route_via_indices)
            route_locations.append(spawn_points[end_idx].location)
            route = trace_route_through_locations(grp, route_locations)
            route = append_route_endpoint(
                route,
                spawn_points[end_idx].location,
                end_yaw=spawn_points[end_idx].rotation.yaw,
                lane_width=getattr(route[-1][0], "lane_width", 3.5),
            )

        if len(route) == 0:
            raise RuntimeError("Global route planner returned an empty route.")
        if route_via_indices:
            print(f"    Route via spawn indices: {route_via_indices}")

        require_route_completion = bool(map_config.get("require_route_completion", False))
        require_route_completion = require_route_completion or (
            (bool(route_via_indices) or bool(manual_route_points)) and start_idx == end_idx
        )
        route_end_index = map_config.get("route_end_index")
        if route_end_index is not None:
            route_end_index = int(route_end_index)
            if route_end_index < 1 or route_end_index >= len(route):
                raise IndexError(
                    f"route_end_index={route_end_index} is outside current route, "
                    f"route length={len(route)}"
                )
            route = route[: route_end_index + 1]
        extend_tail_from_index = map_config.get("extend_tail_from_index")
        if extend_tail_from_index is not None:
            route = extend_route_tail_along_lane(
                route,
                from_index=int(extend_tail_from_index),
                num_points=int(map_config.get("extend_tail_num_points", 20)),
                step_m=float(map_config.get("extend_tail_step_m", sample_res)),
                heading_deg=map_config.get("extend_tail_heading_deg"),
                yaw_tolerance_deg=float(map_config.get("extend_tail_yaw_tolerance_deg", 25.0)),
            )
        straighten_tail_from_index = map_config.get("straighten_tail_from_index")
        if straighten_tail_from_index is not None:
            route = straighten_route_tail(
                route,
                from_index=int(straighten_tail_from_index),
                num_points=int(map_config.get("straighten_tail_num_points", 20)),
                step_m=float(map_config.get("straighten_tail_step_m", sample_res)),
                heading_deg=map_config.get("straighten_tail_heading_deg"),
            )

        agent = Xagent(env, dynamic_model, dt=simu_step)
        applied_agent_dynamic_overrides = {}
        for config_key, agent_attr in AGENT_DYNAMIC_PARAM_OVERRIDES.items():
            override_value = map_config.get(config_key)
            if override_value is None:
                continue
            override_value = float(override_value)
            setattr(agent, agent_attr, override_value)
            applied_agent_dynamic_overrides[config_key] = override_value
        if applied_agent_dynamic_overrides:
            print("    Dynamic lane-change tuning overrides:")
            for config_key, override_value in applied_agent_dynamic_overrides.items():
                print(f"      {config_key} = {override_value}")
        agent.enable_obstacle_avoidance = enable_obstacle_avoidance
        agent.enable_obstacle_reference_shaping = enable_obstacle_reference_shaping
        agent.set_start_end_transforms(start_idx, end_idx)
        if manual_route_points or route_end_index is not None:
            agent._end_transform = route[-1][0].transform
        if require_route_completion:
            agent._terminal_reference_speed = 0.0
            agent._route_completion_distance = route_completion_distance_m
            agent._route_completion_speed = route_completion_speed_mps
            agent._preserve_terminal_waypoint = True
        agent.set_route(route)

        route_points = (
            np.array(agent._route_cache["path_points"], dtype=float)
            if agent._route_cache is not None
            else np.array([[wp.transform.location.x, wp.transform.location.y] for wp, _ in route], dtype=float)
        )

        if draw_route_debug:
            draw_route_points(
                env.world,
                route_points,
                z=0.5,
                color=(0, 255, 0),
                life_time=route_draw_life_time,
                draw_lines=False,
                point_size=0.14,
                line_thickness=0.18,
            )

        obstacle_markers = []
        if spawn_static_obstacle_demo:
            obstacle_markers = spawn_static_obstacles(env, route, static_obstacle_route_groups, static_obstacles)
        elif static_obstacle_route_groups or static_obstacles:
            raise ValueError("Static obstacles configured but spawn_static_obstacles=false")

        pedestrian_markers, scripted_pedestrian_entries = spawn_pedestrian_crossings(env, route, pedestrian_crossings)
        scripted_vehicle_entries = spawn_scripted_vehicle_obstacles(env, route, scripted_vehicle_obstacles)
        update_scripted_pedestrian_crossings(env, scripted_pedestrian_entries, 0.0)
        update_scripted_vehicle_obstacles(env, scripted_vehicle_entries, 0.0)

        first_wp = route[0][0].transform.location
        if len(route) > 1:
            second_wp = route[1][0].transform.location
            route_yaw = np.degrees(np.arctan2(second_wp.y - first_wp.y, second_wp.x - first_wp.x))
        else:
            route_yaw = route[0][0].transform.rotation.yaw
        heading_error = (route_yaw - init_yaw + 180.0) % 360.0 - 180.0
        print(
            "\n[debug] Spawn consistency check:"
            f"\n    actual_vehicle=({init_x:.2f}, {init_y:.2f}, yaw={init_yaw:.2f}, v={init_speed:.2f})"
            f"\n    route_first=({first_wp.x:.2f}, {first_wp.y:.2f})"
            f"\n    route_yaw={route_yaw:.2f}, heading_error={heading_error:.2f} deg"
        )

        if env.display_method == "pygame":
            env.init_display()

        print("\n[5/5] Starting simulation...")
        if "MPC_MAX_STEPS" in os.environ:
            max_sim_steps = int(os.environ["MPC_MAX_STEPS"])
        else:
            max_sim_time_s = float(map_config.get("max_sim_time_s", 300.0))
            max_sim_steps = int(math.ceil(max_sim_time_s / simu_step))

        trajectory = [[init_x, init_y]]
        yaws = [init_yaw]
        velocities = [init_speed]
        accelerations = [0.0]
        jerks = [0.0]
        steerings = [0.0]
        steering_rates = [0.0]
        reference_speeds = [float(getattr(agent, "_last_reference_speed", target_v / 3.6))]
        times = [0.0]
        solve_times = [0.0]
        lateral_errors = [calculate_lateral_error([init_x, init_y], route_points)]
        obstacle_clearances = [float("inf")]
        compute_times = []

        for step in range(max_sim_steps):
            loop_start = time.perf_counter()
            sim_time = step * simu_step
            update_scripted_vehicle_obstacles(env, scripted_vehicle_entries, sim_time)
            update_scripted_pedestrian_crossings(env, scripted_pedestrian_entries, sim_time)

            try:
                a_opt, delta_opt, next_state, solve_time_ms, _ = agent.run_step()
            except StopIteration as e:
                print(f"\n>>> Simulation finished: {e} <<<")
                break

            applied_control = env.step([a_opt, delta_opt])
            actual_state, _ = dynamic_model.get_state_carla()
            x, y, yaw, speed = actual_state[:4]
            dist_to_end = np.linalg.norm([x - agent._end_transform.location.x, y - agent._end_transform.location.y])
            remaining_route_s = float("inf")
            if agent._route_cache is not None:
                remaining_route_s = max(
                    float(agent._route_cache["arc_lengths"][-1]) - float(agent._route_progress_s),
                    0.0,
                )

            if not require_route_completion and dist_to_end < arrival_distance_m:
                print("\n>>> Destination reached! <<<")
                break

            if (
                require_route_completion
                and remaining_route_s <= max(route_completion_distance_m, 1.0)
                and dist_to_end <= route_completion_distance_m
                and speed <= route_completion_speed_mps
            ):
                print("\n>>> Destination reached! <<<")
                break

            trajectory.append([x, y])
            yaws.append(yaw)
            speed_diff_acc = (speed - velocities[-1]) / simu_step
            actual_acc = env.get_ego_longitudinal_acceleration()
            if not np.isfinite(actual_acc):
                actual_acc = speed_diff_acc
            elif (
                (actual_acc > acc_log_upper_bound or actual_acc < acc_log_lower_bound)
                and min(speed, velocities[-1]) < 0.3
                and max(speed, velocities[-1]) < 2.0
            ):
                actual_acc = float(np.clip(speed_diff_acc, acc_log_lower_bound, acc_log_upper_bound))
            elif (
                (actual_acc > acc_log_upper_bound or actual_acc < acc_log_lower_bound)
                and max(speed, velocities[-1]) < 0.8
                and acc_log_lower_bound <= speed_diff_acc <= acc_log_upper_bound
                and (
                    dist_to_end < max(arrival_distance_m + 2.0, 5.0)
                    or len(agent._waypoints_queue) <= 3
                    or min(speed, velocities[-1]) < 0.2
                )
            ):
                actual_acc = float(np.clip(speed_diff_acc, acc_log_lower_bound, acc_log_upper_bound))
            actual_acc = float(np.clip(actual_acc, acc_log_lower_bound, acc_log_upper_bound))
            actual_jerk = (actual_acc - accelerations[-1]) / simu_step
            steering_rate = (delta_opt - steerings[-1]) / simu_step
            velocities.append(speed)
            accelerations.append(actual_acc)
            jerks.append(actual_jerk)
            steerings.append(delta_opt)
            steering_rates.append(steering_rate)
            reference_speeds.append(float(getattr(agent, "_last_reference_speed", target_v / 3.6)))
            times.append((step + 1) * simu_step)
            solve_times.append(solve_time_ms)
            lateral_errors.append(calculate_lateral_error([x, y], route_points))
            obstacle_clearances.append(float(getattr(agent, "_last_min_obstacle_clearance", float("inf"))))
            compute_times.append((time.perf_counter() - loop_start) * 1000.0)

            print(
                f"    Trace {step}: t={(step + 1) * simu_step:.2f}s, "
                f"acc={actual_acc:.3f} m/s^2, jerk={actual_jerk:.3f} m/s^3"
            )

            if step % debug_interval == 0:
                print(
                    f"    Step {step}: pos=({x:.1f}, {y:.1f}), "
                    f"speed={speed*3.6:.1f} km/h, ref={reference_speeds[-1]*3.6:.1f} km/h, "
                    f"dist={dist_to_end:.1f}m, throttle={applied_control.throttle:.2f}, brake={applied_control.brake:.2f}, "
                    f"steer_rate={steering_rate:.3f} rad/s"
                )

            if env.display_method == "pygame":
                env.hud.tick(env, env.clock)
                env.hud.render(env.display)
                pygame.display.flip()
                if env.check_quit():
                    break

    except KeyboardInterrupt:
        print("\nSimulation interrupted")

    print("\nPlotting results...")
    trajectory = np.asarray(trajectory, dtype=float)
    yaws = np.asarray(yaws, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    accelerations = np.asarray(accelerations, dtype=float)
    jerks = np.asarray(jerks, dtype=float)
    steerings = np.asarray(steerings, dtype=float)
    steering_rates = np.asarray(steering_rates, dtype=float)
    reference_speeds = np.asarray(reference_speeds, dtype=float)
    times = np.asarray(times, dtype=float)
    solve_times = np.asarray(solve_times, dtype=float)
    lateral_errors = np.asarray(lateral_errors, dtype=float)
    obstacle_clearances = np.asarray(obstacle_clearances, dtype=float)

    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    axs[0, 0].plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label="Vehicle Path",
        color="darkorange",
        linewidth=2.2,
        zorder=4,
    )
    axs[0, 0].scatter(trajectory[0, 0], trajectory[0, 1], color="green", label="Start", s=100, zorder=5)
    axs[0, 0].scatter(agent._end_transform.location.x, agent._end_transform.location.y, color="red", label="End", s=100, zorder=5)
    axs[0, 0].plot(
        route_points[:, 0],
        route_points[:, 1],
        color="royalblue",
        linestyle="--",
        linewidth=1.8,
        label="Planned Route",
        alpha=0.9,
        zorder=3,
    )

    if obstacle_markers:
        obstacle_markers_arr = np.asarray(obstacle_markers, dtype=float)
        axs[0, 0].scatter(obstacle_markers_arr[:, 0], obstacle_markers_arr[:, 1], marker="X", color="red", label="Static Obstacles", s=150, zorder=6)
    for marker_index, (start_marker, end_marker) in enumerate(pedestrian_markers):
        label = "Pedestrian Crossing" if marker_index == 0 else None
        axs[0, 0].plot([start_marker[0], end_marker[0]], [start_marker[1], end_marker[1]], color="magenta", linestyle=":", linewidth=2, label=label, zorder=5)
    path_label_used = False
    for entry in scripted_vehicle_entries:
        path_points = np.asarray(entry.get("path_points", []), dtype=float)
        if path_points.ndim == 2 and path_points.shape[0] >= 2:
            label = "Dynamic Obstacle Path" if not path_label_used else None
            axs[0, 0].plot(path_points[:, 0], path_points[:, 1], color="red", linestyle="--", linewidth=2, label=label, zorder=5)
            path_label_used = True

    axs[0, 0].set_title("Vehicle Path and Planned Route", fontsize=14)
    axs[0, 0].set_xlabel("X Position", fontsize=12)
    axs[0, 0].set_ylabel("Y Position", fontsize=12)
    axs[0, 0].legend(loc="upper left", fontsize=10)
    axs[0, 0].grid(True, alpha=0.5)
    plot_xlim, plot_ylim = calculate_trajectory_plot_limits(
        route_points,
        trajectory,
        agent._end_transform.location,
        obstacle_markers,
        pedestrian_markers,
        scripted_vehicle_entries,
        default_xlim=trajectory_xlim,
        default_ylim=trajectory_ylim,
    )
    axs[0, 0].set_xlim(plot_xlim)
    axs[0, 0].set_ylim(plot_ylim)

    axs[0, 1].plot(times, velocities, label="Velocity (m/s)", color="royalblue", linewidth=2)
    if len(reference_speeds) == len(times):
        axs[0, 1].plot(times, reference_speeds, label="Local reference (m/s)", color="black", linestyle="-.", linewidth=1.6)
    axs[0, 1].axhline(y=target_v / 3.6, color="r", linestyle="--", label=f"Global target ({target_v/3.6:.1f} m/s)")
    axs[0, 1].set_title("Velocity over Time", fontsize=14)
    axs[0, 1].set_xlabel("Time (s)", fontsize=12)
    axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
    axs[0, 1].legend(loc="upper right", fontsize=10)
    axs[0, 1].grid(True, alpha=0.5)

    axs[1, 0].plot(times, accelerations, label="Acceleration (m/s^2)", color="orange", linewidth=2)
    axs[1, 0].set_title("Acceleration over Time", fontsize=14)
    axs[1, 0].set_xlabel("Time (s)", fontsize=12)
    axs[1, 0].set_ylabel("Acceleration (m/s^2)", fontsize=12)
    axs[1, 0].legend(loc="upper right", fontsize=10)
    axs[1, 0].grid(True, alpha=0.5)

    axs[1, 1].plot(times, steerings, label="Steering Angle (rad)", color="green", linewidth=2)
    axs[1, 1].set_title("Steering Angle over Time", fontsize=14)
    axs[1, 1].set_xlabel("Time (s)", fontsize=12)
    axs[1, 1].set_ylabel("Steering Angle (rad)", fontsize=12)
    axs[1, 1].legend(loc="upper right", fontsize=10)
    axs[1, 1].grid(True, alpha=0.5)

    axs[2, 0].plot(times, solve_times, label="Solve Time (ms)", color="purple", linewidth=1.5)
    axs[2, 0].axhline(y=simu_step * 1000.0, color="red", linestyle="--", label=f"Real-time Limit ({simu_step * 1000:.0f}ms)", linewidth=2)
    avg_solve_time = np.mean(solve_times[1:]) if len(solve_times) > 1 else 0.0
    axs[2, 0].axhline(y=avg_solve_time, color="orange", linestyle=":", label=f"Avg ({avg_solve_time:.1f}ms)", linewidth=2)
    axs[2, 0].set_title("MPC Solve Time over Time", fontsize=14)
    axs[2, 0].set_xlabel("Time (s)", fontsize=12)
    axs[2, 0].set_ylabel("Solve Time (ms)", fontsize=12)
    axs[2, 0].legend(loc="upper right", fontsize=10)
    axs[2, 0].grid(True, alpha=0.5)

    axs[2, 1].plot(times, lateral_errors, label="Lateral Error (m)", color="crimson", linewidth=1.5)
    axs[2, 1].fill_between(times, lateral_errors, 0, alpha=0.3, color="crimson")
    max_lateral_error = np.max(lateral_errors) if len(lateral_errors) > 0 else 0.0
    avg_lateral_error = np.mean(lateral_errors) if len(lateral_errors) > 0 else 0.0
    max_abs_acceleration = np.max(np.abs(accelerations[1:])) if len(accelerations) > 1 else 0.0
    avg_abs_jerk = np.mean(np.abs(jerks[1:])) if len(jerks) > 1 else 0.0
    avg_abs_steering_rate = np.mean(np.abs(steering_rates[1:])) if len(steering_rates) > 1 else 0.0
    axs[2, 1].axhline(y=max_lateral_error, color="darkred", linestyle="--", alpha=0.8, label=f"Max ({max_lateral_error:.3f}m)", linewidth=2)
    axs[2, 1].axhline(y=avg_lateral_error, color="orange", linestyle=":", alpha=0.8, label=f"Avg ({avg_lateral_error:.3f}m)", linewidth=2)
    axs[2, 1].set_title("Lateral Tracking Error over Time", fontsize=14)
    axs[2, 1].set_xlabel("Time (s)", fontsize=12)
    axs[2, 1].set_ylabel("Lateral Error (m)", fontsize=12)
    axs[2, 1].legend(loc="upper right", fontsize=10)
    axs[2, 1].grid(True, alpha=0.5)

    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    save_path = build_result_save_path(selected_map)
    csv_save_path = pathlib.Path(save_path).with_suffix(".csv")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Result saved to: {save_path}")

    with open(csv_save_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        csv_file.write("场景,step,time,x,y,yaw,speed,ac,steer,solve_time,lat_error,min_obstacle_dist,\n")
        for step_index in range(len(times)):
            min_obstacle_dist = (
                f"{obstacle_clearances[step_index]:.6f}"
                if np.isfinite(obstacle_clearances[step_index])
                else ""
            )
            csv_file.write(
                f"{selected_map},"
                f"{step_index},"
                f"{times[step_index]:.6f},"
                f"{trajectory[step_index, 0]:.6f},"
                f"{trajectory[step_index, 1]:.6f},"
                f"{yaws[step_index]:.6f},"
                f"{velocities[step_index]:.6f},"
                f"{accelerations[step_index]:.6f},"
                f"{steerings[step_index]:.6f},"
                f"{solve_times[step_index]:.6f},"
                f"{lateral_errors[step_index]:.6f},"
                f"{min_obstacle_dist},\n"
            )
    print(f"Trace CSV saved to: {csv_save_path}")

    print("\n" + "=" * 50)
    print("Simulation Statistics:")
    print(f"  Total steps: {len(times)}")
    print(f"  Simulation time: {times[-1]:.2f} s")
    print(f"  Average compute time: {np.mean(compute_times) if compute_times else 0.0:.2f} ms")
    print(f"  Average MPC solve time: {avg_solve_time:.2f} ms")
    print(f"  Max lateral error: {max_lateral_error:.3f} m")
    print(f"  Average lateral error: {avg_lateral_error:.3f} m")
    print(f"  Max absolute acceleration: {max_abs_acceleration:.3f} m/s^2")
    print(f"  Average absolute jerk: {avg_abs_jerk:.3f} m/s^3")
    print(f"  Average absolute steering angle change rate: {avg_abs_steering_rate:.3f} rad/s")
    finite_clearances = obstacle_clearances[np.isfinite(obstacle_clearances)]
    if len(finite_clearances) > 0 and (obstacle_markers or scripted_vehicle_entries):
        print(f"  Min predicted obstacle clearance: {np.min(finite_clearances):.3f} m")
    print(f"  Average speed: {np.mean(velocities):.2f} m/s")
    print(f"  Max speed: {np.max(velocities):.2f} m/s")
    print("=" * 50)

    if env.display_method == "pygame":
        pygame.quit()
    if os.environ.get("MPC_SHOW_PLOT", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
    print("\nDone!")
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass
    if env is not None:
        try:
            env.clean()
        except Exception:
            pass


if __name__ == "__main__":
    main()
