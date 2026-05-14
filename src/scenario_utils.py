"""MPC scenario utilities copied from the shared CILQR scenario harness."""

import json
import os
import re
import time
import math

import carla
import matplotlib.pyplot as plt
import numpy as np
import pygame

from src.project_paths import PROJECT_ROOT

def load_map_config(map_name):
    config_path = PROJECT_ROOT / "maps" / f"{map_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Map config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_keys = [
        "display_name",
        "carla_map",
        "start_idx",
        "end_idx",
        "target_v_kmh",
        "sample_res",
        "trajectory_xlim",
        "trajectory_ylim",
        "force_reload_world",
        "route_draw_life_time",
        "draw_route_debug",
        "enable_obstacle_avoidance",
        "enable_obstacle_reference_shaping",
        "spawn_static_obstacles",
        "static_obstacle_route_groups",
        "static_obstacles",
        "pedestrian_crossings",
        "scripted_vehicle_obstacles",
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Map config missing keys: {missing_keys}, file: {config_path}")

    return config

def build_result_save_path(map_name):
    result_dir = os.path.join(PROJECT_ROOT, "result")
    os.makedirs(result_dir, exist_ok=True)

    result_name_pattern = re.compile(rf"^result_{re.escape(map_name)}_(\d+)\.png$")
    max_index = 0
    for filename in os.listdir(result_dir):
        match = result_name_pattern.match(filename)
        if match:
            max_index = max(max_index, int(match.group(1)))

    next_index = max_index + 1
    return os.path.join(result_dir, f"result_{map_name}_{next_index}.png")


def calculate_lateral_error(point, path_points):
    """Return distance from point to the nearest route segment."""
    path = np.asarray(path_points, dtype=float)
    if len(path) < 2:
        return 0.0

    point = np.asarray(point, dtype=float)
    starts = path[:-1]
    ends = path[1:]
    segment_vectors = ends - starts
    segment_lengths_sq = np.sum(segment_vectors * segment_vectors, axis=1)
    segment_lengths_sq = np.maximum(segment_lengths_sq, 1e-9)

    point_vectors = point - starts
    ratios = np.sum(point_vectors * segment_vectors, axis=1) / segment_lengths_sq
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * segment_vectors
    distances = np.linalg.norm(point - projections, axis=1)
    return float(np.min(distances))


def calculate_trajectory_plot_limits(
    route_points,
    trajectory,
    end_location,
    obstacle_markers,
    pedestrian_markers,
    scripted_vehicle_entries,
    default_xlim=None,
    default_ylim=None,
):
    """Return plot limits covering the current scenario content."""
    point_sets = []

    def add_points(points):
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            if arr.shape[0] >= 2:
                arr = arr[:2].reshape(1, 2)
            else:
                return
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
        finite_mask = np.isfinite(arr).all(axis=1)
        if np.any(finite_mask):
            point_sets.append(arr[finite_mask])

    add_points(route_points)
    add_points(trajectory)
    add_points([end_location.x, end_location.y])
    add_points(obstacle_markers)

    for start_marker, end_marker in pedestrian_markers:
        add_points([start_marker[0], start_marker[1]])
        add_points([end_marker[0], end_marker[1]])

    for entry in scripted_vehicle_entries:
        path_points = entry.get("path_points", [])
        if path_points:
            add_points(path_points)

    if not point_sets:
        return default_xlim, default_ylim

    all_points = np.vstack(point_sets)
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    span = max_xy - min_xy
    padding = np.maximum(span * 0.08, 5.0)

    xlim = (float(min_xy[0] - padding[0]), float(max_xy[0] + padding[0]))
    ylim = (float(min_xy[1] - padding[1]), float(max_xy[1] + padding[1]))
    return xlim, ylim


def trace_route_through_locations(route_planner, locations):
    """Trace one continuous route through ordered locations."""
    if len(locations) < 2:
        raise ValueError("Route planning needs at least start and end locations.")

    route = []
    for segment_index in range(len(locations) - 1):
        segment = route_planner.trace_route(
            locations[segment_index],
            locations[segment_index + 1],
        )
        if len(segment) == 0:
            raise RuntimeError(f"Global route planner returned an empty route segment: {segment_index}")

        if route:
            route.extend(segment[1:])
        else:
            route.extend(segment)

    return route


def draw_route_points(
    world,
    points,
    z=0.5,
    color=(0, 255, 0),
    life_time=0.5,
    draw_lines=True,
    persistent_lines=True,
    point_size=0.1,
    line_thickness=0.08,
):
    debug_color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    last_point = None
    for x, y in np.asarray(points, dtype=float):
        begin = carla.Location(x=float(x), y=float(y), z=float(z))
        world.debug.draw_point(
            begin,
            size=point_size,
            color=debug_color,
            life_time=life_time,
            persistent_lines=persistent_lines,
        )
        if draw_lines and last_point is not None:
            world.debug.draw_line(
                last_point,
                begin,
                thickness=line_thickness,
                color=debug_color,
                life_time=life_time,
                persistent_lines=persistent_lines,
            )
        last_point = begin


class ManualRouteWaypoint:
    def __init__(
        self,
        x,
        y,
        yaw,
        lane_width=3.5,
        left_lane=None,
        right_lane=None,
        left_lane_marking=None,
        right_lane_marking=None,
    ):
        self.is_manual_route = True
        self.transform = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=0.6),
            carla.Rotation(yaw=float(yaw)),
        )
        self.lane_width = float(lane_width)
        self._left_lane = left_lane
        self._right_lane = right_lane
        self.left_lane_marking = left_lane_marking
        self.right_lane_marking = right_lane_marking

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane


def append_route_endpoint(route, end_location, end_yaw=None, lane_width=3.5, min_distance=0.2):
    """Append an explicit terminal point when the traced route stops short of the desired end."""
    if not route:
        return route

    end_x = float(end_location.x)
    end_y = float(end_location.y)
    final_wp = route[-1][0]
    final_loc = final_wp.transform.location
    dx = end_x - float(final_loc.x)
    dy = end_y - float(final_loc.y)
    if math.hypot(dx, dy) <= float(min_distance):
        return route

    yaw = float(end_yaw) if end_yaw is not None else math.degrees(math.atan2(dy, dx))
    route.append((ManualRouteWaypoint(end_x, end_y, yaw, lane_width=lane_width), None))
    return route


def build_manual_route(points, repeat_count=1):
    if len(points) < 2:
        raise ValueError("manual_route_points needs at least two points.")

    base_points = [(float(point[0]), float(point[1])) for point in points]
    repeat_count = max(1, int(repeat_count))
    route_points = []
    for repeat_index in range(repeat_count):
        if repeat_index == 0:
            route_points.extend(base_points)
        else:
            route_points.extend(base_points[1:])

    route = []
    for index, (x, y) in enumerate(route_points):
        if index + 1 < len(route_points):
            next_x, next_y = route_points[index + 1]
            yaw = math.degrees(math.atan2(next_y - y, next_x - x))
        elif index > 0:
            prev_x, prev_y = route_points[index - 1]
            yaw = math.degrees(math.atan2(y - prev_y, x - prev_x))
        else:
            yaw = 0.0
        route.append((ManualRouteWaypoint(x, y, yaw), None))
    return route


def straighten_route_tail(route, from_index, num_points, step_m, heading_deg=None):
    """Replace the route tail with a straight extension starting at from_index."""
    if len(route) < 2:
        raise ValueError("route must contain at least two points to straighten the tail.")

    from_index = int(from_index)
    if from_index < 1 or from_index >= len(route):
        raise IndexError(f"straighten tail index {from_index} is outside route length {len(route)}")

    num_points = int(num_points)
    if num_points < 1:
        raise ValueError("straighten tail num_points must be >= 1")

    step_m = float(step_m)
    if step_m <= 0.0:
        raise ValueError("straighten tail step_m must be > 0")

    anchor_wp = route[from_index][0]
    anchor_loc = anchor_wp.transform.location
    prev_loc = route[from_index - 1][0].transform.location

    if heading_deg is None:
        heading_deg = math.degrees(
            math.atan2(
                float(anchor_loc.y) - float(prev_loc.y),
                float(anchor_loc.x) - float(prev_loc.x),
            )
        )
    else:
        heading_deg = float(heading_deg)

    lane_width = float(getattr(anchor_wp, "lane_width", 3.5))
    left_lane = anchor_wp.get_left_lane() if hasattr(anchor_wp, "get_left_lane") else None
    right_lane = anchor_wp.get_right_lane() if hasattr(anchor_wp, "get_right_lane") else None
    left_lane_marking = getattr(anchor_wp, "left_lane_marking", None)
    right_lane_marking = getattr(anchor_wp, "right_lane_marking", None)
    heading_rad = math.radians(heading_deg)
    dx = step_m * math.cos(heading_rad)
    dy = step_m * math.sin(heading_rad)

    straightened_route = list(route[: from_index + 1])
    base_x = float(anchor_loc.x)
    base_y = float(anchor_loc.y)
    for point_index in range(1, num_points + 1):
        straightened_route.append(
            (
                ManualRouteWaypoint(
                    base_x + dx * point_index,
                    base_y + dy * point_index,
                    heading_deg,
                    lane_width=lane_width,
                    left_lane=left_lane,
                    right_lane=right_lane,
                    left_lane_marking=left_lane_marking,
                    right_lane_marking=right_lane_marking,
                ),
                None,
            )
        )
    return straightened_route


def extend_route_tail_along_lane(route, from_index, num_points, step_m, heading_deg=None, yaw_tolerance_deg=25.0):
    """Extend the route tail using real CARLA lane successors that best preserve heading."""
    if len(route) < 2:
        raise ValueError("route must contain at least two points to extend the tail.")

    from_index = int(from_index)
    if from_index < 1 or from_index >= len(route):
        raise IndexError(f"extend tail index {from_index} is outside route length {len(route)}")

    num_points = int(num_points)
    if num_points < 1:
        raise ValueError("extend tail num_points must be >= 1")

    step_m = float(step_m)
    if step_m <= 0.0:
        raise ValueError("extend tail step_m must be > 0")

    anchor_wp = route[from_index][0]
    prev_wp = route[from_index - 1][0]
    anchor_loc = anchor_wp.transform.location
    prev_loc = prev_wp.transform.location

    if heading_deg is None:
        target_heading = math.degrees(
            math.atan2(
                float(anchor_loc.y) - float(prev_loc.y),
                float(anchor_loc.x) - float(prev_loc.x),
            )
        )
    else:
        target_heading = float(heading_deg)

    if bool(getattr(anchor_wp, "is_manual_route", False)) or not hasattr(anchor_wp, "next"):
        return straighten_route_tail(route, from_index, num_points, step_m, heading_deg=target_heading)

    def heading_error_deg(candidate_yaw, reference_yaw):
        return abs(((float(candidate_yaw) - float(reference_yaw) + 180.0) % 360.0) - 180.0)

    extended_route = list(route[: from_index + 1])
    current_wp = anchor_wp
    current_heading = target_heading
    yaw_tolerance_deg = float(yaw_tolerance_deg)

    for _ in range(num_points):
        next_candidates = current_wp.next(step_m)
        if not next_candidates:
            break

        best_wp = min(
            next_candidates,
            key=lambda candidate: heading_error_deg(candidate.transform.rotation.yaw, current_heading),
        )
        best_error = heading_error_deg(best_wp.transform.rotation.yaw, current_heading)
        if best_error > yaw_tolerance_deg:
            break

        extended_route.append((best_wp, None))
        current_wp = best_wp
        current_heading = float(best_wp.transform.rotation.yaw)

    if len(extended_route) <= from_index + 1:
        return straighten_route_tail(route, from_index, num_points, step_m, heading_deg=target_heading)
    return extended_route


def route_transform_with_offset(route, route_index, lateral_offset=0.0, longitudinal_offset=0.0, z_offset=0.5):
    if route_index < 0 or route_index >= len(route):
        raise IndexError(
            f"route_index={route_index} 瓒呭嚭褰撳墠璺嚎鑼冨洿锛屽綋鍓嶈矾绾跨偣鏁?{len(route)}"
        )

    base_transform = route[route_index][0].transform
    yaw_rad = np.radians(base_transform.rotation.yaw)
    forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=float)
    left = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)], dtype=float)
    offset_xy = longitudinal_offset * forward + lateral_offset * left
    base_location = base_transform.location
    location = carla.Location(
        x=base_location.x + float(offset_xy[0]),
        y=base_location.y + float(offset_xy[1]),
        z=base_location.z + float(z_offset),
    )
    return carla.Transform(location, base_transform.rotation)

def _route_point_to_xyz(point):
    """
    鍏煎澶氱 route 鐐规牸寮忥細
    1. carla.Waypoint
    2. carla.Transform
    3. (carla.Waypoint, RoadOption)
    4. (carla.Transform, RoadOption)
    5. (x, y)
    6. (x, y, z)
    """

    # 鎯呭喌 1锛歳oute 鐐规槸 (Waypoint, RoadOption) 鎴?(Transform, RoadOption)
    if isinstance(point, (tuple, list)) and len(point) > 0:
        first = point[0]

        # (Waypoint, RoadOption)
        if hasattr(first, "transform"):
            loc = first.transform.location
            return loc.x, loc.y, loc.z

        # (Transform, RoadOption)
        if hasattr(first, "location"):
            loc = first.location
            return loc.x, loc.y, loc.z

        # (x, y) 鎴?(x, y, z)
        if isinstance(first, (int, float)) and len(point) >= 2:
            x = float(point[0])
            y = float(point[1])
            z = float(point[2]) if len(point) >= 3 and isinstance(point[2], (int, float)) else 0.3
            return x, y, z

    # 鎯呭喌 2锛歳oute 鐐规湰韬槸 Waypoint
    if hasattr(point, "transform"):
        loc = point.transform.location
        return loc.x, loc.y, loc.z

    # 鎯呭喌 3锛歳oute 鐐规湰韬槸 Transform / Location-like
    if hasattr(point, "location"):
        loc = point.location
        return loc.x, loc.y, loc.z

    raise TypeError(f"涓嶆敮鎸佺殑 route 鐐圭被鍨? {type(point)}, 鍐呭: {point}")


def route_transform_with_offset_by_distance(route, start_index, move_distance, lateral_offset=0.0):
    if len(route) < 2:
        raise ValueError("route 鑷冲皯闇€瑕佸寘鍚袱涓偣")

    current_index = max(0, min(int(start_index), len(route) - 2))
    remaining_distance = max(0.0, float(move_distance))

    while current_index < len(route) - 2:
        p0 = route[current_index]
        p1 = route[current_index + 1]

        x0, y0, z0 = _route_point_to_xyz(p0)
        x1, y1, z1 = _route_point_to_xyz(p1)

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)

        if segment_length < 1e-6:
            current_index += 1
            continue

        if remaining_distance <= segment_length:
            ratio = remaining_distance / segment_length

            x = x0 + ratio * dx
            y = y0 + ratio * dy
            z = z0 + ratio * dz

            yaw = math.degrees(math.atan2(dy, dx))

            normal_x = -math.sin(math.radians(yaw))
            normal_y = math.cos(math.radians(yaw))

            location = carla.Location(
                x=x + lateral_offset * normal_x,
                y=y + lateral_offset * normal_y,
                z=z,
            )

            rotation = carla.Rotation(yaw=yaw)
            return carla.Transform(location, rotation)

        remaining_distance -= segment_length
        current_index += 1

    # 瓒呭嚭 route 鏈熬鏃讹紝鏀惧埌鏈€鍚庝竴涓偣
    prev = route[-2]
    last = route[-1]

    x0, y0, z0 = _route_point_to_xyz(prev)
    x1, y1, z1 = _route_point_to_xyz(last)

    yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))

    normal_x = -math.sin(math.radians(yaw))
    normal_y = math.cos(math.radians(yaw))

    location = carla.Location(
        x=x1 + lateral_offset * normal_x,
        y=y1 + lateral_offset * normal_y,
        z=z1,
    )

    rotation = carla.Rotation(yaw=yaw)
    return carla.Transform(location, rotation)


def smoothstep01(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def scripted_merge_lateral_offset(
    move_distance,
    start_lateral_offset,
    end_lateral_offset,
    merge_start_distance,
    merge_distance,
):
    if abs(end_lateral_offset - start_lateral_offset) < 1e-6:
        return float(end_lateral_offset)

    if merge_distance <= 1e-6:
        if move_distance >= merge_start_distance:
            return float(end_lateral_offset)
        return float(start_lateral_offset)

    ratio = (float(move_distance) - float(merge_start_distance)) / float(merge_distance)
    ratio = smoothstep01(ratio)
    return float(start_lateral_offset + ratio * (end_lateral_offset - start_lateral_offset))


def route_transform_with_merge_by_distance(
    route,
    start_index,
    move_distance,
    start_lateral_offset,
    end_lateral_offset,
    merge_start_distance,
    merge_distance,
):
    lateral_offset = scripted_merge_lateral_offset(
        move_distance,
        start_lateral_offset,
        end_lateral_offset,
        merge_start_distance,
        merge_distance,
    )
    transform = route_transform_with_offset_by_distance(
        route,
        start_index,
        move_distance,
        lateral_offset,
    )

    if abs(end_lateral_offset - start_lateral_offset) < 1e-6:
        return transform

    sample_distance = 0.5
    previous_distance = max(0.0, float(move_distance) - sample_distance)
    next_distance = float(move_distance) + sample_distance
    previous_lateral = scripted_merge_lateral_offset(
        previous_distance,
        start_lateral_offset,
        end_lateral_offset,
        merge_start_distance,
        merge_distance,
    )
    next_lateral = scripted_merge_lateral_offset(
        next_distance,
        start_lateral_offset,
        end_lateral_offset,
        merge_start_distance,
        merge_distance,
    )
    previous_transform = route_transform_with_offset_by_distance(
        route,
        start_index,
        previous_distance,
        previous_lateral,
    )
    next_transform = route_transform_with_offset_by_distance(
        route,
        start_index,
        next_distance,
        next_lateral,
    )

    dx = next_transform.location.x - previous_transform.location.x
    dy = next_transform.location.y - previous_transform.location.y
    if math.hypot(dx, dy) > 1e-6:
        transform.rotation.yaw = math.degrees(math.atan2(dy, dx))

    return transform


def project_location_to_route_s(route, location):
    if len(route) < 2:
        raise ValueError("route 鑷冲皯闇€瑕佸寘鍚袱涓偣")

    point = np.array([location.x, location.y], dtype=float)
    best_distance_sq = float("inf")
    best_s = 0.0
    accumulated_s = 0.0

    for index in range(len(route) - 1):
        x0, y0, _ = _route_point_to_xyz(route[index])
        x1, y1, _ = _route_point_to_xyz(route[index + 1])
        start = np.array([x0, y0], dtype=float)
        end = np.array([x1, y1], dtype=float)
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            continue

        ratio = float(np.clip(np.dot(point - start, segment) / (segment_length * segment_length), 0.0, 1.0))
        projected = start + ratio * segment
        distance_sq = float(np.sum((point - projected) * (point - projected)))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_s = accumulated_s + ratio * segment_length

        accumulated_s += segment_length

    return best_s


def destroy_scripted_vehicle(env, entry, sim_time, reason):
    actor = entry["actor"]
    env.scripted_actor_velocities.pop(actor.id, None)
    if actor in env.obstacle_vehicles:
        env.obstacle_vehicles.remove(actor)
    actor.destroy()
    entry["finished"] = True
    print(
        "    Scripted vehicle removed "
        f"(route index={entry['route_index']}, time={sim_time:.2f}s, reason={reason})"
    )


def freeze_scripted_vehicle(env, entry, sim_time, reason):
    actor = entry["actor"]
    if entry.get("frozen", False):
        return

    entry["frozen"] = True
    entry["freeze_transform"] = actor.get_transform()
    env.set_scripted_actor_velocity(actor, (0.0, 0.0))
    print(
        "    Scripted vehicle frozen "
        f"(route index={entry['route_index']}, time={sim_time:.2f}s, reason={reason})"
    )


def spawn_static_obstacles(env, route, route_groups, individual_obstacles):
    obstacle_markers = []

    if hasattr(env, "destroy_obstacle_vehicles"):
        env.destroy_obstacle_vehicles()

    obstacle_configs = []
    for group in route_groups:
        start_index = int(group["start_index"])
        count = int(group["count"])
        step = int(group["step"])
        lateral_offset = float(group["lateral_offset"])
        for offset in range(count):
            obstacle_configs.append(
                {
                    "route_index": start_index + offset * step,
                    "lateral_offset": lateral_offset,
                }
            )

    for obstacle in individual_obstacles:
        obstacle_configs.append(
            {
                "route_index": int(obstacle["route_index"]),
                "lateral_offset": float(obstacle["lateral_offset"]),
            }
        )

    for obstacle in obstacle_configs:
        obstacle_index = obstacle["route_index"]
        obstacle_transform = route_transform_with_offset(
            route,
            obstacle_index,
            lateral_offset=obstacle["lateral_offset"],
        )
        obstacle_actor = env.spawn_obstacle_vehicle(obstacle_transform)
        if obstacle_actor is None:
            raise RuntimeError(f"闈欐€侀殰纰嶇墿鐢熸垚澶辫触锛宺oute_index={obstacle_index}")

        obstacle_loc = obstacle_actor.get_location()
        obstacle_marker = np.array([obstacle_loc.x, obstacle_loc.y], dtype=float)
        obstacle_markers.append(obstacle_marker)
        print(
            "    Static obstacle: spawned "
            f"(route index={obstacle_index}, pos=({obstacle_marker[0]:.2f}, {obstacle_marker[1]:.2f}))"
        )

    return obstacle_markers


def spawn_pedestrian_crossings(env, route, pedestrian_configs):
    pedestrian_markers = []
    scripted_entries = []
    for config in pedestrian_configs:
        route_index = int(config["route_index"])
        speed = float(config["speed"])
        if speed <= 0.0:
            raise ValueError(f"琛屼汉 speed 蹇呴』澶т簬 0锛宺oute_index={route_index}")

        if "start_location" in config and "end_location" in config:
            start_location = config["start_location"]
            end_location = config["end_location"]
            start_transform = carla.Transform(
                carla.Location(
                    x=float(start_location[0]),
                    y=float(start_location[1]),
                    z=float(start_location[2]) if len(start_location) > 2 else 1.0,
                )
            )
            target_transform = carla.Transform(
                carla.Location(
                    x=float(end_location[0]),
                    y=float(end_location[1]),
                    z=float(end_location[2]) if len(end_location) > 2 else 1.0,
                )
            )
        else:
            start_transform = route_transform_with_offset(
                route,
                route_index,
                lateral_offset=float(config["start_lateral_offset"]),
                z_offset=1.0,
            )
            target_transform = route_transform_with_offset(
                route,
                route_index,
                lateral_offset=float(config["end_lateral_offset"]),
                z_offset=1.0,
            )
        dx = target_transform.location.x - start_transform.location.x
        dy = target_transform.location.y - start_transform.location.y
        crossing_yaw = float(np.degrees(np.arctan2(dy, dx)))
        start_transform.rotation.yaw = crossing_yaw
        target_transform.rotation.yaw = crossing_yaw

        actor = env.spawn_scripted_pedestrian(start_transform, speed=speed)
        if actor is None:
            raise RuntimeError(f"鑴氭湰琛屼汉鐢熸垚澶辫触锛宺oute_index={route_index}")

        crossing_length = float(np.hypot(dx, dy))
        duration = float(config.get("duration", crossing_length / speed))
        if duration <= 0.0:
            raise ValueError(f"琛屼汉 duration 蹇呴』澶т簬 0锛宺oute_index={route_index}")

        trigger_distance = config.get("trigger_distance")
        if trigger_distance is not None:
            trigger_distance = float(trigger_distance)
            if trigger_distance < 0.0:
                raise ValueError(
                    f"pedestrian trigger_distance must be non-negative, route_index={route_index}"
                )

        start_time = config.get("start_time")
        if start_time is None and trigger_distance is None:
            start_time = 0.0
        elif start_time is not None:
            start_time = float(start_time)

        entry = {
            "actor": actor,
            "route_index": route_index,
            "start_time": start_time,
            "duration": duration,
            "start_location": start_transform.location,
            "end_location": target_transform.location,
            "rotation": start_transform.rotation,
            "remove_after_crossing": bool(config.get("remove_after_crossing", True)),
            "trigger_distance": trigger_distance,
            "triggered_at_time": None,
            "trigger_location": carla.Location(
                x=0.5 * (start_transform.location.x + target_transform.location.x),
                y=0.5 * (start_transform.location.y + target_transform.location.y),
                z=0.5 * (start_transform.location.z + target_transform.location.z),
            ),
            "finished": False,
        }
        scripted_entries.append(entry)
        env.set_scripted_actor_velocity(actor, (0.0, 0.0))

        pedestrian_markers.append(
            (
                [start_transform.location.x, start_transform.location.y],
                [target_transform.location.x, target_transform.location.y],
            )
        )
        trigger_desc = (
            f"trigger_distance={trigger_distance:.2f}m"
            if trigger_distance is not None
            else f"start_time={entry['start_time']:.2f}s"
        )
        print(
            "    Pedestrian crossing: scripted "
            f"(route index={route_index}, {trigger_desc}, "
            f"duration={duration:.2f}s, speed={speed:.2f}m/s)"
        )

    return pedestrian_markers, scripted_entries


def update_scripted_pedestrian_crossings(env, scripted_entries, sim_time):
    for entry in scripted_entries:
        if entry.get("finished", False):
            continue

        start_time = entry.get("start_time")
        duration = entry["duration"]
        start = entry["start_location"]
        end = entry["end_location"]
        trigger_distance = entry.get("trigger_distance")

        if trigger_distance is not None and entry.get("triggered_at_time") is None:
            ego_vehicle = getattr(env, "ego_vehicle", None)
            if ego_vehicle is not None:
                ego_loc = ego_vehicle.get_location()
                trigger_loc = entry["trigger_location"]
                remaining_distance = math.hypot(
                    ego_loc.x - trigger_loc.x,
                    ego_loc.y - trigger_loc.y,
                )
                if remaining_distance <= trigger_distance:
                    entry["triggered_at_time"] = sim_time
                    print(
                        "    Pedestrian crossing triggered "
                        f"(route index={entry['route_index']}, time={sim_time:.2f}s, "
                        f"distance={remaining_distance:.2f}m)"
                    )

        if trigger_distance is not None:
            effective_start_time = entry.get("triggered_at_time")
        else:
            effective_start_time = start_time

        if effective_start_time is None or sim_time < effective_start_time:
            ratio = 0.0
            velocity = (0.0, 0.0)
        elif sim_time > effective_start_time + duration:
            if entry.get("remove_after_crossing", True):
                actor = entry["actor"]
                try:
                    env.scripted_actor_velocities.pop(actor.id, None)
                    if actor in env.pedestrians:
                        env.pedestrians.remove(actor)
                    actor.destroy()
                except Exception:
                    pass
                entry["finished"] = True
                continue

            ratio = 1.0
            velocity = (0.0, 0.0)
        else:
            ratio = (sim_time - effective_start_time) / duration
            velocity = ((end.x - start.x) / duration, (end.y - start.y) / duration)

        location = carla.Location(
            x=start.x + ratio * (end.x - start.x),
            y=start.y + ratio * (end.y - start.y),
            z=start.z + ratio * (end.z - start.z),
        )
        entry["actor"].set_transform(carla.Transform(location, entry["rotation"]))
        env.set_scripted_actor_velocity(entry["actor"], velocity)


def spawn_scripted_vehicle_obstacles(env, route, vehicle_configs):
    scripted_entries = []

    for config in vehicle_configs:
        route_index = int(config["route_index"])

        start_lateral_offset = float(config["start_lateral_offset"])
        end_lateral_offset = float(config["end_lateral_offset"])

        duration = float(config["duration"])
        if duration <= 0.0:
            raise ValueError(f"鑴氭湰鍔ㄦ€佽溅 duration 蹇呴』澶т簬 0锛宺oute_index={route_index}")

        speed_kmh = float(config.get("speed_kmh", 0.0))
        speed_mps = speed_kmh / 3.6
        lane_change_duration = float(config.get("lane_change_duration", duration))
        if lane_change_duration <= 0.0:
            raise ValueError(f"鑴氭湰鍔ㄦ€佽溅 lane_change_duration 蹇呴』澶т簬 0锛宺oute_index={route_index}")

        freeze_after_ego_passed_s = config.get("freeze_after_ego_passed_s")
        if freeze_after_ego_passed_s is not None:
            freeze_after_ego_passed_s = float(freeze_after_ego_passed_s)
            if freeze_after_ego_passed_s < 0.0:
                raise ValueError(
                    f"scripted vehicle freeze_after_ego_passed_s must be non-negative, route_index={route_index}"
                )

        trigger_distance = config.get("trigger_distance")
        if trigger_distance is not None:
            trigger_distance = float(trigger_distance)
            if trigger_distance < 0.0:
                raise ValueError(
                    f"scripted vehicle trigger_distance must be non-negative, route_index={route_index}"
                )

        has_lateral_change = abs(end_lateral_offset - start_lateral_offset) > 1e-6
        motion_type = str(
            config.get(
                "motion_type",
                "smooth_merge" if has_lateral_change and speed_mps > 0.0 else "offset_follow",
            )
        )
        merge_start_distance = float(config.get("merge_start_distance", 0.0))
        if merge_start_distance < 0.0:
            raise ValueError(f"scripted vehicle merge_start_distance must be non-negative, route_index={route_index}")
        merge_distance = config.get("merge_distance")
        if merge_distance is None:
            merge_distance = speed_mps * lane_change_duration
        merge_distance = float(merge_distance)
        if has_lateral_change and motion_type == "smooth_merge" and merge_distance <= 0.0:
            raise ValueError(f"scripted vehicle merge_distance must be positive, route_index={route_index}")

        start_transform = route_transform_with_merge_by_distance(
            route,
            route_index,
            0.0,
            start_lateral_offset,
            end_lateral_offset,
            merge_start_distance,
            merge_distance,
        )

        if speed_mps > 0.0:
            end_transform = route_transform_with_merge_by_distance(
                route,
                route_index,
                speed_mps * duration,
                start_lateral_offset,
                end_lateral_offset,
                merge_start_distance,
                merge_distance,
            )
        else:
            end_transform = route_transform_with_offset(
                route,
                route_index,
                lateral_offset=end_lateral_offset,
            )

        entry = {
            "actor": None,
            "route": route,
            "route_index": route_index,
            "start_time": float(config["start_time"]),
            "trigger_distance": trigger_distance,
            "triggered_at_time": None,
            "duration": duration,
            "lane_change_duration": lane_change_duration,
            "remove_after_duration": bool(config.get("remove_after_duration", False)),
            "remove_after_ego_passed_m": config.get("remove_after_ego_passed_m"),
            "freeze_after_ego_passed_s": freeze_after_ego_passed_s,
            "start_lateral_offset": start_lateral_offset,
            "end_lateral_offset": end_lateral_offset,
            "motion_type": motion_type,
            "merge_start_distance": merge_start_distance,
            "merge_distance": merge_distance,
            "start_transform": start_transform,
            "start_location": start_transform.location,
            "end_location": end_transform.location,
            "rotation": start_transform.rotation,
            "speed_kmh": speed_kmh,
            "speed_mps": speed_mps,
            "ego_passed_time": None,
            "frozen": False,
            "freeze_transform": None,
            "path_points": [],
        }

        scripted_entries.append(entry)

        print(
            "    Scripted vehicle: scheduled "
            f"(route index={route_index}, "
            f"{f'trigger_distance={trigger_distance:.2f}m, ' if trigger_distance is not None else ''}"
            f"start_time={entry['start_time']:.2f}s, "
            f"duration={duration:.2f}s, "
            f"speed={speed_kmh:.2f}km/h, "
            f"motion={motion_type})"
        )

    return scripted_entries


def update_scripted_vehicle_obstacles(env, scripted_entries, sim_time):
    for entry in scripted_entries:
        if entry.get("finished", False):
            continue

        route = entry["route"]

        start_time = entry["start_time"]
        trigger_distance = entry.get("trigger_distance")
        duration = entry["duration"]
        lane_change_duration = entry.get("lane_change_duration", duration)
        route_index = entry["route_index"]

        start_lateral_offset = entry["start_lateral_offset"]
        end_lateral_offset = entry["end_lateral_offset"]

        speed_mps = entry.get("speed_mps", 0.0)
        motion_type = entry.get("motion_type", "offset_follow")
        merge_start_distance = entry.get("merge_start_distance", 0.0)
        merge_distance = entry.get("merge_distance", speed_mps * lane_change_duration)

        if trigger_distance is not None and entry.get("triggered_at_time") is None:
            ego_vehicle = getattr(env, "ego_vehicle", None)
            if ego_vehicle is None:
                continue
            ego_s = project_location_to_route_s(route, ego_vehicle.get_location())
            obstacle_s = project_location_to_route_s(route, entry["start_location"])
            distance_ahead = obstacle_s - ego_s
            if 0.0 <= distance_ahead <= trigger_distance:
                entry["triggered_at_time"] = sim_time
                print(
                    "    Scripted vehicle triggered "
                    f"(route index={route_index}, time={sim_time:.2f}s, "
                    f"route_distance={distance_ahead:.2f}m)"
                )

        effective_start_time = entry.get("triggered_at_time") if trigger_distance is not None else start_time
        if effective_start_time is None or sim_time < effective_start_time:
            continue
        else:
            elapsed = sim_time - effective_start_time

        actor = entry["actor"]
        if actor is None:
            actor = env.spawn_obstacle_vehicle(entry["start_transform"])
            if actor is None:
                raise RuntimeError(f"鑴氭湰鍔ㄦ€佽溅鐢熸垚澶辫触锛宺oute_index={route_index}")
            entry["actor"] = actor
            print(
                "    Scripted vehicle spawned "
                f"(route index={route_index}, time={sim_time:.2f}s)"
            )

        if entry.get("frozen", False):
            freeze_transform = entry["freeze_transform"]
            actor.set_transform(freeze_transform)
            env.set_scripted_actor_velocity(actor, (0.0, 0.0))
            continue

        remove_after_ego_passed_m = entry.get("remove_after_ego_passed_m")
        freeze_after_ego_passed_s = entry.get("freeze_after_ego_passed_s")
        if (
            elapsed > duration
            and entry.get("remove_after_duration", False)
            and remove_after_ego_passed_m is None
            and freeze_after_ego_passed_s is None
        ):
            destroy_scripted_vehicle(env, entry, sim_time, "duration")
            continue

        # 妯悜鎻掑€艰繘搴︼紝鐢ㄤ簬 cut-in锛涜溅杈嗙旱鍚戣椹舵椂闂翠粛鐢?duration 鎺у埗
        if sim_time < effective_start_time:
            lateral_ratio = 0.0
        elif sim_time > effective_start_time + lane_change_duration:
            lateral_ratio = 1.0
        else:
            lateral_ratio = elapsed / lane_change_duration

        lateral_offset = (
            start_lateral_offset
            + lateral_ratio * (end_lateral_offset - start_lateral_offset)
        )

        # ==================================================
        # 鎯呭喌 1锛歴peed_kmh > 0
        # 浣庨€熷墠杞︼細娌?route 鍓嶈繘锛屽苟鍦?duration 鍚庡仠姝?        # ==================================================
        if speed_mps > 0.0:
            move_elapsed = min(elapsed, duration)
            move_distance = speed_mps * move_elapsed

            if motion_type == "smooth_merge":
                current_transform = route_transform_with_merge_by_distance(
                    route,
                    route_index,
                    move_distance,
                    start_lateral_offset,
                    end_lateral_offset,
                    merge_start_distance,
                    merge_distance,
                )
            else:
                current_transform = route_transform_with_offset_by_distance(
                    route,
                    route_index,
                    move_distance,
                    lateral_offset,
                )

            # 淇濇寔鐢熸垚鏃剁殑楂樺害锛岄伩鍏嶈溅杈嗛璧锋垨闄峰叆鍦伴潰
            if "spawn_z" in entry:
                current_transform.location.z = entry["spawn_z"]

            actor.set_transform(current_transform)
            if elapsed <= duration:
                entry["path_points"].append(
                    (current_transform.location.x, current_transform.location.y)
                )

            # duration 缁撴潫鍚庯紝閫熷害缃浂
            if elapsed >= duration:
                velocity = (0.0, 0.0)
            else:
                yaw_rad = math.radians(current_transform.rotation.yaw)
                velocity = (
                    speed_mps * math.cos(yaw_rad),
                    speed_mps * math.sin(yaw_rad),
                )

            env.set_scripted_actor_velocity(actor, velocity)

        # ==================================================
        # 鎯呭喌 2锛歴peed_kmh == 0
        # 鍘熼€昏緫锛氬悓涓€涓?route_index 涓婂仛妯悜鎻掑€?        # 鐢ㄤ簬鍔ㄦ€佸垏鍏ヨ溅 cut-in
        # ==================================================
        else:
            start = entry["start_location"]
            end = entry["end_location"]

            if sim_time < start_time:
                # keep hidden until trigger
                ratio = 0.0
                velocity = (0.0, 0.0)
            elif sim_time > effective_start_time + duration:
                ratio = 1.0
                velocity = (0.0, 0.0)
            else:
                ratio = (sim_time - effective_start_time) / duration
                velocity = (
                    (end.x - start.x) / duration,
                    (end.y - start.y) / duration,
                )

            location = carla.Location(
                x=start.x + ratio * (end.x - start.x),
                y=start.y + ratio * (end.y - start.y),
                z=start.z + ratio * (end.z - start.z),
            )

            actor.set_transform(carla.Transform(location, entry["rotation"]))
            env.set_scripted_actor_velocity(actor, velocity)
            if elapsed <= duration:
                entry["path_points"].append((location.x, location.y))

        if entry.get("remove_after_duration", False) and remove_after_ego_passed_m is not None:
            ego_vehicle = getattr(env, "ego_vehicle", None)
            if ego_vehicle is None:
                continue
            ego_s = project_location_to_route_s(route, ego_vehicle.get_location())
            obstacle_s = project_location_to_route_s(route, actor.get_location())
            if ego_s - obstacle_s >= float(remove_after_ego_passed_m):
                if freeze_after_ego_passed_s is None:
                    destroy_scripted_vehicle(env, entry, sim_time, "ego_passed")
                    continue
                if entry.get("ego_passed_time") is None:
                    entry["ego_passed_time"] = sim_time

        if (
            freeze_after_ego_passed_s is not None
            and entry.get("ego_passed_time") is not None
            and not entry.get("frozen", False)
            and sim_time - entry["ego_passed_time"] >= freeze_after_ego_passed_s
        ):
            freeze_scripted_vehicle(env, entry, sim_time, "ego_passed")

