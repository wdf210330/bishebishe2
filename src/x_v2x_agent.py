import math
import os
import time
from collections import deque

import carla
import numpy as np
from scipy.interpolate import CubicSpline

try:
    from . import carla_utils as ca_u
    from .global_route_planner import GlobalRoutePlanner
    from .project_paths import ensure_debug_log_dir
except ImportError:
    import carla_utils as ca_u
    from global_route_planner import GlobalRoutePlanner
    from project_paths import ensure_debug_log_dir


class Xagent:
    def __init__(self, env, model, dt=0.1) -> None:
        self._env = env
        self._vehicle = env.ego_vehicle
        self._model = model
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._global_planner = GlobalRoutePlanner(self._map, 2.0)

        self._waypoints_queue = deque(maxlen=100000)
        self._route = []
        self._route_cache = None
        self._route_start_index = 0
        self._route_progress_s = 0.0
        self._last_unwrapped_yaw = None

        self._base_min_distance = 2.0
        self._sample_resolution = 2.0
        self._smooth_path_ds = 0.5
        self._speed_profile_ds = 0.5
        self._reference_yaw_window = 6.0
        self._curvature_sample_window = 3.0
        self._max_reference_lateral_accel = 0.8
        self._reference_comfort_decel = 1.2
        self._reference_comfort_accel = 0.8
        self._min_curve_speed = 2.4
        self._projection_min_lookahead = 15.0
        self._projection_time_lookahead = 3.0

        self._a_opt = np.array([0.0] * self._model.horizon)
        self._delta_opt = np.array([0.0] * self._model.horizon)
        self._last_control = np.array([0.0, 0.0], dtype=float)
        self._next_states = None
        self._dt = dt
        self._draw_planned_trj = os.environ.get("MPC_DRAW_PLANNED_TRAJ", "0") == "1"
        self._debug_log_enabled = os.environ.get("MPC_DEBUG_LOG", "0") == "1"

        self._obstacles = None
        self.enable_obstacle_avoidance = True
        self.enable_obstacle_reference_shaping = True
        self.obstacle_detection_range = 105.0
        self._obstacle_reference_lateral_limit = 2.4
        self._last_reference_speed = float(self._model.target_v)
        self._last_min_obstacle_clearance = float("inf")
        self._obstacle_bypass_speed = min(float(self._model.target_v), 12.0 / 3.6)
        self._obstacle_speed_entry_distance = 22.0
        self._obstacle_speed_exit_distance = 12.0
        self._lane_bound_inactive = 1.0e6
        self._pedestrian_safe_dist = 2.0
        self._pedestrian_yield_distance = 12.0
        self._pedestrian_path_half_width = 1.6
        self._pedestrian_release_lateral = 2.1
        self._pedestrian_stop_buffer = 2.3
        self._pedestrian_crossing_lookahead_time = 3.0
        self._pedestrian_min_decel = 0.25
        self._pedestrian_comfort_decel = 0.7
        self._pedestrian_strong_decel = 1.6
        self._terminal_reference_speed = float(self._model.target_v)
        self._route_completion_distance = 1.0
        self._route_completion_speed = 0.8
        self._preserve_terminal_waypoint = False
        self._dynamic_vehicle_prediction_time = 3.0
        self._dynamic_vehicle_conflict_lateral = 1.4
        self._dynamic_vehicle_watch_lateral = 4.5
        self._dynamic_vehicle_yield_distance = 28.0
        self._dynamic_vehicle_time_headway = 1.6
        self._dynamic_vehicle_stop_buffer = 5.0
        self._dynamic_vehicle_reference_decel = 2.3
        self._dynamic_vehicle_safe_dist = 1.8
        self._dynamic_vehicle_min_decel = 0.3
        self._dynamic_vehicle_comfort_decel = 1.1
        self._dynamic_vehicle_strong_decel = 3.2
        self._dynamic_vehicle_crawl_speed = 1.2
        self._dynamic_vehicle_follow_buffer = 0.8
        self._dynamic_vehicle_follow_time_headway = 0.20
        self._dynamic_vehicle_follow_clearance_margin = 0.65
        self._dynamic_vehicle_follow_speed_deficit_gain = 0.85
        self._dynamic_vehicle_follow_max_slowdown = 1.0
        self._dynamic_overtake_same_lane_lateral = 1.75
        self._dynamic_overtake_min_speed_delta = 0.8
        self._dynamic_overtake_start_distance = 20.0
        self._dynamic_overtake_full_distance = 6.5
        self._dynamic_overtake_return_front_clearance_lengths = 1.2
        self._dynamic_overtake_return_distance = 14.0
        self._dynamic_vehicle_rear_release_clearance = 1.0
        self._dynamic_follow_gate_clearance = 16.0
        self._dynamic_lane_change_front_gap = 14.0
        self._dynamic_lane_change_rear_gap = 6.0
        self._dynamic_lane_change_target_lane_tolerance = 1.6

        self._model.solver_basis(
            Q=np.diag([10, 10, 8, 6, 0]),
            R=np.diag([1, 2]),
            Rd=np.diag([0.5, 60]),
        )
        self._model.initialize_solver()

    def plan_route(self, start_location, end_location):
        route = self.trace_route(start_location.location, end_location.location)
        self.set_route(route)

    def trace_route(self, start_location, end_location):
        return self._global_planner.trace_route(start_location, end_location)

    def set_start_end_transforms(self, start_idx, end_idx):
        spawn_points = self._map.get_spawn_points()
        if start_idx < len(spawn_points) and end_idx < len(spawn_points):
            self._start_transform = spawn_points[start_idx]
            self._end_transform = spawn_points[end_idx]
            return
        raise IndexError("Start or end index out of bounds!")

    def set_route(self, route):
        self._route = list(route)
        self._waypoints_queue.clear()
        for item in self._route:
            self._waypoints_queue.append(item)
        self._route_start_index = 0
        self._route_progress_s = 0.0
        self._last_unwrapped_yaw = None
        self._route_cache = self._build_route_cache(self._route)

    def set_obstacles(self, obstacles):
        self._obstacles = np.asarray(obstacles, dtype=float) if obstacles is not None else None

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _append_debug_log(self, message):
        if not self._debug_log_enabled:
            return
        log_dir = ensure_debug_log_dir()
        log_file = log_dir / "run_debug.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    def _unwrap_angle(self, angle, anchor=None):
        angle = float(angle)
        if anchor is None:
            if self._last_unwrapped_yaw is None:
                self._last_unwrapped_yaw = angle
                return angle
            anchor = float(self._last_unwrapped_yaw)
            unwrapped = anchor + self._normalize_angle(angle - anchor)
            self._last_unwrapped_yaw = unwrapped
            return unwrapped
        anchor = float(anchor)
        return anchor + self._normalize_angle(angle - anchor)

    def _lane_change_allows(self, marking, direction):
        if marking is None:
            return False
        if direction == "left":
            return bool(marking.lane_change & carla.LaneChange.Left)
        return bool(marking.lane_change & carla.LaneChange.Right)

    def _same_direction_driving_lane(self, source_wp, target_wp):
        if target_wp is None or target_wp.lane_type != carla.LaneType.Driving:
            return False
        source_yaw = math.radians(source_wp.transform.rotation.yaw)
        target_yaw = math.radians(target_wp.transform.rotation.yaw)
        return abs(self._normalize_angle(target_yaw - source_yaw)) < math.pi / 2.0

    def _adjacent_lane_lateral_offset(self, waypoint, adjacent_waypoint):
        yaw = math.radians(waypoint.transform.rotation.yaw)
        left_axis = (-math.sin(yaw), math.cos(yaw))
        source_loc = waypoint.transform.location
        target_loc = adjacent_waypoint.transform.location
        dx = target_loc.x - source_loc.x
        dy = target_loc.y - source_loc.y
        return dx * left_axis[0] + dy * left_axis[1]

    def _extend_lane_bound(
        self,
        waypoint,
        adjacent_waypoint,
        current_lane_width,
        ego_half_width,
        margin,
        left_bound,
        right_bound,
    ):
        lateral_offset = self._adjacent_lane_lateral_offset(waypoint, adjacent_waypoint)
        adjacent_width = max(
            float(getattr(adjacent_waypoint, "lane_width", current_lane_width)),
            1e-3,
        )
        extended_bound = max(
            0.0,
            abs(lateral_offset) + 0.5 * adjacent_width - ego_half_width - margin,
        )
        if lateral_offset > 0.0:
            left_bound = max(left_bound, extended_bound)
        elif lateral_offset < 0.0:
            right_bound = max(right_bound, extended_bound)
        return left_bound, right_bound

    def _reference_lane_bounds(self, waypoint):
        if waypoint is None:
            return np.array([self._lane_bound_inactive, self._lane_bound_inactive], dtype=float)

        lane_width = max(float(getattr(waypoint, "lane_width", 3.5)), 1e-3)
        ego_half_width = max(float(getattr(self._model, "ego_footprint_radius", 1.0)), 0.0)
        margin = max(float(getattr(self._model, "lane_boundary_margin", 0.2)), 0.0)
        base_bound = max(0.0, 0.5 * lane_width - ego_half_width - margin)

        left_bound = base_bound
        right_bound = base_bound

        left_lane = waypoint.get_left_lane() if hasattr(waypoint, "get_left_lane") else None
        if (
            self._lane_change_allows(getattr(waypoint, "left_lane_marking", None), "left")
            and self._same_direction_driving_lane(waypoint, left_lane)
        ):
            left_bound, right_bound = self._extend_lane_bound(
                waypoint,
                left_lane,
                lane_width,
                ego_half_width,
                margin,
                left_bound,
                right_bound,
            )

        right_lane = waypoint.get_right_lane() if hasattr(waypoint, "get_right_lane") else None
        if (
            self._lane_change_allows(getattr(waypoint, "right_lane_marking", None), "right")
            and self._same_direction_driving_lane(waypoint, right_lane)
        ):
            left_bound, right_bound = self._extend_lane_bound(
                waypoint,
                right_lane,
                lane_width,
                ego_half_width,
                margin,
                left_bound,
                right_bound,
            )

        return np.array([left_bound, right_bound], dtype=float)

    @staticmethod
    def _route_xy(route_item):
        waypoint = route_item[0] if isinstance(route_item, (tuple, list)) else route_item
        loc = waypoint.transform.location
        return np.array([loc.x, loc.y], dtype=float)

    def _build_route_cache(self, route):
        points = []
        entry_s = []
        raw_lane_bounds = []
        current_s = 0.0
        is_manual_route = True

        for item in route:
            waypoint = item[0] if isinstance(item, (tuple, list)) else item
            is_manual_route = is_manual_route and bool(getattr(waypoint, "is_manual_route", False))
            point = self._route_xy(item)
            if len(points) == 0 or np.linalg.norm(point - points[-1]) > 1e-3:
                if points:
                    current_s += float(np.linalg.norm(point - points[-1]))
                points.append(point)
                raw_lane_bounds.append(self._reference_lane_bounds(waypoint))
            entry_s.append(current_s)

        if len(points) < 2:
            return None

        raw_points = np.array(points, dtype=float)
        raw_lane_bounds = np.asarray(raw_lane_bounds, dtype=float)
        raw_segment_lengths = np.hypot(np.diff(raw_points[:, 0]), np.diff(raw_points[:, 1]))
        raw_arc_lengths = np.concatenate(([0.0], np.cumsum(raw_segment_lengths)))

        if is_manual_route or len(raw_points) < 4:
            source_s, path_points, segment_lengths, arc_lengths = self._resample_polyline(
                raw_points,
                raw_arc_lengths,
            )
        else:
            source_s, path_points, segment_lengths, arc_lengths = self._smooth_polyline(
                raw_points,
                raw_arc_lengths,
            )

        lane_bounds = np.column_stack(
            (
                np.interp(source_s, raw_arc_lengths, raw_lane_bounds[:, 0]),
                np.interp(source_s, raw_arc_lengths, raw_lane_bounds[:, 1]),
            )
        )

        entry_s = np.interp(np.array(entry_s, dtype=float), source_s, arc_lengths)
        speed_profile = self._reference_speed_profile(
            path_points,
            arc_lengths,
            segment_lengths,
            0.0,
            end_s=float(arc_lengths[-1]),
        )

        return {
            "path_points": path_points,
            "segment_lengths": segment_lengths,
            "arc_lengths": arc_lengths,
            "entry_s": entry_s,
            "speed_profile": speed_profile,
            "lane_bounds": lane_bounds,
        }

    def _resample_polyline(self, points, arc_lengths):
        total_s = float(arc_lengths[-1])
        ds = max(float(self._smooth_path_ds), 0.05)
        source_s = np.arange(0.0, total_s, ds, dtype=float)
        if len(source_s) == 0 or source_s[-1] < total_s:
            source_s = np.append(source_s, total_s)
        sampled = np.column_stack(
            (
                np.interp(source_s, arc_lengths, points[:, 0]),
                np.interp(source_s, arc_lengths, points[:, 1]),
            )
        )
        seg = np.hypot(np.diff(sampled[:, 0]), np.diff(sampled[:, 1]))
        sampled_s = np.concatenate(([0.0], np.cumsum(seg)))
        return source_s, sampled, seg, sampled_s

    def _smooth_polyline(self, points, arc_lengths):
        total_s = float(arc_lengths[-1])
        ds = max(float(self._smooth_path_ds), 0.05)
        source_s = np.arange(0.0, total_s, ds, dtype=float)
        if len(source_s) == 0 or source_s[-1] < total_s:
            source_s = np.append(source_s, total_s)
        spline_x = CubicSpline(arc_lengths, points[:, 0], bc_type="natural")
        spline_y = CubicSpline(arc_lengths, points[:, 1], bc_type="natural")
        sampled = np.column_stack((spline_x(source_s), spline_y(source_s)))
        keep = np.ones(len(sampled), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(sampled, axis=0), axis=1) > 1e-5
        source_s = source_s[keep]
        sampled = sampled[keep]
        seg = np.hypot(np.diff(sampled[:, 0]), np.diff(sampled[:, 1]))
        sampled_s = np.concatenate(([0.0], np.cumsum(seg)))
        return source_s, sampled, seg, sampled_s

    def _project_to_path(self, point, path_points, arc_lengths, segment_lengths, min_s=None, max_s=None, heading_yaw=None):
        point = np.asarray(point, dtype=float)
        segment_count = len(segment_lengths)
        if segment_count <= 0:
            return 0.0, path_points[0], 0.0, 0.0

        lower_s = 0.0 if min_s is None else float(min_s)
        upper_s = float(arc_lengths[-1]) if max_s is None else float(max_s)
        if upper_s < lower_s:
            lower_s, upper_s = upper_s, lower_s
        start_idx = int(np.searchsorted(arc_lengths[1:], lower_s, side="left"))
        end_idx = int(np.searchsorted(arc_lengths[:-1], upper_s, side="right"))
        start_idx = min(max(start_idx, 0), segment_count - 1)
        end_idx = min(max(end_idx, start_idx + 1), segment_count)

        starts = path_points[start_idx:end_idx]
        ends = path_points[start_idx + 1 : end_idx + 1]
        local_lengths = segment_lengths[start_idx:end_idx]
        vectors = ends - starts
        ratios = np.sum((point - starts) * vectors, axis=1) / np.maximum(local_lengths * local_lengths, 1e-9)
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts + ratios[:, None] * vectors
        distances_sq = np.sum((point - projections) * (point - projections), axis=1)
        candidate_s = arc_lengths[start_idx:end_idx] + ratios * local_lengths

        scores = distances_sq.copy()
        if heading_yaw is not None:
            segment_yaws = np.arctan2(vectors[:, 1], vectors[:, 0])
            yaw_errors = np.arctan2(np.sin(segment_yaws - heading_yaw), np.cos(segment_yaws - heading_yaw))
            scores += 4.0 * yaw_errors * yaw_errors

        local_idx = int(np.argmin(scores))
        yaw = float(np.arctan2(vectors[local_idx, 1], vectors[local_idx, 0]))
        lat_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=float)
        lateral = float((point - projections[local_idx]) @ lat_axis)
        return float(candidate_s[local_idx]), projections[local_idx], yaw, lateral

    def _sample_path_position(self, path_points, arc_lengths, segment_lengths, path_s):
        path_s = float(np.clip(path_s, 0.0, arc_lengths[-1]))
        seg_idx = int(np.searchsorted(arc_lengths, path_s, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(segment_lengths) - 1)
        ratio = (path_s - arc_lengths[seg_idx]) / max(float(segment_lengths[seg_idx]), 1e-6)
        return path_points[seg_idx] + float(np.clip(ratio, 0.0, 1.0)) * (path_points[seg_idx + 1] - path_points[seg_idx])

    def _sample_path(self, path_points, arc_lengths, segment_lengths, path_s):
        path_s = float(np.clip(path_s, 0.0, arc_lengths[-1]))
        pos = self._sample_path_position(path_points, arc_lengths, segment_lengths, path_s)
        s0 = max(path_s - self._reference_yaw_window, 0.0)
        s1 = min(path_s + self._reference_yaw_window, arc_lengths[-1])
        p0 = self._sample_path_position(path_points, arc_lengths, segment_lengths, s0)
        p1 = self._sample_path_position(path_points, arc_lengths, segment_lengths, s1)
        delta = p1 - p0
        if np.linalg.norm(delta) < 1e-6:
            s1 = min(path_s + self._reference_yaw_window, arc_lengths[-1])
            p1 = self._sample_path_position(path_points, arc_lengths, segment_lengths, s1)
            delta = p1 - pos
        if np.linalg.norm(delta) < 1e-6:
            seg_idx = int(np.searchsorted(arc_lengths, path_s, side="right") - 1)
            seg_idx = min(max(seg_idx, 0), len(segment_lengths) - 1)
            delta = path_points[seg_idx + 1] - path_points[seg_idx]
        yaw = float(np.arctan2(delta[1], delta[0]))
        return pos, yaw

    def _path_curvature_at(self, path_points, arc_lengths, segment_lengths, path_s):
        window = max(float(self._curvature_sample_window), 0.5)
        p0 = self._sample_path_position(path_points, arc_lengths, segment_lengths, max(path_s - window, 0.0))
        p1 = self._sample_path_position(path_points, arc_lengths, segment_lengths, path_s)
        p2 = self._sample_path_position(path_points, arc_lengths, segment_lengths, min(path_s + window, arc_lengths[-1]))
        a = float(np.linalg.norm(p1 - p0))
        b = float(np.linalg.norm(p2 - p1))
        c = float(np.linalg.norm(p2 - p0))
        if min(a, b, c) < 1e-6:
            return 0.0
        area_twice = abs(float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])))
        return 2.0 * area_twice / max(a * b * c, 1e-9)

    def _curve_speed_limit_at(self, path_points, arc_lengths, segment_lengths, path_s):
        target_speed = float(self._model.target_v)
        curvature = self._path_curvature_at(path_points, arc_lengths, segment_lengths, path_s)
        if curvature < 1e-4:
            return target_speed
        return float(np.clip(math.sqrt(self._max_reference_lateral_accel / curvature), self._min_curve_speed, target_speed))

    def _reference_speed_profile(self, path_points, arc_lengths, segment_lengths, current_s, end_s=None):
        target_speed = float(self._model.target_v)
        if target_speed <= 0.0:
            return np.array([current_s], dtype=float), np.array([target_speed], dtype=float)
        end_s = float(arc_lengths[-1]) if end_s is None else min(float(end_s), float(arc_lengths[-1]))
        ds = max(float(self._speed_profile_ds), 0.1)
        grid_s = np.arange(float(current_s), end_s, ds, dtype=float)
        if len(grid_s) == 0 or abs(grid_s[-1] - end_s) > 1e-6:
            grid_s = np.append(grid_s, end_s)
        grid_v = np.array([self._curve_speed_limit_at(path_points, arc_lengths, segment_lengths, s) for s in grid_s], dtype=float)
        if len(grid_v) > 0:
            grid_v[-1] = min(grid_v[-1], float(self._terminal_reference_speed))

        max_decel = max(float(self._reference_comfort_decel), 1e-3)
        for i in range(len(grid_s) - 2, -1, -1):
            ds_i = max(float(grid_s[i + 1] - grid_s[i]), 1e-6)
            grid_v[i] = min(grid_v[i], math.sqrt(grid_v[i + 1] * grid_v[i + 1] + 2.0 * max_decel * ds_i))
        max_accel = max(float(self._reference_comfort_accel), 1e-3)
        for i in range(len(grid_s) - 1):
            ds_i = max(float(grid_s[i + 1] - grid_s[i]), 1e-6)
            grid_v[i + 1] = min(grid_v[i + 1], math.sqrt(grid_v[i] * grid_v[i] + 2.0 * max_accel * ds_i))
        return grid_s, grid_v

    @staticmethod
    def _speed_at_profile_s(speed_profile, path_s):
        grid_s, grid_v = speed_profile
        return float(np.interp(float(path_s), grid_s, grid_v))

    def _advance_reference_s(self, current_s, speed_profile, total_length):
        ref_speed = self._speed_at_profile_s(speed_profile, current_s)
        return min(float(current_s) + max(ref_speed, 0.5) * self._dt, float(total_length))

    def _update_route_progress(self, x, y, yaw, speed):
        cache = self._route_cache
        if cache is None:
            return
        total_s = float(cache["arc_lengths"][-1])
        previous_s = float(np.clip(self._route_progress_s, 0.0, total_s))
        search_back = max(4.0, self._base_min_distance + 0.2 * max(speed, 0.0))
        search_forward = max(
            self._projection_min_lookahead,
            self._base_min_distance + self._projection_time_lookahead * max(speed, 0.0),
            25.0,
        )
        projected_s, _, _, _ = self._project_to_path(
            [x, y],
            cache["path_points"],
            cache["arc_lengths"],
            cache["segment_lengths"],
            min_s=max(previous_s - search_back, 0.0),
            max_s=min(previous_s + search_forward, total_s),
            heading_yaw=yaw,
        )
        self._route_progress_s = max(previous_s, float(projected_s))
        new_start_index = int(np.searchsorted(cache["entry_s"], self._route_progress_s, side="right") - 1)
        new_start_index = min(max(new_start_index, 0), len(cache["entry_s"]) - 1)
        remove_count = max(new_start_index - self._route_start_index, 0)
        if self._preserve_terminal_waypoint and len(self._waypoints_queue) > 0:
            remove_count = min(remove_count, max(len(self._waypoints_queue) - 1, 0))
        for _ in range(min(remove_count, len(self._waypoints_queue))):
            self._waypoints_queue.popleft()
        self._route_start_index = new_start_index

    def _window_geometry_from_route_cache(self):
        cache = self._route_cache
        if cache is None:
            return None

        total_s = float(cache["arc_lengths"][-1])
        if total_s <= 1e-6:
            return None

        start_s = float(np.clip(self._route_progress_s, 0.0, total_s))
        preview_distance = max(
            self._projection_min_lookahead,
            float(self._model.target_v) * self._dt * (self._model.horizon + 80),
            60.0,
        )
        end_s = min(start_s + preview_distance, total_s)
        if end_s <= start_s + 1e-6:
            return None

        smooth_arc = cache["arc_lengths"]
        start_index = int(np.searchsorted(smooth_arc, start_s, side="right"))
        end_index = int(np.searchsorted(smooth_arc, end_s, side="left"))
        window_s = np.concatenate(([start_s], smooth_arc[start_index:end_index], [end_s]))

        path_points = np.empty((len(window_s), 2), dtype=float)
        cache_points = cache["path_points"]
        path_points[:, 0] = np.interp(window_s, smooth_arc, cache_points[:, 0])
        path_points[:, 1] = np.interp(window_s, smooth_arc, cache_points[:, 1])
        lane_bounds = np.empty((len(window_s), 2), dtype=float)
        cache_lane_bounds = cache["lane_bounds"]
        lane_bounds[:, 0] = np.interp(window_s, smooth_arc, cache_lane_bounds[:, 0])
        lane_bounds[:, 1] = np.interp(window_s, smooth_arc, cache_lane_bounds[:, 1])

        if len(path_points) > 1:
            keep = np.ones(len(path_points), dtype=bool)
            deltas = np.diff(path_points, axis=0)
            keep[1:] = np.einsum("ij,ij->i", deltas, deltas) > 1e-8
            path_points = path_points[keep]
            lane_bounds = lane_bounds[keep]

        if len(path_points) < 2:
            return None

        segment_lengths = np.hypot(
            np.diff(path_points[:, 0]),
            np.diff(path_points[:, 1]),
        )
        arc_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        origin_s = start_s
        return path_points, segment_lengths, arc_lengths, origin_s, lane_bounds

    def _sample_path_bounds(self, lane_bounds, arc_lengths, segment_lengths, path_s):
        path_s = float(np.clip(path_s, 0.0, arc_lengths[-1]))
        seg_idx = int(np.searchsorted(arc_lengths, path_s, side="right") - 1)
        seg_idx = min(max(seg_idx, 0), len(segment_lengths) - 1)
        seg_len = max(float(segment_lengths[seg_idx]), 1e-6)
        ratio = float(np.clip((path_s - arc_lengths[seg_idx]) / seg_len, 0.0, 1.0))
        return (1.0 - ratio) * lane_bounds[seg_idx] + ratio * lane_bounds[seg_idx + 1]

    def _obstacle_speed_limit_at(self, path_s, obstacles, cache):
        if obstacles is None or len(obstacles) == 0:
            return float(self._model.target_v)

        speed_limit = float(self._model.target_v)
        static_speed_threshold = float(self._model.obstacle_static_speed_threshold)
        for obstacle in np.asarray(obstacles, dtype=float):
            obstacle = np.asarray(obstacle, dtype=float).flatten()
            if len(obstacle) < 8:
                continue
            obs_speed = float(obstacle[5]) if len(obstacle) > 5 else 0.0
            is_vehicle = float(obstacle[6]) if len(obstacle) > 6 else 1.0
            pass_lateral = float(obstacle[7]) if len(obstacle) > 7 else 0.0
            if is_vehicle < 0.5:
                continue

            obs_s, _, _, obs_lateral = self._project_to_path(
                obstacle[:2],
                cache["path_points"],
                cache["arc_lengths"],
                cache["segment_lengths"],
            )
            half_length = max(float(obstacle[4]) if len(obstacle) > 4 else 2.2, 2.2)
            route_pass_lateral = float(obs_lateral) + pass_lateral
            if obs_speed > static_speed_threshold:
                if abs(route_pass_lateral) <= 1e-6:
                    route_pass_lateral = self._infer_dynamic_route_pass_lateral(
                        cache,
                        float(obs_s),
                        lookahead_distance=self._dynamic_overtake_start_distance + half_length,
                    )
                # Only treat vehicles already occupying the reference lane as
                # primary follow/overtake targets. Vehicles on the adjacent lane
                # should block the target gap, but they should not directly cap
                # the ego reference speed after the ego has started moving out.
                if abs(float(obs_lateral)) > self._dynamic_overtake_same_lane_lateral:
                    continue
                obs_radius = max(
                    float(obstacle[3]) if len(obstacle) > 3 else self._model.obstacle_min_radius,
                    float(self._model.obstacle_min_radius),
                )
                current_clearance = (
                    obs_s
                    - path_s
                    - half_length
                    - float(self._model.ego_footprint_half_length)
                )
                if (
                    current_clearance < -self._dynamic_vehicle_rear_release_clearance
                    or current_clearance > self._dynamic_vehicle_yield_distance
                ):
                    continue

                lane_gap_blocked, lane_front_gap = self._dynamic_target_lane_gap_blocked(
                    obstacles,
                    cache,
                    path_s,
                    float(obs_s),
                    half_length,
                    route_pass_lateral,
                )

                ego_lateral_now = float(cache.get("ego_lateral_now", 0.0))
                reference_lateral = float(cache.get("reference_lateral", ego_lateral_now))
                current_lateral_gap = max(
                    abs(ego_lateral_now - float(obs_lateral)),
                    abs(reference_lateral - float(obs_lateral)),
                )
                release_lateral = max(1.2, 0.42 * abs(route_pass_lateral)) if abs(route_pass_lateral) > 1e-6 else 1.2
                target_lateral_gap = max(abs(pass_lateral), release_lateral, 1e-3)
                lateral_progress = float(
                    np.clip(current_lateral_gap / target_lateral_gap, 0.0, 1.0)
                )
                if current_lateral_gap >= release_lateral and not lane_gap_blocked:
                    continue

                dynamic_cluster = {
                    "rear_s": float(obs_s) - half_length,
                    "front_s": float(obs_s) + half_length,
                    "bypass_start_distance": self._dynamic_overtake_start_distance,
                    "bypass_full_distance": self._dynamic_overtake_full_distance,
                    "return_front_clearance_lengths": self._dynamic_overtake_return_front_clearance_lengths,
                    "return_distance": self._dynamic_overtake_return_distance,
                }
                pass_profile = self._route_bypass_profile(path_s, dynamic_cluster)
                legal_bounds_now = self._sample_path_bounds(
                    cache["lane_bounds"],
                    cache["arc_lengths"],
                    cache["segment_lengths"],
                    path_s,
                )
                if (
                    not lane_gap_blocked
                    and
                    pass_profile >= 0.55
                    and self._lane_bound_supports_offset(
                        route_pass_lateral,
                        legal_bounds_now,
                        required_ratio=0.55,
                    )
                ):
                    continue

                support_window_end = min(
                    obs_s + half_length + self._dynamic_overtake_start_distance,
                    float(cache["arc_lengths"][-1]),
                )
                side_support_ahead = self._lane_bound_supports_offset_ahead(
                    route_pass_lateral,
                    cache,
                    path_s,
                    support_window_end,
                )
                clearance_margin = max(0.0, float(self._dynamic_vehicle_follow_clearance_margin))
                clearance_release_gate = max(
                    self._dynamic_follow_gate_clearance,
                    clearance_margin + 2.0,
                )
                if (
                    not lane_gap_blocked
                    and side_support_ahead
                    and current_clearance > clearance_release_gate
                ):
                    if self._debug_log_enabled and current_clearance < 18.0:
                        self._append_debug_log(
                            "[DYN-SPEED] release_far "
                            f"path_s={path_s:.2f} obs_s={float(obs_s):.2f} "
                            f"clear={current_clearance:.2f} obs_v={obs_speed:.2f} "
                            f"pass_lat={route_pass_lateral:.2f} ref_lat={reference_lateral:.2f} "
                            f"gap={current_lateral_gap:.2f} lat_prog={lateral_progress:.2f}"
                        )
                    continue
                if not lane_gap_blocked and side_support_ahead:
                    if lateral_progress >= 0.24:
                        if self._debug_log_enabled and current_clearance < 18.0:
                            self._append_debug_log(
                                "[DYN-SPEED] release_side "
                                f"path_s={path_s:.2f} obs_s={float(obs_s):.2f} "
                                f"clear={current_clearance:.2f} obs_v={obs_speed:.2f} "
                                f"pass_lat={route_pass_lateral:.2f} ref_lat={reference_lateral:.2f} "
                                f"gap={current_lateral_gap:.2f} lat_prog={lateral_progress:.2f}"
                            )
                        continue

                release_progress = 0.0
                if not lane_gap_blocked:
                    if pass_profile > 0.10:
                        release_progress = max(
                            release_progress,
                            float(np.clip((pass_profile - 0.10) / 0.50, 0.0, 1.0)),
                        )
                    if lateral_progress > 0.12:
                        release_progress = max(
                            release_progress,
                            float(np.clip((lateral_progress - 0.12) / 0.45, 0.0, 1.0)),
                        )

                follow_distance = (
                    self._dynamic_vehicle_follow_buffer
                    + self._dynamic_vehicle_follow_time_headway * max(obs_speed, 0.0)
                )
                safe_follow_distance = follow_distance + clearance_margin
                if current_clearance <= safe_follow_distance:
                    deficit_ratio = float(
                        np.clip(
                            (safe_follow_distance - current_clearance)
                            / max(safe_follow_distance, 1e-3),
                            0.0,
                            1.0,
                        )
                    )
                    slowdown = min(
                        float(self._dynamic_vehicle_follow_max_slowdown),
                        float(self._dynamic_vehicle_follow_speed_deficit_gain) * deficit_ratio,
                    )
                    follow_speed = max(
                        self._dynamic_vehicle_crawl_speed,
                        obs_speed - slowdown,
                    )
                    if release_progress > 0.0:
                        released_speed = follow_speed + release_progress * max(
                            float(self._model.target_v) - follow_speed,
                            0.0,
                        )
                        if self._debug_log_enabled and current_clearance < 18.0:
                            self._append_debug_log(
                                "[DYN-SPEED] follow_release "
                                f"path_s={path_s:.2f} obs_s={float(obs_s):.2f} "
                                f"clear={current_clearance:.2f} follow_d={follow_distance:.2f} "
                                f"safe_d={safe_follow_distance:.2f} "
                                f"obs_v={obs_speed:.2f} pass_prof={pass_profile:.2f} "
                                f"slowdown={slowdown:.2f} rel_prog={release_progress:.2f} "
                                f"limit={released_speed:.2f}"
                            )
                        speed_limit = min(speed_limit, released_speed)
                        continue
                    if self._debug_log_enabled and current_clearance < 18.0:
                        blocker_suffix = (
                            f" lane_gap={lane_front_gap:.2f}"
                            if lane_gap_blocked and np.isfinite(lane_front_gap)
                            else ""
                        )
                        self._append_debug_log(
                            "[DYN-SPEED] follow_hold "
                            f"path_s={path_s:.2f} obs_s={float(obs_s):.2f} "
                            f"clear={current_clearance:.2f} follow_d={follow_distance:.2f} "
                            f"safe_d={safe_follow_distance:.2f} "
                            f"obs_v={obs_speed:.2f} pass_prof={pass_profile:.2f} "
                            f"slowdown={slowdown:.2f} lat_prog={lateral_progress:.2f} "
                            f"limit={follow_speed:.2f}"
                            f"{blocker_suffix}"
                        )
                    speed_limit = min(
                        speed_limit,
                        follow_speed,
                    )
                    continue

                follow_cap_sq = (
                    obs_speed * obs_speed
                    + 2.0
                    * self._dynamic_vehicle_reference_decel
                    * max(current_clearance - safe_follow_distance, 0.0)
                )
                follow_cap = math.sqrt(max(follow_cap_sq, 0.0))
                if not lane_gap_blocked:
                    if pass_profile >= 0.55:
                        follow_cap = max(
                            follow_cap,
                            obs_speed + 0.70 * max(float(self._model.target_v) - obs_speed, 0.0),
                        )
                    elif lateral_progress >= 0.22:
                        follow_cap = max(
                            follow_cap,
                            obs_speed + 0.50 * max(float(self._model.target_v) - obs_speed, 0.0),
                        )
                if release_progress > 0.0:
                    follow_cap = follow_cap + release_progress * max(
                        float(self._model.target_v) - follow_cap,
                        0.0,
                    )
                if self._debug_log_enabled and current_clearance < 18.0:
                    blocker_suffix = (
                        f" lane_gap={lane_front_gap:.2f}"
                        if lane_gap_blocked and np.isfinite(lane_front_gap)
                        else ""
                    )
                    self._append_debug_log(
                        "[DYN-SPEED] cap "
                        f"path_s={path_s:.2f} obs_s={float(obs_s):.2f} "
                        f"clear={current_clearance:.2f} follow_d={follow_distance:.2f} "
                        f"safe_d={safe_follow_distance:.2f} "
                        f"obs_v={obs_speed:.2f} pass_prof={pass_profile:.2f} "
                        f"lat_prog={lateral_progress:.2f} rel_prog={release_progress:.2f} "
                        f"limit={max(obs_speed, follow_cap):.2f}{blocker_suffix}"
                    )
                speed_limit = min(speed_limit, max(obs_speed, follow_cap))
                continue

            if abs(route_pass_lateral) <= 1e-6:
                continue

            entry_s = obs_s - half_length - self._obstacle_speed_entry_distance
            exit_s = obs_s + half_length + self._obstacle_speed_exit_distance
            if entry_s <= path_s <= exit_s:
                speed_limit = min(speed_limit, self._obstacle_bypass_speed)
        return speed_limit

    @staticmethod
    def _smoothstep_value(value):
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _lane_bound_supports_offset(target_lateral, legal_bounds, required_ratio=0.75):
        target_lateral = float(target_lateral)
        if abs(target_lateral) <= 1e-6:
            return True
        left_bound = max(float(legal_bounds[0]), 0.0)
        right_bound = max(float(legal_bounds[1]), 0.0)
        needed = required_ratio * abs(target_lateral)
        if target_lateral > 0.0:
            return left_bound >= needed
        return right_bound >= needed

    def _lane_bound_supports_offset_ahead(
        self,
        target_lateral,
        cache,
        start_s,
        end_s,
        required_ratio=0.65,
        min_support_count=2,
        sample_count=7,
    ):
        if cache is None or abs(float(target_lateral)) <= 1e-6:
            return False

        total_s = float(cache["arc_lengths"][-1])
        start_s = float(np.clip(start_s, 0.0, total_s))
        end_s = float(np.clip(end_s, start_s, total_s))
        if end_s <= start_s + 1e-6:
            return False

        support_count = 0
        for sample_s in np.linspace(start_s, end_s, int(sample_count), dtype=float):
            legal_bounds = self._sample_path_bounds(
                cache["lane_bounds"],
                cache["arc_lengths"],
                cache["segment_lengths"],
                float(sample_s),
            )
            if self._lane_bound_supports_offset(
                target_lateral,
                legal_bounds,
                required_ratio=required_ratio,
            ):
                support_count += 1
                if support_count >= int(min_support_count):
                    return True
        return False

    def _infer_dynamic_route_pass_lateral(
        self,
        cache,
        start_s,
        lookahead_distance=None,
        sample_count=9,
        min_extra_lateral=1.6,
    ):
        if cache is None:
            return 0.0

        total_s = float(cache["arc_lengths"][-1])
        start_s = float(np.clip(start_s, 0.0, total_s))
        lookahead_distance = float(
            self._dynamic_overtake_start_distance if lookahead_distance is None else lookahead_distance
        )
        end_s = float(np.clip(start_s + max(lookahead_distance, 2.0), start_s, total_s))
        if end_s <= start_s + 1e-6:
            return 0.0

        best_lateral = 0.0
        best_extra = 0.0
        for sample_s in np.linspace(start_s, end_s, int(sample_count), dtype=float):
            legal_bounds = self._sample_path_bounds(
                cache["lane_bounds"],
                cache["arc_lengths"],
                cache["segment_lengths"],
                float(sample_s),
            )
            left_bound = max(float(legal_bounds[0]), 0.0)
            right_bound = max(float(legal_bounds[1]), 0.0)
            base_bound = min(left_bound, right_bound)
            extra_left = max(left_bound - base_bound, 0.0)
            extra_right = max(right_bound - base_bound, 0.0)

            if extra_left >= float(min_extra_lateral) and extra_left > best_extra:
                best_extra = extra_left
                best_lateral = extra_left
            if extra_right >= float(min_extra_lateral) and extra_right > best_extra:
                best_extra = extra_right
                best_lateral = -extra_right

        return float(best_lateral)

    def _dynamic_target_lane_gap_blocked(
        self,
        obstacles,
        cache,
        current_s,
        candidate_obs_s,
        candidate_half_length,
        route_pass_lateral,
    ):
        if obstacles is None or len(obstacles) == 0 or cache is None or abs(route_pass_lateral) <= 1e-6:
            return False, float("inf")

        obstacle_rows = np.asarray(obstacles, dtype=float)
        if obstacle_rows.ndim == 1:
            obstacle_rows = obstacle_rows.reshape(1, -1)

        static_speed_threshold = float(self._model.obstacle_static_speed_threshold)
        candidate_rear_s = float(candidate_obs_s) - float(candidate_half_length)
        candidate_front_s = float(candidate_obs_s) + float(candidate_half_length)
        ego_merge_start = max(
            float(current_s) + float(self._model.ego_footprint_half_length),
            candidate_rear_s - float(self._dynamic_overtake_start_distance),
        )
        corridor_start = ego_merge_start - float(self._dynamic_lane_change_rear_gap)
        corridor_end = candidate_front_s + float(self._dynamic_lane_change_front_gap)
        target_lane_center = float(route_pass_lateral)
        best_front_gap = float("inf")
        blocked = False

        for row in obstacle_rows:
            row = np.asarray(row, dtype=float).flatten()
            if len(row) < 8:
                continue

            obs_speed = float(row[5])
            is_vehicle = float(row[6])
            if is_vehicle < 0.5 or obs_speed <= static_speed_threshold:
                continue

            other_s, _, _, other_lateral = self._project_to_path(
                row[:2],
                cache["path_points"],
                cache["arc_lengths"],
                cache["segment_lengths"],
            )
            other_half_length = max(
                float(row[4]) if len(row) > 4 else float(self._model.obstacle_longitudinal_min_half),
                float(self._model.obstacle_longitudinal_min_half),
            )
            if (
                abs(float(other_s) - float(candidate_obs_s)) < 1.0
                and abs(float(other_lateral)) <= self._dynamic_overtake_same_lane_lateral
            ):
                continue

            if abs(float(other_lateral) - target_lane_center) > self._dynamic_lane_change_target_lane_tolerance:
                continue

            other_rear_s = float(other_s) - other_half_length
            other_front_s = float(other_s) + other_half_length

            if other_rear_s >= candidate_front_s:
                best_front_gap = min(best_front_gap, other_rear_s - candidate_front_s)

            if other_rear_s <= corridor_end and other_front_s >= corridor_start:
                blocked = True

        return blocked, best_front_gap

    def _build_route_bypass_clusters(self, obstacles, cache, current_s=None):
        if obstacles is None or len(obstacles) == 0:
            return []

        obstacle_rows = np.asarray(obstacles, dtype=float)
        if obstacle_rows.ndim == 1:
            obstacle_rows = obstacle_rows.reshape(1, -1)

        candidates = []
        static_speed_threshold = float(self._model.obstacle_static_speed_threshold)
        dynamic_overtake_max_speed = max(
            static_speed_threshold,
            float(self._model.target_v) - self._dynamic_overtake_min_speed_delta,
        )
        for row in obstacle_rows:
            row = np.asarray(row, dtype=float).flatten()
            if len(row) < 8:
                continue
            obs_speed = float(row[5])
            is_vehicle = float(row[6])
            pass_lateral = float(row[7])
            if is_vehicle < 0.5 or abs(pass_lateral) <= 1e-6:
                continue

            obs_s, _, _, obs_lateral = self._project_to_path(
                row[:2],
                cache["path_points"],
                cache["arc_lengths"],
                cache["segment_lengths"],
            )
            route_pass_lateral = float(obs_lateral) + pass_lateral
            half_length = max(
                float(row[4]) if len(row) > 4 else float(self._model.obstacle_longitudinal_min_half),
                float(self._model.obstacle_longitudinal_min_half),
            )
            if obs_speed > static_speed_threshold:
                if current_s is not None:
                    current_clearance = (
                        float(obs_s)
                        - float(current_s)
                        - half_length
                        - float(self._model.ego_footprint_half_length)
                    )
                    if (
                        current_clearance < -self._dynamic_vehicle_rear_release_clearance
                        or current_clearance > self._dynamic_vehicle_yield_distance + 15.0
                    ):
                        continue
                if abs(route_pass_lateral) <= 1e-6:
                    route_pass_lateral = self._infer_dynamic_route_pass_lateral(
                        cache,
                        float(obs_s),
                        lookahead_distance=self._dynamic_overtake_start_distance + half_length,
                    )
                if abs(obs_lateral) > self._dynamic_overtake_same_lane_lateral:
                    continue
                if obs_speed >= dynamic_overtake_max_speed:
                    continue
                if abs(route_pass_lateral) <= 1e-6:
                    continue
                lane_gap_blocked, _ = self._dynamic_target_lane_gap_blocked(
                    obstacles,
                    cache,
                    float(current_s) if current_s is not None else float(obs_s),
                    float(obs_s),
                    half_length,
                    route_pass_lateral,
                )
                if lane_gap_blocked:
                    continue
                candidate = {
                    "rear_s": float(obs_s) - half_length,
                    "front_s": float(obs_s) + half_length,
                    "pass_lateral": route_pass_lateral,
                    "bypass_start_distance": self._dynamic_overtake_start_distance,
                    "bypass_full_distance": self._dynamic_overtake_full_distance,
                    "return_front_clearance_lengths": self._dynamic_overtake_return_front_clearance_lengths,
                    "return_distance": self._dynamic_overtake_return_distance,
                    "reference_lateral_limit": max(abs(route_pass_lateral), self._obstacle_reference_lateral_limit),
                }
            else:
                if abs(route_pass_lateral) <= 1e-6:
                    continue
                candidate = {
                    "rear_s": float(obs_s) - half_length,
                    "front_s": float(obs_s) + half_length,
                    "pass_lateral": route_pass_lateral,
                    "bypass_start_distance": float(self._model.obstacle_bypass_start_distance),
                    "bypass_full_distance": float(self._model.obstacle_bypass_full_distance),
                    "return_front_clearance_lengths": float(self._model.obstacle_return_front_clearance_lengths),
                    "return_distance": float(self._model.obstacle_return_distance),
                    "reference_lateral_limit": float(self._obstacle_reference_lateral_limit),
                }
            candidates.append(
                candidate
            )

        candidates.sort(key=lambda item: item["rear_s"])
        clusters = []
        cluster_gap = float(self._model.obstacle_cluster_gap)
        for candidate in candidates:
            merged = False
            for cluster in clusters:
                if (
                    abs(candidate["pass_lateral"]) > 1e-6
                    and abs(cluster["pass_lateral"]) > 1e-6
                    and candidate["pass_lateral"] * cluster["pass_lateral"] <= 0.0
                ):
                    continue
                gap = max(
                    candidate["rear_s"] - cluster["front_s"],
                    cluster["rear_s"] - candidate["front_s"],
                    0.0,
                )
                if gap > cluster_gap:
                    continue

                cluster["rear_s"] = min(cluster["rear_s"], candidate["rear_s"])
                cluster["front_s"] = max(cluster["front_s"], candidate["front_s"])
                cluster["bypass_start_distance"] = max(
                    float(cluster.get("bypass_start_distance", self._model.obstacle_bypass_start_distance)),
                    float(candidate.get("bypass_start_distance", self._model.obstacle_bypass_start_distance)),
                )
                cluster["bypass_full_distance"] = max(
                    float(cluster.get("bypass_full_distance", self._model.obstacle_bypass_full_distance)),
                    float(candidate.get("bypass_full_distance", self._model.obstacle_bypass_full_distance)),
                )
                cluster["return_front_clearance_lengths"] = max(
                    float(cluster.get("return_front_clearance_lengths", self._model.obstacle_return_front_clearance_lengths)),
                    float(candidate.get("return_front_clearance_lengths", self._model.obstacle_return_front_clearance_lengths)),
                )
                cluster["return_distance"] = max(
                    float(cluster.get("return_distance", self._model.obstacle_return_distance)),
                    float(candidate.get("return_distance", self._model.obstacle_return_distance)),
                )
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
                clusters.append(dict(candidate))

        return [cluster for cluster in clusters if abs(cluster["pass_lateral"]) > 1e-6]

    def _route_bypass_profile(self, path_s, cluster):
        ego_half_length = float(self._model.ego_footprint_half_length)
        bypass_start_distance = float(
            cluster.get("bypass_start_distance", self._model.obstacle_bypass_start_distance)
        )
        bypass_full_distance = float(
            cluster.get("bypass_full_distance", self._model.obstacle_bypass_full_distance)
        )
        return_front_clearance_lengths = float(
            cluster.get(
                "return_front_clearance_lengths",
                self._model.obstacle_return_front_clearance_lengths,
            )
        )
        return_distance = float(
            cluster.get("return_distance", self._model.obstacle_return_distance)
        )

        entry_start = cluster["rear_s"] - ego_half_length - bypass_start_distance
        entry_full = cluster["rear_s"] - ego_half_length - bypass_full_distance
        return_start = cluster["front_s"] + max(
            (2.0 * return_front_clearance_lengths - 1.0) * ego_half_length,
            0.0,
        )
        return_end = return_start + return_distance

        if path_s <= entry_start or path_s >= return_end:
            return 0.0
        if path_s < entry_full:
            return self._smoothstep_value(
                (path_s - entry_start) / max(entry_full - entry_start, 1e-6)
            )
        if path_s <= return_start:
            return 1.0
        return 1.0 - self._smoothstep_value(
            (path_s - return_start) / max(return_end - return_start, 1e-6)
        )

    def _route_bypass_offset(self, path_s, clusters, legal_bounds=None):
        if not clusters:
            return 0.0

        best_offset = 0.0
        if legal_bounds is None:
            left_bound = self._lane_bound_inactive
            right_bound = self._lane_bound_inactive
        else:
            left_bound = max(float(legal_bounds[0]), 0.0)
            right_bound = max(float(legal_bounds[1]), 0.0)
        for cluster in clusters:
            profile = self._route_bypass_profile(path_s, cluster)
            if profile <= 0.0:
                continue
            pass_lateral = float(cluster["pass_lateral"])
            side_clearance = min(
                float(self._model.obstacle_side_clearance),
                float(cluster.get("reference_lateral_limit", self._obstacle_reference_lateral_limit)),
            )
            desired_lateral = math.copysign(
                min(abs(pass_lateral), side_clearance),
                pass_lateral,
            )
            if not self._lane_bound_supports_offset(desired_lateral, (left_bound, right_bound)):
                continue
            target_lateral = float(np.clip(desired_lateral, -right_bound, left_bound))
            offset = profile * target_lateral
            if abs(offset) > abs(best_offset):
                best_offset = offset
        return best_offset

    def _sample_shifted_reference_position(self, cache, path_s, bypass_clusters, legal_bounds=None):
        pos, ref_yaw = self._sample_path(
            cache["path_points"],
            cache["arc_lengths"],
            cache["segment_lengths"],
            path_s,
        )
        offset = self._route_bypass_offset(path_s, bypass_clusters, legal_bounds=legal_bounds)
        if abs(offset) <= 1e-6:
            return pos
        left_axis = np.array([-math.sin(ref_yaw), math.cos(ref_yaw)], dtype=float)
        return pos + offset * left_axis

    def _sample_shifted_reference(self, cache, path_s, bypass_clusters, lane_bounds_path):
        path_s = float(np.clip(path_s, 0.0, cache["arc_lengths"][-1]))
        pos = self._sample_shifted_reference_position(
            cache,
            path_s,
            bypass_clusters,
            legal_bounds=self._sample_path_bounds(
                lane_bounds_path,
                cache["arc_lengths"],
                cache["segment_lengths"],
                path_s,
            ),
        )
        s0 = max(path_s - self._reference_yaw_window, 0.0)
        s1 = min(path_s + self._reference_yaw_window, cache["arc_lengths"][-1])
        p0 = self._sample_shifted_reference_position(
            cache,
            s0,
            bypass_clusters,
            legal_bounds=self._sample_path_bounds(
                lane_bounds_path,
                cache["arc_lengths"],
                cache["segment_lengths"],
                s0,
            ),
        )
        p1 = self._sample_shifted_reference_position(
            cache,
            s1,
            bypass_clusters,
            legal_bounds=self._sample_path_bounds(
                lane_bounds_path,
                cache["arc_lengths"],
                cache["segment_lengths"],
                s1,
            ),
        )
        delta = p1 - p0
        if np.linalg.norm(delta) < 1e-6:
            _, ref_yaw = self._sample_path(
                cache["path_points"],
                cache["arc_lengths"],
                cache["segment_lengths"],
                path_s,
            )
            return pos, ref_yaw
        return pos, float(np.arctan2(delta[1], delta[0]))

    def _build_reference(self, x, y, yaw, speed, obstacles=None):
        cache = self._route_cache
        geometry = self._window_geometry_from_route_cache()
        if cache is None or geometry is None:
            ref_rows = np.array([[x, y, yaw, self._model.target_v, 0.0, 0.0] for _ in range(self._model.horizon)], dtype=float)
            lane_bound_rows = np.full((self._model.horizon + 1, 2), self._lane_bound_inactive, dtype=float)
            return ref_rows, lane_bound_rows
        path_points, segment_lengths, arc_lengths, origin_s, lane_bounds_path = geometry
        total_s = float(arc_lengths[-1])
        projection_limit = max(
            self._projection_min_lookahead,
            self._base_min_distance + self._projection_time_lookahead * max(speed, 0.0),
        )
        current_s, _, _, current_lateral = self._project_to_path(
            [x, y],
            path_points,
            arc_lengths,
            segment_lengths,
            min_s=0.0,
            max_s=min(projection_limit, total_s),
            heading_yaw=yaw,
        )
        speed_profile = cache["speed_profile"]
        bypass_clusters = []
        reference_bypass_clusters = []
        if self.enable_obstacle_avoidance:
            local_cache = {
                "path_points": path_points,
                "segment_lengths": segment_lengths,
                "arc_lengths": arc_lengths,
                "lane_bounds": lane_bounds_path,
            }
            bypass_clusters = self._build_route_bypass_clusters(obstacles, local_cache, current_s=current_s)
            if self.enable_obstacle_reference_shaping:
                reference_bypass_clusters = bypass_clusters
        rows = []
        lane_bound_rows = [self._sample_path_bounds(lane_bounds_path, arc_lengths, segment_lengths, current_s)]
        path_s = current_s
        ref_yaw_anchor = float(yaw)
        for _ in range(self._model.horizon):
            legal_bounds = self._sample_path_bounds(
                lane_bounds_path,
                arc_lengths,
                segment_lengths,
                path_s,
            )
            reference_lateral = 0.0
            if reference_bypass_clusters:
                local_cache = {
                    "path_points": path_points,
                    "segment_lengths": segment_lengths,
                    "arc_lengths": arc_lengths,
                }
                reference_lateral = self._route_bypass_offset(
                    path_s,
                    reference_bypass_clusters,
                    legal_bounds=legal_bounds,
                )
                pos, ref_yaw = self._sample_shifted_reference(
                    local_cache,
                    path_s,
                    reference_bypass_clusters,
                    lane_bounds_path,
                )
            else:
                pos, ref_yaw = self._sample_path(path_points, arc_lengths, segment_lengths, path_s)
            ref_yaw = self._unwrap_angle(ref_yaw, anchor=ref_yaw_anchor)
            ref_yaw_anchor = ref_yaw
            profile_s = origin_s + path_s
            ref_speed = self._speed_at_profile_s(speed_profile, profile_s)
            obstacle_cache = {
                "path_points": path_points,
                "segment_lengths": segment_lengths,
                "arc_lengths": arc_lengths,
                "lane_bounds": lane_bounds_path,
                "ego_lateral_now": float(current_lateral),
                "reference_lateral": float(reference_lateral),
            }
            ref_speed = min(ref_speed, self._obstacle_speed_limit_at(path_s, obstacles, obstacle_cache))
            rows.append([pos[0], pos[1], ref_yaw, ref_speed, 0.0, 0.0])
            lane_bound_rows.append(legal_bounds)
            next_profile_s = self._advance_reference_s(profile_s, speed_profile, float(cache["arc_lengths"][-1]))
            path_s = min(max(next_profile_s - origin_s, path_s), total_s)
        self._last_reference_speed = float(rows[0][3])
        return np.asarray(rows, dtype=float), np.asarray(lane_bound_rows, dtype=float)

    def _collect_obstacles(self):
        obstacles = []
        if self._obstacles is not None and len(self._obstacles) > 0:
            for item in np.asarray(self._obstacles, dtype=float):
                item = np.asarray(item, dtype=float).flatten()
                if len(item) < 2:
                    continue
                obstacles.append(
                    [
                        float(item[0]),
                        float(item[1]),
                        float(item[2]) if len(item) > 2 else 0.0,
                        float(item[3]) if len(item) > 3 else 1.2,
                        float(item[4]) if len(item) > 4 else 2.2,
                        float(item[5]) if len(item) > 5 else 0.0,
                        float(item[6]) if len(item) > 6 else 1.0,
                        float(item[7]) if len(item) > 7 else 0.0,
                        float(item[8]) if len(item) > 8 else 0.0,
                        float(item[9]) if len(item) > 9 else 0.0,
                    ]
                )
        if self.enable_obstacle_avoidance and hasattr(self._env, "get_obstacle_states"):
            for item in self._env.get_obstacle_states(
                max_distance=self.obstacle_detection_range,
                include_walkers=False,
                front_only=True,
                rear_release_distance=75.0,
            ):
                if len(item) >= 2:
                    speed = math.hypot(float(item[2]), float(item[3])) if len(item) > 3 else 0.0
                    obstacles.append(
                        [
                            float(item[0]),
                            float(item[1]),
                            float(item[6]) if len(item) > 6 else 0.0,
                            float(item[4]) if len(item) > 4 else 1.2,
                            float(item[5]) if len(item) > 5 else 2.2,
                            speed,
                            float(item[7]) if len(item) > 7 else 1.0,
                            float(item[8]) if len(item) > 8 else 0.0,
                            float(item[2]) if len(item) > 2 else 0.0,
                            float(item[3]) if len(item) > 3 else 0.0,
                        ]
                    )
        if not obstacles:
            return None
        return np.asarray(obstacles, dtype=float)

    def _collect_pedestrians(self):
        if not hasattr(self._env, "get_pedestrian_states"):
            return None
        pedestrians = self._env.get_pedestrian_states()
        if not pedestrians:
            return None
        ped_arr = np.asarray(pedestrians, dtype=float)
        if ped_arr.ndim == 1:
            ped_arr = ped_arr.reshape(1, -1)
        return ped_arr

    def _should_enforce_lane_bounds(self, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return False

        obstacle_rows = np.asarray(obstacles, dtype=float)
        if obstacle_rows.ndim == 1:
            obstacle_rows = obstacle_rows.reshape(1, -1)

        static_speed_threshold = float(getattr(self._model, "obstacle_static_speed_threshold", 0.5))
        for row in obstacle_rows:
            if row.shape[0] < 7:
                continue
            obs_speed = float(row[5]) if row.shape[0] > 5 else 0.0
            is_vehicle = float(row[6]) if row.shape[0] > 6 else 1.0
            if is_vehicle >= 0.5 and obs_speed > static_speed_threshold:
                return True
        return False

    def _route_progress_and_lateral(self, point_xy):
        cache = self._route_cache
        if cache is None:
            return float("inf"), float("inf")
        total_s = float(cache["arc_lengths"][-1])
        projected_s, _, _, lateral = self._project_to_path(
            point_xy,
            cache["path_points"],
            cache["arc_lengths"],
            cache["segment_lengths"],
            min_s=max(self._route_progress_s - 5.0, 0.0),
            max_s=min(self._route_progress_s + self._pedestrian_yield_distance + 25.0, total_s),
        )
        return float(projected_s - self._route_progress_s), float(lateral)

    def _apply_pedestrian_brake(self, a_cmd, ego_speed, predicted_states, pedestrians):
        if pedestrians is None or len(pedestrians) == 0:
            return a_cmd

        ped_arr = np.asarray(pedestrians, dtype=float)
        if ped_arr.ndim == 1:
            ped_arr = ped_arr.reshape(1, -1)

        moving_pedestrian_seen = False
        closest_yield_ahead = float("inf")
        min_path_dist = float("inf")

        for pedestrian in ped_arr:
            if pedestrian.shape[0] < 4:
                continue
            ped_vx = float(pedestrian[2])
            ped_vy = float(pedestrian[3])
            if math.hypot(ped_vx, ped_vy) < 0.05:
                continue

            moving_pedestrian_seen = True
            for dt in np.linspace(0.0, self._pedestrian_crossing_lookahead_time, 17):
                ped_xy = np.array(
                    [
                        float(pedestrian[0]) + ped_vx * dt,
                        float(pedestrian[1]) + ped_vy * dt,
                    ],
                    dtype=float,
                )
                ahead, lateral = self._route_progress_and_lateral(ped_xy)
                if 0.0 < ahead < self._pedestrian_yield_distance and abs(lateral) <= self._pedestrian_path_half_width:
                    closest_yield_ahead = min(closest_yield_ahead, ahead)
                    break

        horizon = min(len(predicted_states), self._model.horizon)
        for step in range(horizon):
            state = np.asarray(predicted_states[step], dtype=float)
            dt = step * self._dt
            for pedestrian in ped_arr:
                if pedestrian.shape[0] < 4:
                    continue
                ped_vx = float(pedestrian[2])
                ped_vy = float(pedestrian[3])
                if math.hypot(ped_vx, ped_vy) < 0.05:
                    continue

                ped_x = float(pedestrian[0]) + ped_vx * dt
                ped_y = float(pedestrian[1]) + ped_vy * dt
                ahead, lateral = self._route_progress_and_lateral([ped_x, ped_y])
                if 0.0 <= ahead <= self._pedestrian_yield_distance and abs(lateral) <= self._pedestrian_release_lateral:
                    min_path_dist = min(min_path_dist, math.hypot(state[0] - ped_x, state[1] - ped_y))

        if not moving_pedestrian_seen:
            return a_cmd

        should_yield = closest_yield_ahead < float("inf")
        if not should_yield and min_path_dist >= self._pedestrian_safe_dist:
            return a_cmd

        target_decel = 0.0
        if should_yield:
            stop_distance = max(closest_yield_ahead - self._pedestrian_stop_buffer, 1.0)
            target_decel = ego_speed * ego_speed / (2.0 * stop_distance)
            target_decel = float(
                np.clip(
                    target_decel + 0.1,
                    self._pedestrian_min_decel,
                    self._pedestrian_strong_decel,
                )
            )

        if min_path_dist < 0.8:
            target_decel = max(target_decel, 4.0)
        elif min_path_dist < 1.1:
            target_decel = max(target_decel, 2.2)
        elif min_path_dist < 1.5:
            target_decel = max(target_decel, 1.4)
        elif min_path_dist < self._pedestrian_safe_dist:
            target_decel = max(target_decel, self._pedestrian_comfort_decel)

        if ego_speed < 0.2 and should_yield:
            target_decel = max(target_decel, self._pedestrian_min_decel)
        if target_decel <= 0.0:
            return a_cmd
        return min(float(a_cmd), -target_decel)

    def _apply_dynamic_vehicle_brake(self, a_cmd, ego_speed, predicted_states, obstacles):
        if obstacles is None or len(obstacles) == 0:
            return a_cmd

        obstacle_rows = np.asarray(obstacles, dtype=float)
        if obstacle_rows.ndim == 1:
            obstacle_rows = obstacle_rows.reshape(1, -1)

        static_speed_threshold = float(getattr(self._model, "obstacle_static_speed_threshold", 0.5))
        horizon = min(len(predicted_states), self._model.horizon + 1)
        if horizon <= 0:
            return a_cmd

        min_predicted_clearance = float("inf")

        for obstacle in obstacle_rows:
            if obstacle.shape[0] < 10:
                continue

            obs_speed = float(obstacle[5]) if obstacle.shape[0] > 5 else 0.0
            is_vehicle = float(obstacle[6]) if obstacle.shape[0] > 6 else 1.0
            if is_vehicle < 0.5 or obs_speed <= static_speed_threshold:
                continue

            current_ahead, obstacle_lateral_now = self._route_progress_and_lateral(obstacle[:2])
            if (
                current_ahead < -self._dynamic_vehicle_rear_release_clearance
                or current_ahead > self._dynamic_vehicle_yield_distance + 20.0
                or abs(obstacle_lateral_now) > self._dynamic_vehicle_watch_lateral
            ):
                continue

            radius = max(
                float(obstacle[3]) if obstacle.shape[0] > 3 else self._model.obstacle_min_radius,
                float(self._model.obstacle_min_radius),
            )
            half_length = max(
                float(obstacle[4]) if obstacle.shape[0] > 4 else self._model.obstacle_longitudinal_min_half,
                float(self._model.obstacle_longitudinal_min_half),
            )
            pass_lateral = abs(float(obstacle[7])) if obstacle.shape[0] > 7 else 0.0
            vx = float(obstacle[8])
            vy = float(obstacle[9])
            local_predicted_clearance = float("inf")

            for step in range(horizon):
                dt = step * self._dt
                obs_xy = np.array(
                    [
                        float(obstacle[0]) + vx * dt,
                        float(obstacle[1]) + vy * dt,
                    ],
                    dtype=float,
                )
                ahead, lateral = self._route_progress_and_lateral(obs_xy)
                ego_state = np.asarray(predicted_states[step], dtype=float)
                ego_ahead, ego_lateral = self._route_progress_and_lateral(ego_state[:2])
                route_clearance = (
                    ahead
                    - ego_ahead
                    - half_length
                    - float(self._model.ego_footprint_half_length)
                )
                if (
                    route_clearance < -self._dynamic_vehicle_rear_release_clearance
                    or route_clearance > self._dynamic_vehicle_yield_distance
                    or abs(lateral) > self._dynamic_vehicle_watch_lateral
                ):
                    continue
                route_lateral_gap = abs(float(ego_lateral) - float(lateral))
                conflict_lateral = max(
                    self._dynamic_vehicle_conflict_lateral,
                    radius + float(self._model.ego_footprint_radius) + 0.1,
                )
                early_release_lateral = max(1.5, 0.55 * pass_lateral) if pass_lateral > 1e-6 else 1.5
                if route_lateral_gap >= early_release_lateral and route_clearance > -1.4:
                    continue
                desired_lateral_gap = max(pass_lateral, conflict_lateral, 1e-3)
                lateral_progress = float(
                    np.clip(route_lateral_gap / desired_lateral_gap, 0.0, 1.0)
                )
                if route_lateral_gap >= conflict_lateral:
                    continue
                if lateral_progress >= 0.32 and route_clearance > -1.4:
                    continue
                center_clearance = (
                    math.hypot(ego_state[0] - obs_xy[0], ego_state[1] - obs_xy[1])
                    - radius
                    - float(self._model.ego_footprint_radius)
                )
                local_predicted_clearance = min(local_predicted_clearance, center_clearance)

            min_predicted_clearance = min(min_predicted_clearance, local_predicted_clearance)

        if self._debug_log_enabled and min_predicted_clearance < 2.0:
            self._append_debug_log(
                "[DYN-BRAKE] "
                f"ego_v={ego_speed:.2f} min_clear={min_predicted_clearance:.2f} a_in={float(a_cmd):.2f}"
            )

        if min_predicted_clearance < 0.80:
            return min(float(a_cmd), -4.8)
        elif min_predicted_clearance < 1.20:
            return min(float(a_cmd), -3.2)
        elif min_predicted_clearance < self._dynamic_vehicle_safe_dist:
            return min(float(a_cmd), -self._dynamic_vehicle_comfort_decel)
        return a_cmd

    def run_step(self, lv=None):
        state, height = self._model.get_state_carla()
        x, y, yaw_deg, vx, vy, omega = state
        yaw = self._unwrap_angle(math.radians(yaw_deg))
        speed = math.sqrt(vx * vx + vy * vy)
        current_state = np.array([x, y, yaw, speed, 0.0, math.radians(omega)], dtype=float)

        self._update_route_progress(x, y, yaw, speed)
        if len(self._waypoints_queue) == 0:
            raise StopIteration("No waypoints to follow")

        obstacles = self._collect_obstacles()
        pedestrians = self._collect_pedestrians()
        ref_traj, lane_bounds = self._build_reference(x, y, yaw, speed, obstacles)
        if self._next_states is None:
            self._next_states = np.zeros((self._model.horizon + 1, self._model.n_states), dtype=float)
        self._next_states[:, 3] = speed
        self._next_states[0] = current_state
        u0 = np.column_stack((self._a_opt, self._delta_opt))
        state = self._model.solve_MPC(
            ref_traj,
            current_state,
            self._next_states,
            u0,
            previous_control=self._last_control,
            obstacles=obstacles,
            lane_bounds=lane_bounds if self._should_enforce_lane_bounds(obstacles) else None,
        )
        self._last_min_obstacle_clearance = getattr(
            self._model,
            "last_min_obstacle_clearance",
            float("inf"),
        )

        if self._draw_planned_trj:
            ca_u.draw_planned_trj(self._world, state[2][:, :2], height + 0.5, color=(255, 0, 0))

        a_sequence = np.asarray(state[0], dtype=float)
        delta_sequence = np.asarray(state[1], dtype=float)
        a_cmd = float(a_sequence[0])
        delta_cmd = float(delta_sequence[0])
        a_cmd = self._apply_dynamic_vehicle_brake(a_cmd, speed, state[2], obstacles)
        a_cmd = self._apply_pedestrian_brake(a_cmd, speed, state[2], pedestrians)
        next_state = self._model.predict(current_state, (a_cmd, delta_cmd))
        self._model.set_state(next_state)
        self._last_control = np.array([a_cmd, delta_cmd], dtype=float)
        self._next_states = np.vstack((state[2][1:], state[2][-1:]))
        self._a_opt = np.concatenate((a_sequence[1:], a_sequence[-1:]))
        self._delta_opt = np.concatenate((delta_sequence[1:], delta_sequence[-1:]))

        _, _, _, lateral_error = self._project_to_path(
            [x, y],
            self._route_cache["path_points"],
            self._route_cache["arc_lengths"],
            self._route_cache["segment_lengths"],
            heading_yaw=yaw,
        )
        if self._debug_log_enabled:
            self._append_debug_log(
                f"[DEBUG] ref_speed={self._last_reference_speed:.3f}, "
                f"a={a_cmd:.3f}, steer={delta_cmd:.3f}"
            )
        return a_cmd, delta_cmd, (next_state, height + 0.05), state[-1] * 1000.0, abs(lateral_error)
