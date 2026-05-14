"""
CARLA simulation environment wrapper.
"""

import math
import random

import carla
import numpy as np
import pygame


class Env:
    def __init__(
        self,
        display_method="spec",
        dt=0.05,
        max_steer_rad=0.7,
        no_rendering=False,
        town_name=None,
        client_timeout=120.0,
        force_reload_world=False,
    ):
        self.display_method = display_method
        self.dt = dt
        self.max_steer_rad = max(max_steer_rad, 1e-3)
        self.no_rendering = no_rendering
        self.town_name = town_name
        self.throttle_acc_scale = 8.0
        self.brake_acc_scale = 20.0
        self.max_throttle = 0.50
        self.max_brake = 0.25
        self.launch_min_throttle = 0.14
        self.launch_speed_threshold = 1.0
        self.launch_max_throttle = 0.24
        self.launch_release_speed = 2.5
        self.accel_feedback_kp = 0.10
        self.accel_feedback_ki = 0.015
        self.accel_integral_limit = 3.0
        self.brake_accel_feedback_kp = 0.005
        self.longitudinal_accel_filter_alpha = 0.35
        self.stopped_speed_threshold = 0.5
        self.max_longitudinal_decel = 5.0
        self._last_longitudinal_speed = None
        self._filtered_longitudinal_accel = 0.0
        self._accel_error_integral = 0.0
        self.client_timeout = client_timeout

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(client_timeout)
        self.world = self.client.get_world()
        if town_name is not None:
            current_map_name = self.world.get_map().name
            if force_reload_world or not current_map_name.endswith(town_name):
                self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()

        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=dt,
            no_rendering_mode=no_rendering,
        )
        self.world.apply_settings(settings)

        self.display = None
        self.hud = None
        self.clock = None

        self.spectator = self.world.get_spectator()
        self.ego_vehicle = None

        self.pedestrians = []
        self.pedestrian_controllers = []
        self.obstacle_vehicles = []
        self.scripted_actor_velocities = {}
        self.last_vehicle_pos = None
        self.last_vehicle_control = None

    def _reset_longitudinal_controller(self):
        self._last_longitudinal_speed = None
        self._filtered_longitudinal_accel = 0.0
        self._accel_error_integral = 0.0

    def _ego_speed(self):
        if self.ego_vehicle is None:
            return 0.0
        velocity = self.ego_vehicle.get_velocity()
        return math.sqrt(
            velocity.x * velocity.x
            + velocity.y * velocity.y
            + velocity.z * velocity.z
        )

    def get_ego_longitudinal_acceleration(self):
        if self.ego_vehicle is None:
            return 0.0
        velocity = self.ego_vehicle.get_velocity()
        speed_xy = math.hypot(velocity.x, velocity.y)
        if speed_xy > self.stopped_speed_threshold:
            dir_x = velocity.x / speed_xy
            dir_y = velocity.y / speed_xy
        else:
            yaw = math.radians(self.ego_vehicle.get_transform().rotation.yaw)
            dir_x = math.cos(yaw)
            dir_y = math.sin(yaw)
        acceleration = self.ego_vehicle.get_acceleration()
        return acceleration.x * dir_x + acceleration.y * dir_y

    def _longitudinal_feedback_state(self):
        speed = self._ego_speed()
        measured_acc = self.get_ego_longitudinal_acceleration()
        if not np.isfinite(measured_acc):
            if self._last_longitudinal_speed is None:
                measured_acc = 0.0
            else:
                measured_acc = (speed - self._last_longitudinal_speed) / max(self.dt, 1e-6)
        self._last_longitudinal_speed = speed

        alpha = float(np.clip(self.longitudinal_accel_filter_alpha, 0.0, 1.0))
        self._filtered_longitudinal_accel = (
            alpha * measured_acc + (1.0 - alpha) * self._filtered_longitudinal_accel
        )
        return speed, self._filtered_longitudinal_accel

    def spawn_pedestrian(self, location, speed=1.0, target_location=None, yaw=0.0):
        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")[0]
        walker_bp.set_attribute("speed", str(speed))

        spawn_z = float(location[2]) if len(location) > 2 else 1.0
        transform = carla.Transform(
            carla.Location(x=float(location[0]), y=float(location[1]), z=spawn_z),
            carla.Rotation(yaw=float(yaw)),
        )

        walker = self.world.spawn_actor(walker_bp, transform)

        controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

        controller.start()
        if target_location is None:
            target = carla.Location(
                x=location[0] + random.uniform(-5, 5),
                y=location[1] + random.uniform(-5, 5),
                z=0,
            )
        else:
            target_z = float(target_location[2]) if len(target_location) > 2 else 0.0
            target = carla.Location(
                x=float(target_location[0]),
                y=float(target_location[1]),
                z=target_z,
            )

        controller.go_to_location(target)
        controller.set_max_speed(speed)

        self.pedestrians.append(walker)
        self.pedestrian_controllers.append(controller)
        return walker, controller

    def spawn_scripted_pedestrian(self, transform, speed=1.0, max_retries=5):
        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")[0]
        if walker_bp.has_attribute("speed"):
            walker_bp.set_attribute("speed", str(speed))

        for attempt in range(max_retries):
            loc = carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + 0.2 * attempt,
            )
            candidate = carla.Transform(loc, transform.rotation)
            try:
                walker = self.world.try_spawn_actor(walker_bp, candidate)
                if walker is None:
                    continue
                walker.set_simulate_physics(False)
                self.pedestrians.append(walker)
                self.world.tick()
                return walker
            except RuntimeError:
                continue

        return None

    def spawn_pedestrians_on_path(
        self, path_points, num_pedestrians=3, radius=3.0, speed_range=(0.5, 1.5)
    ):
        spawned = []

        for _ in range(num_pedestrians):
            path_idx = random.randint(5, len(path_points) - 10)
            base_x, base_y = path_points[path_idx]

            x = base_x + random.uniform(-radius, radius)
            y = base_y + random.uniform(-radius, radius)
            speed = random.uniform(*speed_range)

            walker, controller = self.spawn_pedestrian([x, y], speed)
            spawned.append(
                {
                    "walker": walker,
                    "controller": controller,
                    "spawn_path_idx": path_idx,
                    "base_x": base_x,
                    "base_y": base_y,
                }
            )

        return spawned

    def get_pedestrian_states(self):
        states = []
        for ped in self.pedestrians:
            try:
                transform = ped.get_transform()
                velocity = ped.get_velocity()
                scripted_velocity = self.scripted_actor_velocities.get(ped.id)
                if scripted_velocity is not None:
                    velocity = carla.Vector3D(
                        x=float(scripted_velocity[0]),
                        y=float(scripted_velocity[1]),
                        z=0.0,
                    )
                states.append(
                    [
                        transform.location.x,
                        transform.location.y,
                        velocity.x,
                        velocity.y,
                    ]
                )
            except Exception:
                pass
        return states

    def get_obstacle_states(
        self,
        max_distance=28.0,
        include_walkers=True,
        front_only=True,
        lateral_limit=12.0,
        rear_release_distance=30.0,
    ):
        """Return nearby obstacle states.

        Each row is [x, y, vx, vy, radius, half_length, yaw_rad, is_vehicle, pass_lateral].
        """
        states = []
        if self.ego_vehicle is None:
            return states

        ego_transform = self.ego_vehicle.get_transform()
        ego_loc = self.ego_vehicle.get_location()
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        def is_relevant(location):
            dx = location.x - ego_loc.x
            dy = location.y - ego_loc.y
            rel_x = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
            rel_y = -dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
            if front_only and rel_x < -rear_release_distance:
                return False
            if abs(rel_y) > lateral_limit:
                return False
            return True

        def angle_diff_rad(a, b):
            return math.atan2(math.sin(a - b), math.cos(a - b))

        def is_same_direction(source_wp, target_wp):
            if target_wp is None or target_wp.lane_type != carla.LaneType.Driving:
                return False
            source_yaw = math.radians(source_wp.transform.rotation.yaw)
            target_yaw = math.radians(target_wp.transform.rotation.yaw)
            return abs(angle_diff_rad(target_yaw, source_yaw)) < math.pi / 2.0

        def marking_allows(marking, direction):
            if marking is None:
                return False
            return bool(marking.lane_change & direction)

        def legal_pass_lateral_for_actor(actor):
            loc = actor.get_location()
            waypoint = self.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                return 0.0

            yaw_rad = math.radians(waypoint.transform.rotation.yaw)
            left_axis = (-math.sin(yaw_rad), math.cos(yaw_rad))
            source_loc = waypoint.transform.location

            def signed_lateral_to(target_wp):
                target_loc = target_wp.transform.location
                dx = target_loc.x - source_loc.x
                dy = target_loc.y - source_loc.y
                return dx * left_axis[0] + dy * left_axis[1]

            right_lane = waypoint.get_right_lane()
            if (
                marking_allows(waypoint.right_lane_marking, carla.LaneChange.Right)
                and is_same_direction(waypoint, right_lane)
            ):
                return signed_lateral_to(right_lane)

            left_lane = waypoint.get_left_lane()
            if (
                marking_allows(waypoint.left_lane_marking, carla.LaneChange.Left)
                and is_same_direction(waypoint, left_lane)
            ):
                return signed_lateral_to(left_lane)

            return 0.0

        for vehicle in self.world.get_actors().filter("vehicle.*"):
            if vehicle.id == self.ego_vehicle.id:
                continue
            try:
                loc = vehicle.get_location()
                if ego_loc.distance(loc) > max_distance:
                    continue
                if not is_relevant(loc):
                    continue
                vel = vehicle.get_velocity()
                scripted_velocity = self.scripted_actor_velocities.get(vehicle.id)
                if scripted_velocity is not None:
                    vel = carla.Vector3D(
                        x=float(scripted_velocity[0]),
                        y=float(scripted_velocity[1]),
                        z=0.0,
                    )
                extent = vehicle.bounding_box.extent
                # The CILQR state is treated as a point-mass center. Using the
                # full vehicle half-length as a circular obstacle radius is too
                # conservative and can make the optimizer orbit the obstacle.
                radius = max(extent.y, 0.8) + 0.4
                half_length = max(extent.x, radius)
                waypoint = self.map.get_waypoint(
                    loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                yaw_rad = (
                    math.radians(waypoint.transform.rotation.yaw)
                    if waypoint is not None
                    else math.radians(vehicle.get_transform().rotation.yaw)
                )
                pass_lateral = legal_pass_lateral_for_actor(vehicle)
                states.append([loc.x, loc.y, vel.x, vel.y, radius, half_length, yaw_rad, 1.0, pass_lateral])
            except Exception:
                pass

        if include_walkers:
            for walker in self.world.get_actors().filter("walker.*"):
                try:
                    loc = walker.get_location()
                    if ego_loc.distance(loc) > max_distance:
                        continue
                    if not is_relevant(loc):
                        continue
                    vel = walker.get_velocity()
                    scripted_velocity = self.scripted_actor_velocities.get(walker.id)
                    if scripted_velocity is not None:
                        vel = carla.Vector3D(
                            x=float(scripted_velocity[0]),
                            y=float(scripted_velocity[1]),
                            z=0.0,
                        )
                    extent = walker.bounding_box.extent
                    radius = max(extent.x, extent.y, 0.4) + 0.6
                    yaw_rad = math.radians(walker.get_transform().rotation.yaw)
                    states.append([loc.x, loc.y, vel.x, vel.y, radius, radius, yaw_rad, 0.0, 0.0])
                except Exception:
                    pass

        return states

    def spawn_obstacle_vehicle(self, transform, blueprint=None, max_retries=5):
        """Spawn a static obstacle vehicle for CILQR avoidance tests.

        Comment out the call site in test_main.py or set
        spawn_static_obstacle_demo = False to disable the test obstacle without
        affecting normal simulation.
        """
        if blueprint is None:
            blueprints = self.world.get_blueprint_library().filter("vehicle.*")
            blueprint = blueprints[0]

        for attempt in range(max_retries):
            loc = carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + 0.25 * attempt,
            )
            candidate = carla.Transform(loc, transform.rotation)
            try:
                obstacle = self.world.try_spawn_actor(blueprint, candidate)
                if obstacle is None:
                    continue
                obstacle.set_simulate_physics(False)
                self.obstacle_vehicles.append(obstacle)
                self.world.tick()
                return obstacle
            except RuntimeError:
                continue

        return None

    def set_scripted_actor_velocity(self, actor, velocity):
        self.scripted_actor_velocities[actor.id] = (
            float(velocity[0]),
            float(velocity[1]),
        )

    def destroy_obstacle_vehicles(self):
        """Destroy only the test obstacles spawned by spawn_obstacle_vehicle."""
        for obstacle in list(self.obstacle_vehicles):
            try:
                self.scripted_actor_velocities.pop(obstacle.id, None)
                obstacle.destroy()
            except Exception:
                pass
        self.obstacle_vehicles = []

    def update_pedestrians_random(self):
        for controller in self.pedestrian_controllers:
            try:
                if random.random() < 0.02:
                    if self.last_vehicle_pos is not None:
                        vx, vy = self.last_vehicle_pos
                        target_x = vx + random.uniform(-20, 20)
                        target_y = vy + random.uniform(-20, 20)
                    else:
                        target_x = random.uniform(-50, 50)
                        target_y = random.uniform(-50, 50)

                    controller.go_to_location(carla.Location(x=target_x, y=target_y, z=0))
            except Exception:
                pass

    def set_vehicle_pos_for_pedestrians(self, x, y):
        self.last_vehicle_pos = (x, y)

    def clean(self):
        actors = self.world.get_actors()

        for controller in list(actors.filter("controller.*")):
            try:
                controller.stop()
                controller.destroy()
            except Exception:
                pass

        for pattern in ("sensor.*", "vehicle.*", "walker.*"):
            for actor in list(actors.filter(pattern)):
                try:
                    actor.destroy()
                except Exception:
                    pass

        for _ in range(3):
            try:
                self.world.tick()
            except Exception:
                break

        self.ego_vehicle = None
        self.pedestrians = []
        self.pedestrian_controllers = []
        self.obstacle_vehicles = []
        self.scripted_actor_velocities = {}
        self._reset_longitudinal_controller()

    def reset(self, spawn_point=None):
        if spawn_point is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[0]

        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
                self.world.tick()
            except Exception:
                pass

        blueprints = self.world.get_blueprint_library().filter("vehicle.tesla.model3")
        if not blueprints:
            blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        blueprint = blueprints[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        self.ego_vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        if self.ego_vehicle is None:
            raise RuntimeError(
                "Failed to spawn ego vehicle. The selected spawn point may be occupied."
            )

        self.ego_vehicle.set_simulate_physics(True)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        if self.display_method == "spec" and not self.no_rendering:
            self._update_spectator()

        for _ in range(10):
            self.world.tick()

        self._reset_longitudinal_controller()
        return self.ego_vehicle

    def _update_spectator(self):
        if self.ego_vehicle and self.spectator:
            transform = self.ego_vehicle.get_transform()
            camera_location = carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=40.0,
            )
            camera_rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
            self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def step(self, control):
        if len(control) >= 2:
            acc, steer = control[0], control[1]
            speed, measured_acc = self._longitudinal_feedback_state()

            if acc >= -0.05:
                acc_error = float(acc) - measured_acc
                self._accel_error_integral = float(
                    np.clip(
                        self._accel_error_integral + acc_error * self.dt,
                        -self.accel_integral_limit,
                        self.accel_integral_limit,
                    )
                )
                throttle = max(float(acc), 0.0) / self.throttle_acc_scale
                throttle += self.accel_feedback_kp * acc_error
                throttle += self.accel_feedback_ki * self._accel_error_integral
                if acc > 0.05 and speed < self.launch_speed_threshold:
                    launch_ratio = 1.0 - speed / max(self.launch_speed_threshold, 1e-6)
                    throttle = max(throttle, self.launch_min_throttle * launch_ratio)
                if speed < self.launch_release_speed:
                    release_ratio = speed / max(self.launch_release_speed, 1e-6)
                    launch_cap = self.launch_max_throttle + (
                        self.max_throttle - self.launch_max_throttle
                    ) * float(np.clip(release_ratio, 0.0, 1.0))
                    throttle = min(throttle, launch_cap)
                throttle = float(np.clip(throttle, 0.0, self.max_throttle))
                brake = 0.0
            else:
                self._accel_error_integral = 0.0
                throttle = 0.0
                desired_decel = min(-float(acc), self.max_longitudinal_decel)
                measured_decel = max(-measured_acc, 0.0)
                brake = desired_decel / self.brake_acc_scale
                brake += self.brake_accel_feedback_kp * (desired_decel - measured_decel)
                brake = float(np.clip(brake, 0.0, self.max_brake))

            steer = float(np.clip(steer, -self.max_steer_rad, self.max_steer_rad))
            carla_steer = float(np.clip(steer / self.max_steer_rad, -1.0, 1.0))

            vehicle_control = carla.VehicleControl()
            vehicle_control.throttle = throttle
            vehicle_control.brake = brake
            vehicle_control.steer = carla_steer
            vehicle_control.hand_brake = False
            vehicle_control.reverse = False
            vehicle_control.manual_gear_shift = False
            self.last_vehicle_control = vehicle_control
            self.ego_vehicle.apply_control(vehicle_control)

        if self.display_method == "spec" and not self.no_rendering:
            self._update_spectator()

        self.world.tick()

        return self.last_vehicle_control

    def init_display(self):
        pygame.init()
        pygame.font.init()

        width, height = 1280, 720
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CILQR Controller")

        self.clock = pygame.time.Clock()
        self.hud = HUD(width, height)

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False


class HUD:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)

        self.simulation_time = 0
        self.vehicle_speed = 0
        self.control_throttle = 0
        self.control_steer = 0
        self.control_brake = 0

    def tick(self, env, clock):
        self.simulation_time += env.dt

        vel = env.ego_vehicle.get_velocity()
        self.vehicle_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2)

        control = env.ego_vehicle.get_control()
        self.control_throttle = control.throttle
        self.control_brake = control.brake
        self.control_steer = control.steer

    def render(self, display):
        display.fill((0, 0, 0))

        v_info = self.font.render(
            f"Speed: {self.vehicle_speed:.1f} km/h", True, (255, 255, 255)
        )
        display.blit(v_info, (10, 10))

        c_info = self.font.render(
            (
                f"Throttle: {self.control_throttle:.2f}  "
                f"Brake: {self.control_brake:.2f}  "
                f"Steer: {self.control_steer:.2f}"
            ),
            True,
            (255, 255, 255),
        )
        display.blit(c_info, (10, 40))

        t_info = self.font.render(f"Time: {self.simulation_time:.1f}s", True, (255, 255, 255))
        display.blit(t_info, (10, 70))


def draw_waypoints(
    world,
    waypoints,
    z=0.5,
    color=(255, 0, 0),
    life_time=0.5,
    draw_lines=False,
    persistent_lines=True,
):
    debug_color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    last_point = None
    for waypoint in waypoints:
        transform = waypoint.transform
        begin = transform.location + carla.Location(z=z)
        world.debug.draw_point(
            begin,
            size=0.1,
            color=debug_color,
            life_time=life_time,
            persistent_lines=persistent_lines,
        )
        if draw_lines and last_point is not None:
            world.debug.draw_line(
                last_point,
                begin,
                thickness=0.08,
                color=debug_color,
                life_time=life_time,
                persistent_lines=persistent_lines,
            )
        last_point = begin
