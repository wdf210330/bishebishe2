"""
env.py - CARLA仿真环境封装

功能说明：
    封装CARLA自动驾驶仿真器，提供统一的Python接口
    支持两种显示模式（spec无渲染 / pygame图形界面）
    负责车辆生成、控制信号转换、仿真步进等核心功能

使用方法：
    env = Env(display_method="spec")  # 创建环境（无渲染模式）
    env.reset(spawn_point)            # 生成车辆
    env.step([acc, steer])           # 发送控制指令
    
依赖：
    - carla (CARLA Python API)
    - pygame (可选，仅pygame模式需要)
"""

import atexit
import json
import logging
import os
import random
import math
import datetime
import sys

# 添加父目录到路径（用于导入pgconfig等配置）
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except IndexError:
    pass

import numpy as np
import carla
from pgconfig import *  # 导入配置文件

# pygame为可选依赖，pygame模式需要安装
try:
    import pygame
except ImportError:
    pygame = None


def draw_waypoints(world, waypoints, z=0.5, color=(255, 0, 0), life_time=100.0):
    """
    在CARLA世界中绘制路径点（用于可视化参考轨迹）
    
    参数：
        world: CARLA世界对象
        waypoints: 路径点列表（CARLA.Waypoint对象）
        z: 抬升高度（米），避免点在地表重叠
        color: RGB颜色元组，默认红色
        life_time: 点的存活时间（秒），100秒足够长
    
    示例：
        draw_waypoints(world, route_waypoints, z=0.5, color=(0, 255, 0))  # 绿色路径
    """
    color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    for w in waypoints:
        t = w.transform
        # 在原始位置基础上抬升z米
        begin = t.location + carla.Location(z=z)
        # 绘制一个点
        world.debug.draw_point(begin, size=0.1, color=color, life_time=life_time)


class Env:
    """
    CARLA仿真环境封装类
    
    功能：
        - 连接CARLA仿真器
        - 生成和管理自动驾驶车辆
        - 提供标准化的控制接口
        - 支持两种显示模式
    
    属性：
        client: CARLA客户端
        world: CARLA世界
        map: 地图数据
        ego_vehicle: 被控车辆
        display_method: 显示模式 "spec" 或 "pygame"
    """
    
    def __init__(self, host="localhost", port=2000, dt=0.05, 
                 display_method="spec", steer_ratio=1/0.7):
        """
        初始化仿真环境
        
        参数：
            host: CARLA服务器地址，默认本地
            port: CARLA端口，默认2000
            dt: 仿真步长（秒），默认0.05s=20Hz
            display_method: 显示模式
                - "spec": 无渲染模式，鸟瞰视角，跟踪车辆（省资源，推荐）
                - "pygame": 图形界面，有HUD显示（需要pygame）
            steer_ratio: 方向盘比例系数，默认1/0.7≈1.43
                用于将转向角归一化值转换为实际转向角
        """
        # -------------------------------------------
        # 连接CARLA服务器
        # -------------------------------------------
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)  # 10秒超时，防止卡死
        
        self.world = self.client.get_world()  # 获取当前世界
        self.map = self.world.get_map()        # 获取地图
        self.ego_vehicle = None                # 初始化车辆为空
        self.actor_list = []                   # 管理所有生成的Actor
        
        # 显示相关属性（后续init_display初始化）
        self.display = None
        self.camera_sensor = None
        self.hud = None
        self.clock = None
        
        # 控制参数
        self.steer_ratio = steer_ratio
        self.dt = dt
        
        # -------------------------------------------
        # 配置仿真器设置
        # -------------------------------------------
        self.original_settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,           # 启用渲染
            synchronous_mode=True,             # 同步模式（关键！）
            fixed_delta_seconds=self.dt         # 固定仿真步长
        ))
        
        self.display_method = display_method
        
        # spec模式：创建俯视视角跟随车辆
        if self.display_method == "spec":
            self.spectator = self.world.get_spectator()
        
        # 注册退出清理函数（程序结束时自动调用）
        atexit.register(self.clean)
    
    def init_display(self, size=(1280, 720)):
        """
        初始化pygame图形界面（仅pygame模式需要）
        
        创建窗口、HUD和跟随摄像头
        
        参数：
            size: 窗口分辨率，默认(1280, 720)
        
        异常：
            RuntimeError: pygame未安装时抛出
        """
        if pygame is None:
            raise RuntimeError("pygame is not installed or failed to import.")
        
        # 初始化pygame
        pygame.init()
        pygame.display.set_caption("MPC-Controller")
        
        # 创建窗口
        self.display = pygame.display.set_mode(
            size, 
            pygame.HWSURFACE | pygame.DOUBLEBUF  # 硬件加速 + 双缓冲
        )
        
        # 创建HUD（显示速度、时间等信息）
        self.hud = HUD(size[0], size[1])
        
        # 创建时钟（用于控制帧率）
        self.clock = pygame.time.Clock()
        
        # -------------------------------------------
        # 添加跟随摄像头
        # -------------------------------------------
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")  # RGB摄像头
        camera_bp.set_attribute('image_size_x', str(self.hud.dim[0]))
        camera_bp.set_attribute('image_size_y', str(self.hud.dim[1]))
        
        # 计算摄像头安装位置（车辆后方偏上）
        vehicle = self.ego_vehicle
        bound_x = 0.5 + vehicle.bounding_box.extent.x  # 车辆x方向半长
        bound_y = 0.5 + vehicle.bounding_box.extent.y  # 车辆y方向半宽
        bound_z = 0.5 + vehicle.bounding_box.extent.z  # 车辆z方向半高
        
        spawn_point = carla.Transform(
            carla.Location(x=-3.0 * bound_x, y=0.0 * bound_y, z=3.0 * bound_z),
            carla.Rotation(pitch=8.0)  # 微微俯视
        )
        
        # 挂载摄像头到车辆（SpringArmGhost模式会有轻微晃动效果）
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, 
            spawn_point, 
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost
        )
        
        # 设置摄像头回调（每帧图像的处理函数）
        self.camera_sensor.listen(lambda image: self.camera_callback(image))
    
    def camera_callback(self, image):
        """
        摄像头图像回调函数（每帧图像触发）
        
        将CARLA的图像数据转换为pygame可以显示的格式
        
        参数：
            image: CARLA图像对象
        """
        # 将原始图像数据转换为numpy数组
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))  # RGBA格式
        array = array[:, :, :3]  # 只取RGB，去掉Alpha通道
        array = array[:, :, ::-1]  # BGR -> RGB
        
        # 转换为pygame表面并显示
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))
    
    def reset(self, spawn_point=None):
        """
        重置环境：在指定位置生成车辆
        
        参数：
            spawn_point: CARLA出生点，如果为None则随机选择
        
        效果：
            - 生成一辆Tesla Model 3（CARLA内置车辆）
            - 将其设置为ego_vehicle
            - 放入actor_list管理
        """
        # 如果未指定出生点，随机选择一个
        if spawn_point is None:
            spawn_point = random.choice(self.map.get_spawn_points())
        
        # 获取车辆蓝图（选择特斯拉Model 3）
        blueprint_library = self.world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]  # filter返回列表，取第一个
        
        # 在指定位置生成车辆
        self.ego_vehicle = self.world.spawn_actor(bp, spawn_point)
        self.actor_list.append(self.ego_vehicle)  # 加入管理列表
        
        # 推进一帧（确保车辆完全生成）
        self.world.tick()
    
    def get_cmd(self, action):
        """
        将MPC输出的控制量转换为CARLA车辆控制指令
        
        MPC输出的是物理单位：
            - acc: 加速度（m/s²），正=加速，负=制动
            - steer: 转向角（弧度）
        
        CARLA接受的是归一化值：
            - throttle: [0, 1]，油门开度
            - brake: [0, 1]，制动开度
            - steer: [-1, 1]，方向盘归一化位置
            - reverse: True/False，是否倒车
        
        参数：
            action: [acc_cmd, steer_cmd] MPC控制输出
        
        返回：
            (throttle, steer_norm, brake, reverse) CARLA控制元组
        """
        acc_cmd, steer_cmd = action
        
        # MPC内部用max_wheel_angle=1.22，这里用1/0.7≈1.43恢复
        max_acc = 5.0      # 最大加速度（m/s²）
        max_brake = 3.0   # 最大制动减速度（m/s²）
        max_wheel_angle = 1.22  # 最大车轮转向角（弧度）
        
        # 转向角归一化到[-1, 1]
        steer_norm = float(np.clip(steer_cmd / max_wheel_angle, -1.0, 1.0))
        
        # 处理加减速
        if acc_cmd >= 0:
            # 加速：throttle = acc/max_acc，归一化到[0,1]
            throttle = float(np.clip(acc_cmd / max_acc, 0.0, 1.0))
            brake = 0.0  # 不制动
            reverse = False
        else:
            # 制动：brake = -acc/max_brake，归一化到[0,1]
            throttle = 0.0
            brake = float(np.clip(-acc_cmd / max_brake, 0.0, 1.0))
            reverse = False
        
        return throttle, steer_norm, brake, reverse
    
    def step(self, action):
        """
        执行一步仿真（标准化的环境step接口）
        
        流程：
            1. 转换控制指令
            2. 发送到CARLA执行
            3. 推进仿真一帧
        
        参数：
            action: [acc, steer] MPC控制输出
        """
        # 转换控制指令
        throttle, steer_norm, brake, reverse = self.get_cmd(action)
        
        # 发送控制指令到CARLA
        self.ego_vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer_norm,
            brake=brake,
            reverse=reverse
        ))
        
        # 推进仿真（同步模式下会等待dt秒）
        self.world.tick()
        
        # 更新相机视角（跟随车辆）
        if self.display_method == "spec":
            # spec模式：鸟瞰视角，相机位于车辆正上方30米
            transform = self.ego_vehicle.get_transform()
            self.spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=30),
                carla.Rotation(pitch=-90)  # 俯视
            ))
        elif self.display_method == "pygame" and pygame is not None:
            # pygame模式：使用摄像头的视角
            transform = self.ego_vehicle.get_transform()
    
    def check_quit(self):
        """
        检查是否按下退出键（pygame模式）
        
        在pygame窗口中检测QUIT事件（点击X按钮或Alt+F4）
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    def clean(self):
        """
        清理环境（析构函数，程序退出时自动调用）
        
        恢复原始世界设置，销毁所有生成的Actor（车辆）
        防止CARLA服务器残留状态影响下次运行
        """
        # 恢复原始世界设置
        self.world.apply_settings(self.original_settings)
        
        # 批量销毁所有Actor（车辆、传感器等）
        self.client.apply_batch([
            carla.command.DestroyActor(x) for x in self.actor_list
        ])

    def spawn_obstacle_vehicle(self, spawn_point, blueprint=None, max_retries=5):
        """
        在指定位置生成一辆静止障碍车辆（带碰撞检测重试）
        
        参数：
            spawn_point: carla.Transform，车辆生成位置
            blueprint: 车辆蓝图，None则随机选择
            max_retries: 最大重试次数
        
        返回：
            生成的车辆actor，或 None（生成失败）
        """
        if blueprint is None:
            blueprint = self.world.get_blueprint_library().filter('vehicle.*')[0]
        
        # 尝试多次，找到无碰撞的位置
        for attempt in range(max_retries):
            try:
                # 尝试向上偏移0.5m，避免地面碰撞
                adjusted_loc = carla.Location(
                    x=spawn_point.location.x,
                    y=spawn_point.location.y,
                    z=spawn_point.location.z + 0.5 * attempt
                )
                adjusted_transform = carla.Transform(
                    location=adjusted_loc,
                    rotation=spawn_point.rotation
                )
                obstacle = self.world.spawn_actor(blueprint, adjusted_transform)
                obstacle.set_simulate_physics(False)  # 静止不动
                self.actor_list.append(obstacle)
                return obstacle
            except RuntimeError:
                # 碰撞了，尝试下一个位置
                continue
        
        print(f"[WARN] Failed to spawn obstacle after {max_retries} attempts")
        return None


class HUD:
    """
    Heads-Up Display（平视显示系统）
    
    在pygame窗口上叠加显示车辆状态信息：
        - FPS（服务器和客户端）
        - 当前速度（km/h）
        - 当前位置（x, y）
        - 控制输入（油门、刹车、转向）
        - 附近车辆信息
    
    视觉上类似赛车游戏的仪表盘
    """
    
    def __init__(self, width, height):
        """
        初始化HUD
        
        参数：
            width: 显示区域宽度（像素）
            height: 显示区域高度（像素）
        """
        self.dim = (width, height)
        
        # 设置等宽字体（用于数值显示）
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        
        # 状态变量
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._info_text = []       # 要显示的文本行
        self._show_info = True    # 是否显示信息
        self._server_clock = pygame.time.Clock()  # 服务器时钟
    
    def on_world_tick(self, timestamp):
        """
        CARLA世界Tick回调（每次仿真推进时调用）
        
        更新FPS等时间相关信息
        
        参数：
            timestamp: CARLA时间戳对象
        """
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def tick(self, env, clock):
        """
        更新HUD显示内容（每帧调用）
        
        从车辆获取最新状态，构建要显示的文本列表
        
        参数：
            env: Env环境对象
            clock: pygame时钟对象
        """
        if not self._show_info:
            return
        
        # 获取车辆状态
        t = env.ego_vehicle.get_transform()      # 位置和朝向
        v = env.ego_vehicle.get_velocity()       # 速度向量
        c = env.ego_vehicle.get_control()        # 当前控制输入
        
        # 获取附近车辆列表
        vehicles = env.world.get_actors().filter('vehicle.*')
        
        # -------------------------------------------
        # 构建显示文本（第一部分：基本信息）
        # -------------------------------------------
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(env.ego_vehicle, truncate=20),
            'Map:     % 20s' % env.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
        ]
        
        # 添加位置信息
        self._info_text += [
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
        ]
        
        # -------------------------------------------
        # 添加控制信息（如果有）
        # -------------------------------------------
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),   # 元组会被渲染为进度条
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),                # 布尔值渲染为方块
                ('Hand brake:', c.hand_brake),
            ]
        
        # -------------------------------------------
        # 添加附近车辆信息
        # -------------------------------------------
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            # 计算距离
            distance = lambda l: math.sqrt(
                (l.x - t.location.x)**2 + 
                (l.y - t.location.y)**2 + 
                (l.z - t.location.z)**2
            )
            # 按距离排序
            vehicles = [(distance(x.get_location()), x) 
                       for x in vehicles if x.id != env.ego_vehicle.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:  # 只显示200米内的车辆
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
    
    def toggle_info(self):
        """切换信息显示/隐藏"""
        self._show_info = not self._show_info
    
    def render(self, display):
        """
        渲染HUD到pygame显示表面
        
        参数：
            display: pygame.Surface对象
        """
        if self._show_info:
            # 创建半透明信息面板
            if not hasattr(self, 'info_surface'):
                self.info_surface = pygame.Surface((220, self.dim[1]))
                self.info_surface.set_alpha(40)  # 40/255 透明度
                display.blit(self.info_surface, (0, 0))
            
            # 渲染文本
            v_offset = 4      # 垂直偏移
            bar_h_offset = 100  # 进度条水平起始位置
            bar_width = 106    # 进度条宽度
            
            for item in self._info_text:
                # 检查是否超出显示区域
                if v_offset + 18 > self.dim[1]:
                    break
                
                # 处理进度条类型的数据
                if isinstance(item, list):
                    if len(item) > 1:
                        # 绘制进度条
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) 
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    # 根据数据类型绘制不同样式
                    if isinstance(item[1], bool):
                        # 布尔值：绘制方块
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, 
                            (255, 255, 255), 
                            rect, 
                            0 if item[1] else 1
                        )
                    else:
                        # 数值：绘制进度条
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), 
                            (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        
                        # 计算填充比例
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            # 负数进度条从中间开始
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), 
                                (6, 6)
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), 
                                (f * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]  # 保留标签名
                
                # 绘制文本
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18


def get_actor_display_name(actor, truncate=250):
    """
    获取Actor的显示名称（格式化类型ID）
    
    将CARLA的type_id（如"vehicle.tesla.model3"）
    转换为可读名称（如"Tesla Model3"）
    
    参数：
        actor: CARLA Actor对象
        truncate: 最大字符长度
    
    返回：
        格式化后的车辆名称
    """
    name = ' '.join(
        actor.type_id.replace('_', '.').title().split('.')[1:]
    )
    return (name[:truncate - 1] + '…') if len(name) > truncate else name
