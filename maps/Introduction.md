# maps 场景说明

这里保存的是实验场景配置，不是 CARLA 的 Unreal 地图包。

程序会加载 CARLA 自带的 Town10HD，再按 JSON 配置生成起点、终点、参考路线、静态障碍物、行人或脚本车辆。这样每个算法都能在同一套场景下比较，结果更公平。

## 如何切换场景

CILQR 推荐用环境变量切换，不需要改代码：

```powershell
$env:CILQR_MAP='static_obstacle_avoidance'
```

CMD 中可写成：

```cmd
set CILQR_MAP=static_obstacle_avoidance
```

场景名就是 `maps` 文件夹中的 JSON 文件名，不带 `.json` 后缀。当前可选场景：

- `high_speed_straight_tracking`
- `urban_curve_tracking`
- `static_obstacle_avoidance`
- `occluded_pedestrian_crossing`
- `dynamic_vehicle_interaction`
- `narrow_corridor_passage`
- `double_intersection_u_turn`
- `opportunistic_lane_change_blocked_town05`

## 常用字段

- `carla_map`：调用的 CARLA 地图，目前都使用 `Town10HD`。
- `start_idx` / `end_idx`：自车起点和终点，对应 CARLA spawn point 编号。
- `route_end_index`：先生成完整路线，再把终点提前截断到指定路线点。
- `route_via_indices`：按起点、途经点、终点拼接路线。
- `manual_route_points`：直接给出参考线坐标点，用于全局规划器无法表达的特殊路线，例如路口直接掉头。
- `route_repeat_count`：把手工路线重复几轮。
- `target_v_kmh`：主目标速度，单位 km/h。
- `sample_res`：全局路线采样间距，单位 m。
- `trajectory_xlim` / `trajectory_ylim`：结果图中轨迹子图的显示范围。
- `force_reload_world`：进入场景时是否重载 CARLA 地图，用于清理上一轮残留。
- `draw_route_debug`：是否允许绘制绿色粗参考路线，实际是否绘制还会被 `CILQR_DRAW_GREEN_ROUTE` 覆盖。
- `enable_obstacle_avoidance`：是否启用避障代价。
- `enable_obstacle_corridor_passage`：是否允许左右成对障碍形成的窄通道按中心通行。
- `enable_obstacle_reference_shaping`：是否启用障碍参考轨迹整形。
- `spawn_static_obstacles`：是否生成静态障碍车。
- `static_obstacle_route_groups`：按路线索引批量生成静态障碍车。
- `static_obstacles`：按路线索引单独生成静态障碍车，可配置横向偏移。
- `pedestrian_crossings`：生成脚本横穿行人。
- `scripted_vehicle_obstacles`：生成脚本车辆，例如低速前车、旁车切入车。
- `freeze_after_ego_passed_s`：脚本车辆被试验车超过指定距离后，保留数秒再停在当前位置。

## 1. high_speed_straight_tracking.json：高速直道基准跟踪

配置：`start_idx=92`，`end_idx=86`，`target_v_kmh=40.0`，无障碍物，不启用避障。

这个场景只考察最基础的跟踪能力。路线没有额外障碍，速度比其他场景高，适合检查控制器在简单道路上的上限表现。

重点考察：横向误差是否小，速度能否稳定接近目标值，方向盘是否有不必要摆动，求解时间是否稳定。

## 2. urban_curve_tracking.json：城市连续弯道跟踪

配置：`start_idx=30`，`end_idx=112`，`target_v_kmh=18.0`，无障碍物，不启用避障。

这个场景重点是连续城市弯道。路线包含多次方向变化和路口转弯，不靠障碍物增加难度，而是用道路曲率和路线变化考察跟踪能力。

重点考察：连续转弯时是否贴合参考线，车辆姿态是否平顺，方向盘变化是否自然，路线交叉或弯道密集处是否出现错误跟踪。

## 3. static_obstacle_avoidance.json：静态障碍绕行

配置：`start_idx=92`，`end_idx=86`，`route_end_index=145`，`target_v_kmh=20.0`，启用避障，生成静态障碍车。

障碍物按路线布置在直道、弯道附近和连续障碍段：起步后有单车占道，中段有两车连续占道，后段有三车连续占道，末段还有一个靠近弯道或路口的障碍。

这个场景考察车辆遇到静态占道时，能否提前减速、绕行、保持安全距离，并在绕过后回到合理路线。

重点考察：最小障碍距离，绕行动作是否平顺，连续障碍是否被当成一整段危险区域处理，绕行后是否正常回到原路线。

## 4. occluded_pedestrian_crossing.json：遮挡行人横穿

配置：`start_idx=62`，`end_idx=50`，`target_v_kmh=12.0`，启用避障，生成 5 个横穿行人，并生成 1 辆静态车模拟遮挡。

这个场景不是普通行人横穿，而是包含遮挡风险。部分行人从固定横穿点出现，旁边的静态车模拟视野被挡住后的“突然出现”。

重点考察：车辆是否及时减速或停车，遮挡点附近是否保留足够安全距离，行人与静态车同时存在时是否仍能稳定求解。

## 5. dynamic_vehicle_interaction.json：动态车辆交互

配置：`start_idx=92`，`end_idx=86`，`target_v_kmh=20.0`，启用避障，生成 4 辆脚本车辆。

脚本车辆覆盖三类动态交互：低速前车需要超越，旁车从相邻车道切入，后段车辆让自车在变道或跟车过程中继续处理动态障碍。

这个场景考察算法面对移动车辆时的实时响应能力，不只是绕开静态物体。

重点考察：是否能稳定超过低速车，旁车切入时是否保持安全距离，暂时不能超车时是否能跟车等待，速度曲线是否合理。

## 6. narrow_corridor_passage.json：窄通道通行

配置：`start_idx=92`，`end_idx=86`，`route_end_index=104`，`target_v_kmh=10.0`，启用避障和窄通道通行，生成 12 辆左右成对静态障碍车。

障碍物左右成对布置，形成多个窄通道。车辆需要在起步段、弯道段和长直线段依次穿过这些狭窄空间。

这个场景专门考察可行性和安全余量，难点不是速度，而是在障碍物很近时仍然保持稳定控制。

重点考察：是否碰撞，是否贴近障碍，是否越界，是否出现频繁左右修正，控制器是否直接求解失败。

## 7. double_intersection_u_turn.json：双路口掉头跟踪

配置：`start_idx=101`，`end_idx=101`，使用 `manual_route_points`，`route_repeat_count=2`，`target_v_kmh=10.0`，不生成障碍物，不启用避障。

这个场景考察真实路口掉头逻辑。车辆平时走外侧普通车道，到每个路口掉头前约 90m 快速变道到更靠中央的内侧车道；越过斑马线后立即从内侧车道掉头进入对向外侧车道，不继续驶入路口中央；随后保持外侧车道，直到下一个路口前再重复同样动作。

一圈包含 `(-100,15)` 和 `(100,15)` 两个路口，两圈共完成四次掉头。手工路线用于保证参考线符合真实掉头动作，而不是被全局规划器带到其他路口。

重点考察：掉头前是否在约 90m 处快速并线，掉头是否从内侧车道直接进入对向外侧车道，掉头弧线是否过斑马线就开始，车辆是否卡住、绕圈或长时间方向盘打满。
