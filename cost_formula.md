# Cost Formula

## 2026-05-09 22:35 航向连续化与跟踪代价解释补充
- 修改原因：`solver_add_cost()` 中虽然已经把 yaw 跟踪误差写成 `angle_error = atan2(sin(dyaw), cos(dyaw))`，但若 agent 传入的当前 yaw 与参考 yaw 在 `+pi/-pi` 分支处发生跳变，MPC 仍会在相邻帧或同一 horizon 内看到不连续参考，从而触发左右来回修正。
- 修改后的作用：代价函数形式不变，但 `src/x_v2x_agent.py` 在进入 `J_track` 之前先把当前 yaw 和整个参考轨迹 yaw 展开到连续角度坐标。这样 `J_track` 中的 yaw 误差始终表示真实的小角度偏差，而不是分支切换造成的伪大误差。

### 当前跟踪代价的关键点
```text
J_track(k) =
    [x_err, y_err, yaw_err_wrapped, vx_err, vy_err]^T
    Q
    [x_err, y_err, yaw_err_wrapped, vx_err, vy_err]

yaw_err_wrapped =
    atan2(sin(yaw_state - yaw_ref), cos(yaw_state - yaw_ref))
```

### 当前配套机制
- `state yaw`：由 `run_step()` 对 CARLA 原始 `yaw_deg` 做连续展开后再传入 MPC。
- `reference yaw`：由 `_build_reference()` 以 `ref_yaw_anchor` 为锚点，沿 horizon 逐点连续展开。
- `J_previous_control`：保持 `0.5*(acc_0-acc_prev)^2 + 100*(steer_0-steer_prev)^2`，避免首步转向抖动。
- steering delta bound：首步 `±0.18rad`，后续步 `±0.12rad`，允许必要修正但不再像早期那样被过紧约束卡成左右抢修正。

## 2026-05-09 16:07 弯道二车避障后回正代价修改

- 修改时间：2026-05-09 16:07。
- 修改原因：旁车道距离偏近；弯道二车避障后绕行约束保持过久，压住弯道参考线跟踪，导致后续直道大幅左右摆动。
- 修改后的作用：车辆进入旁车道后保持更大的横向距离；超过障碍车头后更快释放“贴旁车道”和“平行障碍车”的约束，让原参考路线重新接管弯道跟踪。

### 当前总代价函数

```text
J = sum(k = 0..N-1) [
      J_track(k)
    + J_control(k)
    + J_smooth(k)
    + active_obstacle * J_obstacle_safe(k)
    + active_cluster  * J_obstacle_bypass_cluster(k)
] + J_previous_control
```

### 当前关键公式

```text
desired_lateral =
    sign(pass_lateral) * min(abs(pass_lateral), 3.2m)

lateral_profile:
    before obstacle window       -> 0
    entering bypass window       -> smoothstep(0 -> 1)
    before obstacle front        -> 1
    after obstacle front         -> smoothstep(1 -> 0) over 7m

yaw_profile:
    before obstacle window       -> 0
    entering bypass window       -> smoothstep(0 -> 1)
    before obstacle front        -> 1
    after obstacle front         -> smoothstep(1 -> 0) over 2m
```

### 当前参数

```text
obstacle_influence_dist = 1.2
obstacle_safe_dist = 0.45
obstacle_cost_weight = 12.0
obstacle_violation_weight = 500.0
obstacle_side_cost_weight = 28.0
obstacle_side_clearance = 3.2
obstacle_bypass_start_distance = 10.0
obstacle_bypass_full_distance = 1.5
obstacle_return_front_clearance_lengths = 0.5
obstacle_return_distance = 7.0
obstacle_parallel_yaw_weight = 3.0
obstacle_yaw_release_distance = 2.0
obstacle_speed_entry_distance = 22.0
```

## 2026-05-09 15:29 静态避障平滑转向代价修改

- 修改时间：2026-05-09 15:29。
- 修改原因：第一处静态障碍前车辆开始转向的距离基本合适，但转向峰值偏大；根因是绕行目标强迫车辆尽量贴到旁车道中心，横移距离过大。
- 修改后的作用：MPC 仍然自己求解控制量，不加输出滤波；绕行目标改为“达到安全横移即可”，降低第一处避障的转向峰值。

### 当前总代价函数

```text
J = sum(k = 0..N-1) [
      J_track(k)
    + J_control(k)
    + J_smooth(k)
    + active_obstacle * J_obstacle_safe(k)
    + active_cluster  * J_obstacle_bypass_cluster(k)
] + J_previous_control
```

### 各部分作用

- `J_track`：跟踪参考位置、航向和速度。
- `J_control`：限制过大的加速度和转向角。
- `J_smooth`：限制预测序列中相邻控制量突变，当前 `Rd = diag([0.5, 200])`。
- `J_previous_control`：限制本帧第一个控制量相对上一帧实际控制量突变，当前为 `0.5*(acc_0-acc_prev)^2 + 420*(steer_0-steer_prev)^2`。
- `J_obstacle_safe`：保证自车占用圆和障碍物占用圆之间的安全距离。
- `J_obstacle_bypass_cluster`：对合法旁车道的静态障碍段引导车辆绕行。

### 当前静态绕行目标

```text
desired_lateral =
    sign(pass_lateral) * min(abs(pass_lateral), obstacle_side_clearance)

obstacle_side_clearance = 2.8m
obstacle_bypass_start_distance = 10.0m
obstacle_bypass_full_distance = 1.5m
obstacle_return_distance = 12.0m
obstacle_side_cost_weight = 30.0
obstacle_parallel_yaw_weight = 6.0
steer_delta_bound = ±0.035rad
```

## 历史摘要

- 2026-05-09 11:06：静态避障从“只远离障碍物”的对称斥力，改为“安全距离 + 绕行目标 + 通过时航向平行”。
- 2026-05-09 11:34：绕行阶段改为现实三段逻辑：障碍车前约 10m 开始并道，通过后再回原车道。
- 2026-05-09 12:06：固定 `pass_side` 改为 `pass_lateral`；连续障碍合并成一个绕行段；当前公式以下方为准。

## 2026-05-09 12:06 静态避障代价函数修正

- 修改原因：旧代价函数用固定 `pass_side`，符号和 MPC 横向投影坐标不一致，车辆会被拉向黄线侧；连续障碍逐个施加绕行目标，会让车辆在第三处障碍附近同时收到“回正”和“继续绕行”的冲突目标。
- 修改后的作用：绕行目标改为 `pass_lateral`，表示合法旁车道中心相对障碍车道中心的真实横向距离；同向、同侧、距离较近的静态障碍先合并成一个障碍段，再统一生成进入旁车道、平行通过、回原车道的代价；静态障碍窗口内参考速度限制到 12km/h。

### 当前总代价函数

```text
J = sum(k = 0..N-1) [
      J_track(k)
    + J_control(k)
    + J_smooth(k)
    + J_obstacle_safe(k)
    + J_obstacle_bypass_cluster(k)
]
```

- `J_track`：跟踪参考位置、参考航向和参考速度；静态障碍窗口内 `v_ref <= 12km/h`。
- `J_control`：限制过大的加速度和转向角。
- `J_smooth`：限制相邻预测步控制量突变。
- `J_obstacle_safe`：用自车占用圆和障碍物占用圆保持安全距离。
- `J_obstacle_bypass_cluster`：只对有合法旁车道的静态车辆启用，按连续障碍段引导车辆合法绕行。

### 当前静态障碍安全代价

```text
clearance = distance(ego_circle, obstacle_circle)
            - ego_radius
            - obstacle_radius

J_obstacle_safe =
    12.0  * max(0, 1.2  - clearance)^2
  + 500.0 * max(0, 0.35 - clearance)^2
```

### 当前连续障碍绕行代价

```text
rel_long = dot(ego_xy - cluster_origin_xy, cluster_forward)
rel_lat  = dot(ego_xy - cluster_origin_xy, cluster_left)

entry_start = cluster_rear - ego_half_length - 10.0m
entry_full  = cluster_rear - ego_half_length - 3.5m
return_start = cluster_front + (2 * 1.5 - 1) * ego_half_length
return_end   = return_start + 8.0m

profile = 0                     before entry_start
profile = smoothstep(0 -> 1)     entry_start to entry_full
profile = 1                     parallel pass
profile = smoothstep(1 -> 0)     return_start to return_end

target_lat = sign(pass_lateral) * max(abs(pass_lateral), 4.0)

J_obstacle_bypass_cluster =
    profile * 60.0 * (rel_lat - target_lat)^2
  + profile * 14.0 * angle_error(ego_yaw, cluster_yaw)^2
```

### 当前关键参数

- `obstacle_influence_dist = 1.2`
- `obstacle_safe_dist = 0.35`
- `obstacle_cost_weight = 12.0`
- `obstacle_violation_weight = 500.0`
- `obstacle_side_cost_weight = 60.0`
- `obstacle_side_clearance = 4.0`
- `obstacle_bypass_start_distance = 10.0`
- `obstacle_bypass_full_distance = 3.5`
- `obstacle_return_front_clearance_lengths = 1.5`
- `obstacle_return_distance = 8.0`
- `obstacle_cluster_gap = 40.0`
- `obstacle_parallel_yaw_weight = 14.0`
- `pass_lateral`：合法旁车道中心相对障碍车道中心的有符号横向距离，0 表示没有合法旁车道，不启用绕行横向目标。
## 2026-05-09 13:18 求解器参数化实现记录

- 修改原因：原来每帧把障碍物直接写进 CasADi 表达式，导致 IPOPT 求解器每帧重建，仿真卡顿。
- 修改后的作用：代价函数数学形式不变，但障碍物和连续绕行段改为固定参数槽输入；求解器只初始化一次，每帧只更新参数。`MPC_MAX_OBSTACLES` 默认 8，`MPC_MAX_BYPASS_CLUSTERS` 默认 8，超过上限直接报错。
- 当前总代价函数仍为：
```text
J = sum(k = 0..N-1) [
      J_track(k)
    + J_control(k)
    + J_smooth(k)
    + active_obstacle * J_obstacle_safe(k)
    + active_cluster  * J_obstacle_bypass_cluster(k)
]
```
- `active_obstacle`：障碍物参数槽是否启用，未启用时该槽代价为 0。
- `active_cluster`：连续绕行段参数槽是否启用，未启用时该槽代价为 0。
## 2026-05-10 00:12 车道边界软约束与失败帧兜底

- 修改原因：
  - `dynamic_vehicle_interaction` 需要显式道路边界约束来抑制越过黄实线的非法超车；
  - `occluded_pedestrian_crossing` 起步异常刹车的根因是 IPOPT 在默认 `30` 次迭代内未收敛，而不是行人制动逻辑本身。
- 修改后的作用：
  - 对前方存在动态车辆障碍的场景，MPC 会对“相对参考线横向偏移超出合法车道包络”的状态增加二次惩罚；
  - 若当前帧 `ipopt.max_iter=30` 未收敛，则自动以更高迭代上限重求同一帧，避免把失败帧控制直接发给车辆。

### 当前总代价函数
```text
J = sum(k = 0..N-1) [
      J_track(k)
    + J_control(k)
    + J_smooth(k)
    + J_lane_boundary(k)
    + active_obstacle * J_obstacle_safe(k)
    + active_cluster  * J_obstacle_bypass_cluster(k)
] + J_lane_boundary(N) + J_previous_control
```

### 当前车道边界软约束
```text
route_lateral(k) =
    dot(
        [x_k - x_ref_k, y_k - y_ref_k],
        [-sin(yaw_ref_k), cos(yaw_ref_k)]
    )

legal_interval(k) = [-right_bound_k, left_bound_k]

J_lane_boundary(k) =
    900.0 * max(0, route_lateral(k) - left_bound_k)^2
  + 900.0 * max(0, -right_bound_k - route_lateral(k))^2
```

- `lane_boundary_margin = 0.20`
- `lane_boundary_weight = 900.0`
- `lane_bounds` 仅在检测到前方动态车辆障碍时送入 MPC；
- 未激活时使用 `lane_bound_inactive = 1e6`，对应 `J_lane_boundary = 0`。

### 当前静态参考绕行约束
```text
target_lateral_ref =
    sign(pass_lateral) * min(abs(pass_lateral), 2.4m)
```

- 这条 `2.4m` 上限只用于 `src/x_v2x_agent.py` 的静态参考轨迹塑形；
- 求解器里的静态障碍软代价目前只保留 `J_obstacle_safe`，不再额外施加横向簇牵引，避免与遮挡行人场景冲突。

### 当前失败帧兜底
```text
primary solve:
    ipopt.max_iter = 30

if primary_status != success:
    fallback solve:
        ipopt.max_iter = 120
```

- 本轮在 `occluded_pedestrian_crossing` 首帧验证中：
  - 主求解器返回 `Maximum_Iterations_Exceeded`
  - fallback 求解器返回 `Solve_Succeeded`
  - 首帧控制从异常刹车恢复为正常起步加速

## 2026-05-10 01:42 动态障碍时变参数补充

### 新增障碍物参数块
```text
Obstacle(j) =
    [active, obs_x0, obs_y0, obs_yaw, obs_radius, obs_half_length, obs_vx, obs_vy]
```

### 动态障碍物在预测域内的位置
```text
obs_x(j, k) = obs_x0(j) + obs_vx(j) * k * dt
obs_y(j, k) = obs_y0(j) + obs_vy(j) * k * dt
```

### 更新后的软避障代价
```text
J_obstacle_safe(k) =
    sum_over_obstacles sum_over_ego_centers sum_over_obs_centers [
        12.0  * max(0, 1.2  - clearance(j, k))^2
      + 500.0 * max(0, 0.45 - clearance(j, k))^2
    ] / (k + 1)
```

- 说明：
  - 这次没有再额外提高障碍物斥力权重，核心变化是把脚本动态车从“当前帧静态位置”改成“随预测步前推的位置”；
  - `dynamic_vehicle_interaction` 的改善主要来自时空预测一致，而不是简单加大惩罚。
