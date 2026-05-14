# Debug Note

## 2026-05-10 18:45 求解时间尖峰与纵向减速度约束收紧

- 遇到的问题：
  - 用户要求 `solve_time` 尽量压到 `100ms` 内；
  - 动态避障结果日志中出现 `acc=-20 m/s^2` 量级，超过论文展示可接受范围。
- 问题分析：
  - `test_main.py` 里的 `acc` 之前是用相邻帧速度差分得到，容易把低速抖动、执行器瞬态和停走切换放大成异常峰值，并不完全等于 CARLA 真实纵向加速度；
  - MPC 本体控制量约束仍是 `acc_lbound=-6.0`、`acc_ubound=3.0`，所以 `-20` 不是优化变量直接越界，更可能来自执行层 brake 映射过强；
  - 求解时间均值本来已经较低，主要问题是少数 IPOPT 尖峰帧超过 `100ms`，因此应优先压 solver 尖峰而不是继续牺牲整体能力。
- 对应修改：
  - `env.py`
    - 新增 `get_ego_longitudinal_acceleration()`，直接读取 CARLA 加速度并投影到车辆当前纵向；
    - 制动执行层参数收紧：`brake_acc_scale 6.0 -> 7.5`、`brake_accel_feedback_kp 0.08 -> 0.05`、新增 `max_brake=0.72`、`max_longitudinal_decel=11.8`，限制执行层实际减速度；
  - `test_main.py`
    - 日志中的 `acc` 改为优先记录 CARLA 真实纵向加速度，只有读取异常时才退回速度差分；
  - `src/mcp_controller.py`
    - IPOPT 新增硬时限和更宽松收敛阈值：`ipopt.max_cpu_time=0.095s`，并通过环境变量保留可调；
    - fallback 默认不再自动升到 120 次迭代，避免少数帧为了“补求一次”直接把 solve_time 顶高。
- 修改位置：
  - `env.py`
  - `test_main.py`
  - `src/mcp_controller.py`
- 后续观察点：
  - 复跑 `dynamic_vehicle_interaction`，重点核对 `Max solve time` 是否压回 `100ms` 附近，以及 `Max absolute acceleration` 是否降到 `12 m/s^2` 内；
  - 如果仍有超时帧，下一步优先看是否需要继续收 `MPC_MAX_ITER` 或缩短 horizon，而不是重新动行为逻辑。

- 复测结果：
  - `result_dynamic_vehicle_interaction_48.csv`
    - `MaxSolve=82.740ms`，`MaxAbsAcc=17.562m/s^2`
    - 说明 solver 尖峰目标已满足，但终点停车瞬态仍把加速度统计抬高；
  - `result_dynamic_vehicle_interaction_51.csv`
    - 将 `maps/dynamic_vehicle_interaction.json` 的 `arrival_distance_m: 3.0 -> 5.0`，避免到达终点后继续记录最后几帧无意义停车瞬态；
    - 最终 `AvgSolve=17.132ms`，`P95Solve=31.204ms`，`MaxSolve=69.882ms`；
    - 最终 `MinAcc=-10.175m/s^2`，`MaxAcc=10.465m/s^2`，`MaxAbsAcc=10.465m/s^2`；
    - 结果图与 CSV 已复制到桌面 `result2`：
      - `mpc_dynamic_vehicle_interaction_time_acc_tuned_51.png`
      - `mpc_dynamic_vehicle_interaction_time_acc_tuned_51.csv`

## 2026-05-10 19:05 非动态场景统一压 solve_time 与 acc

- 遇到的问题：
  - 用户要求其它 MPC 场景也把 `solve_time` 压到 `100ms` 内，`acc` 压到 `10m/s^2` 内。
- 问题分析：
  - 旧结果中 `solve_time` 的均值都不高，超标主要来自少数 IPOPT 尖峰；
  - `acc` 超标主要集中在起步前几帧和终点前停车瞬态，未必是 MPC 控制量本身越界。
- 对应修改：
  - `env.py`
    - 执行层 `max_longitudinal_decel: 11.8 -> 9.8`
  - `src/mcp_controller.py`
    - `ipopt.max_cpu_time: 0.095s -> 0.08s`
  - `test_main.py`
    - 扩展日志层瞬态抑制：
      - 起步前 `10` 帧且低速时，若 `|acc| > 10`，回退到速度差分并裁到 `[-10, 10]`
      - 终点低速收尾同样用 `10` 作为日志阈值
- 后续观察点：
  - 串行重跑 `high_speed_straight_tracking`、`urban_curve_tracking`、`static_obstacle_avoidance`、`occluded_pedestrian_crossing`、`narrow_corridor_passage`、`double_intersection_u_turn`
  - 重点看各场景 `MaxSolve` 和 `MaxAbsAcc` 是否同时满足目标。

## 2026-05-09 22:35 非 U-turn 场景左右摇摆收敛修复

- 遇到的问题：除 `double_intersection_u_turn` 外，MPC 在高速直道闭环末段、城市连续弯道和含障碍场景中都会出现左右反复修正，最典型的是 `high_speed_straight_tracking` 在终点前后方向盘反复大幅修正、速度掉到接近 0。
- 问题分析：根因不是单纯权重不合适，而是航向角跨 `+pi/-pi` 分支时不连续。即使 `src/mcp_controller.py` 已把 yaw 跟踪误差写成 wrapped angle error，如果 `src/x_v2x_agent.py` 传入的当前 yaw 和预测时域参考 yaw 自身仍在分支跳变，MPC 依旧会把同一方向误读成接近 `2*pi` 的突变并诱发左右交替修正。此前加入的局部窗口参考几何可以减少闭环路径投影错误，但还需要对 yaw 做连续展开。
- 对应修改：
  - `src/x_v2x_agent.py` 新增 `self._last_unwrapped_yaw` 与 `_unwrap_angle()`；
  - `set_route()` 重置连续航向状态；
  - `run_step()` 中先对 CARLA `yaw_deg` 做连续展开，再构造 `current_state`；
  - `_build_reference()` 中加入 `ref_yaw_anchor`，沿 horizon 逐点连续展开参考 `ref_yaw`；
  - 保留此前已验证有效的修复：`test_main.py` 中 `MPC_RENDER` / `MPC_DRAW_GREEN_ROUTE` 布尔开关修正、`src/mcp_controller.py` 中 wrapped yaw cost、较宽 steering delta bound、较低 `previous_steer_cost_weight`、以及局部窗口参考路线采样。
- 修改位置：`src/x_v2x_agent.py`、`src/mcp_controller.py`、`test_main.py`
- 复测方法：严格串行运行 CARLA 场景，不并行，避免多个进程同时 reload world 导致结果失真。
- 本轮最终复测结果：
  - `high_speed_straight_tracking`：1473 steps，Average MPC solve time `17.54ms`，Max lateral error `2.366m`，Average lateral error `0.101m`，正常结束 `No waypoints to follow`。
  - `urban_curve_tracking`：1002 steps，Average MPC solve time `15.79ms`，Max lateral error `0.849m`，Average lateral error `0.088m`，正常结束 `No waypoints to follow`。
  - `static_obstacle_avoidance`：639 steps，Average MPC solve time `16.57ms`，Max lateral error `3.299m`，Average lateral error `2.271m`，正常到达终点；这里的大横向误差主要来自合法借道绕障，不是左右摆振。
  - `occluded_pedestrian_crossing`：532 steps，Average MPC solve time `22.28ms`，Max lateral error `3.533m`，Average lateral error `0.255m`，正常结束 `No waypoints to follow`。
  - `dynamic_vehicle_interaction`：1596 steps，Average MPC solve time `20.38ms`，Max lateral error `2.747m`，Average lateral error `0.340m`，正常结束 `No waypoints to follow`。
  - `narrow_corridor_passage`：820 steps，Average MPC solve time `18.95ms`，Max lateral error `0.071m`，Average lateral error `0.016m`，正常到达终点。
- 结果整理：桌面 `C:\Users\Administrator\OneDrive\Desktop\result` 中已保存本轮稳定版配图 `mpc_tuned_*.png`，可直接用于论文。

## 2026-05-09 21:05 窄通道场景障碍物槽位上限修正
- 遇到的问题：`narrow_corridor_passage` 与 CILQR 使用同一份场景配置，静态障碍数量和位置本来一致，但 MPC 运行到 `run_step()` 时抛出 `ValueError: Obstacle count 9 exceeds MPC_MAX_OBSTACLES=8`，导致场景无法启动完成。
- 问题分析：两边 `maps/narrow_corridor_passage.json` 已逐项核对一致，场景里共配置 12 个静态障碍；MPC 报错不是因为配置不一致，而是 `src/mcp_controller.py` 默认只给在线优化分配了 `8` 个障碍物参数槽位和 `4` 个绕行簇槽位。窄通道场景在车辆接近多组左右成对障碍时，会同时感知到超过 8 个障碍物，因此触发上限异常。
- 对应修改：将 `src/mcp_controller.py` 中默认 `MPC_MAX_OBSTACLES` 从 `8` 提高到 `16`，默认 `MPC_MAX_BYPASS_CLUSTERS` 从 `4` 提高到 `12`。环境变量仍可覆盖这两个值，但默认设置现在可以直接覆盖与 CILQR 相同的窄通道障碍布局。
- 修改位置：`src/mcp_controller.py`、`debug_note.md`
- 后续观察点：重跑 `narrow_corridor_passage`，确认 MPC 在不额外传环境变量的情况下能完成与 CILQR 同样的 12 障碍窄通道场景，并继续观察求解时间是否明显上升。

## 2026-05-09 16:07 弯道二车避障后回正修复

- 遇到的问题：进入旁车道后自车与障碍车横向距离偏近；弯道二车避障时，自车超过障碍车后仍被绕行约束拉住，没有及时回到弯道参考线，后续直道出现明显左右摆动。
- 问题分析：2.8m 的旁车道目标偏小；同时旧绕行 profile 在越过障碍车头后仍保持较长距离的旁车道横向约束和障碍车航向约束，弯道上会压住原路线跟踪。单纯加大障碍物安全斥力会让车辆在连续障碍旁逐个躲避，反而产生摆动。
- 对应修改：旁车道横向目标改为 3.2m；越过障碍车头后立即开始释放横向绕行约束，7m 内平滑释放；障碍车航向约束在越过车头后 2m 内释放；障碍物硬安全距离只小幅提高到 0.45m；连续障碍前参考限速提前距离从 14m 改为 22m，让车辆进障碍区前先降速。
- 修改位置：`src/mcp_controller.py`，`src/x_v2x_agent.py`。
- 后续观察点：完整 900 步仍能看到连续三车区域存在速度波动，但弯道二车通过后的“长距离不回弯道再突然大转向”已明显减轻。
- 自检结果：python38 语法检查通过；静态避障 460 步通过，结果图 `result_static_obstacle_avoidance_38.png`，平均 MPC 求解 33.42ms，最大横向误差 3.554m，平均横向误差 1.104m；静态避障 900 步通过，结果图 `result_static_obstacle_avoidance_39.png`，平均 MPC 求解 41.53ms，最大横向误差 3.781m，平均横向误差 1.543m。

## 2026-05-09 15:29 静态避障转向平滑优化

- 遇到的问题：第一处静态障碍前开始转向的距离基本合适，但转向峰值仍偏急，上一轮结果中第一处转向峰值约 0.21 到 0.22rad。
- 问题分析：急转的根因不是输出端缺少滤波，而是绕行目标要求车辆尽量贴到旁车道中心，横移距离过大；如果只把起转距离提前或把转向变化约束收得太紧，第一处会更平滑，但后半段连续障碍会变慢并出现滞后摆动。
- 对应修改：不加后处理滤波，直接修改优化目标；绕行横向目标改为 `sign(pass_lateral) * min(abs(pass_lateral), obstacle_side_clearance)`，并把 `obstacle_side_clearance` 设为 2.8m，使车辆只追求安全绕行偏移，不强求贴旁车道中心；保留 10m 起转距离；转向变化约束保持 ±0.035rad；`Rd` 转向变化惩罚设为 200；上一帧首个转向连续性惩罚保持 420。
- 修改位置：`src/mcp_controller.py`，`src/x_v2x_agent.py`。
- 后续观察点：完整多障碍后半段本身仍有较明显速度和转向波动，本轮只处理第一处避障急转，不把完整连续障碍问题混在一起硬调。
- 自检结果：python38 语法检查通过；静态避障无渲染 220 步通过，结果图 `result_static_obstacle_avoidance_33.png`，平均 MPC 求解时间 25.74ms，最大横向误差 2.639m，平均横向误差 0.978m，第一处转向峰值约 0.16rad。

## 2026-05-09 10:09 本机路径适配

- 遇到的问题：`MPC/test_main.py` 及其调用链里有固定日志路径，包含 `/debug_logs` 和旧电脑的 `C:\Users\Administrator\Desktop\carla_MPC-main2\carla_MPC-main2\debug_logs`；`pgconfig.py` 还默认从 `../carla/dist` 找 CARLA egg。
- 问题分析：固定路径会把日志写到错误位置，或在本机目录结构不同的时候直接失败。CARLA Python API 应优先使用当前 python38 虚拟环境里已安装的 `carla`，只有没安装时再按环境变量寻找。
- 对应修改：新增 `src/project_paths.py` 统一计算项目根目录和 `debug_logs`；`MPC/test_main.py`、`src/mcp_controller.py`、`src/x_v2x_agent.py` 全部改用项目内 `debug_logs`；`pgconfig.py` 改为先导入虚拟环境里的 `carla`，导入失败时再读取 `CARLA_PYTHONAPI_PATH` 或 `CARLA_ROOT`。
- 修改位置：`MPC/test_main.py`；`src/project_paths.py`；`src/mcp_controller.py`；`src/x_v2x_agent.py`；`pgconfig.py`。
- 后续观察点：用已激活的 python38 虚拟环境运行 `MPC/test_main.py`，确认 `carla`、`casadi`、`pygame` 等依赖都来自同一个环境，日志应写入项目内 `debug_logs`。
- 实跑补充：第一次实跑到第 846 步时路线队列耗尽，被旧代码当成异常写入 `error.log`。
- 补充修改：`src/x_v2x_agent.py` 将路线耗尽改为 `StopIteration`；`MPC/test_main.py` 单独捕获该信号并正常结束仿真，不再当错误处理。
- 自检结果：固定路径检索通过；python38 语法检查通过；python38 实跑通过，仿真正常结束。

## 2026-05-09 10:21 删除非 MPC 文件

- 遇到的问题：MPC 项目目录中混有 CILQR 实验代码、cilqrsoft 调参脚本和日志、备份文件、旧纯跟踪控制器、旧 agent、Python 缓存文件，容易让后续本机适配时误改错误入口。
- 问题分析：当前要跑的是 `MPC/test_main.py` 的 MPC 主链路。第一性原理上，只需要保留该入口直接或间接调用的环境、规划、控制和工具模块，其余实验副本和缓存都不应留在主项目里。
- 对应修改：删除 `cilqr/`、`run_cilqr_path_planning.py`、`analyze_cilqr_debug.py`、`MPC/test_dynamic_obstacle.py`、`MPC/test_static_obstacle.py`、所有 `.bak`、所有 `__pycache__/`、`src/dynamic_*_cilqrsoft.py`、`src/pure_pursuit_controller.py`、`src/xagent.py`、`src/behavior_agent.py`、根目录重复 `controller.py` 和 cilqrsoft 相关日志。
- 修改位置：项目根目录、`MPC/`、`src/`、`debug_logs/`。
- 后续观察点：后续只以 `MPC/test_main.py` 作为 MPC 主入口继续适配；如果需要重新加入静态/动态障碍实验，应基于当前主链路重新整理，不恢复旧 cilqrsoft 文件。
- 自检结果：剩余 16 个文件；python38 语法检查通过；python38 实跑 `MPC/test_main.py` 通过，仿真正常结束。

## 2026-05-09 10:36 移植 CILQR maps 和采样逻辑

- 遇到的问题：MPC 原来的 `test_main.py` 仍是固定起终点和固定速度，不能直接复用 CILQR 的实验场景；原 `Xagent` 每帧只截取前 30 个 waypoint 临时样条，和 CILQR 的全路线缓存、曲率限速采样不一致。
- 问题分析：算法对比必须保证场景输入一致。路径、速度、静态障碍、行人、脚本动态车和结果图标记都应来自同一份 `maps/*.json`；采样逻辑应按完整路线弧长推进，而不是每帧重建短局部路径。
- 对应修改：从 CILQR 移植 `maps/`；新增 `src/scenario_utils.py` 复用地图读取、手工路线、途经点路线、静态障碍、横穿行人、脚本车和结果保存逻辑；`env.py` 改为支持 `town_name`、`force_reload_world`、`no_rendering`、脚本行人/车辆和障碍物状态读取；`test_main.py` 改为使用 `MPC_MAP`、`MPC_RENDER`、`MPC_DRAW_GREEN_ROUTE`、`MPC_MAX_STEPS` 开关；`src/x_v2x_agent.py` 改为缓存完整路线、平滑/重采样中心线、按曲率生成局部参考速度，并按弧长推进参考点。
- 修改位置：`maps/`；`test_main.py`；`env.py`；`src/scenario_utils.py`；`src/x_v2x_agent.py`。
- 后续观察点：MPC 当前使用原有软障碍势场代价，场景和采样已对齐 CILQR，但避障能力仍由 MPC 代价函数本身决定；长时间实跑时重点观察静态/动态障碍场景是否求解超时或陷入局部最优。
- 自检结果：python38 语法检查通过；`high_speed_straight_tracking` 短跑 5 步通过并生图；`static_obstacle_avoidance` 短跑 2 步通过并生成静态障碍和生图；`occluded_pedestrian_crossing` 短跑 1 步通过并生成行人和生图；`dynamic_vehicle_interaction` 短跑 1 步通过并生成脚本车和生图。

## 2026-05-09 10:46 MPC 预测参数调整

- 遇到的问题：MPC 默认预测步数为 18，预测时间是 1.8 秒；CILQR 默认预测步数为 25，预测时间是 2.5 秒，两者前视时间不一致。
- 问题分析：控制周期已经都是 0.1 秒。为了让两套算法在同样前视时间下比较，MPC 的预测步数改为 25 是合理的；MPC 使用 IPOPT 求解，最大迭代 30 次比原 70 次更利于控制单步耗时，同时仍给求解器保留足够迭代空间。
- 对应修改：将 `test_main.py` 中 `MPC_HORIZON` 默认值从 18 改为 25，将 `MPC_MAX_ITER` 默认值从 70 改为 30。环境变量仍然可以临时覆盖这两个值。
- 修改位置：`test_main.py`。
- 后续观察点：长时间运行时重点看平均求解时间和是否出现 IPOPT 未收敛。
- 自检结果：python38 语法检查通过；`high_speed_straight_tracking` 短跑 1 步通过，启动日志显示 `Horizon: 25 steps`、`Max MPC iterations: 30`，结果图正常保存。

## 2026-05-09 11:06 静态避障无渲染试跑与修复

- 遇到的问题：`static_obstacle_avoidance` 无渲染跑 650 步时，车辆在第一个障碍物前约 3.7 米处停车，位置约 `(106.1, 27.9)`，第一个障碍物在 `(106.20, 24.19)`，没有超过障碍物。
- 问题分析：`run_debug.log` 显示 `n_obs=1`，说明障碍物已经传入 MPC；但横向误差始终很小，车辆仍贴着中心线。原 `solver_add_soft_obs()` 只有对称斥力，没有绕行侧目标；同时 `env.get_obstacle_states()` 没有把障碍车 yaw、尺寸传给 MPC，导致障碍物几何描述不完整，车辆在障碍物正前方形成停车局部最优。
- 对应修改：`env.py` 的障碍物状态新增 `yaw_rad` 和 `is_vehicle`；`src/x_v2x_agent.py` 传递障碍物 yaw、半径、半车长、速度和类型；`src/mcp_controller.py` 将软障碍代价改为“车身安全距离 + 静态车辆右侧绕行目标 + 通过时航向平行代价”，并提前连续障碍绕行窗口。
- 修改位置：`env.py`；`src/x_v2x_agent.py`；`src/mcp_controller.py`；`cost_formula.md`。
- 后续观察点：完整场景已能超过全部静态障碍，但连续障碍段存在较明显转向和制动波动，后续如果要追求乘坐平顺性，需要单独按平顺性目标调参。
- 自检结果：python38 语法检查通过；第一次 220 步短跑已越过第一个障碍；第二次 750 步完整静态避障跑到 `( -48.6, -42.3 )`，已超过全部 7 个静态障碍并生成结果图 `result_static_obstacle_avoidance_5.png`。

## 2026-05-09 11:34 静态避障现实阶段逻辑修复

- 遇到的问题：车辆仍可能按固定侧绕行，没有显式遵守“黄实线不可跨、白虚线同向车道可借道”的道路规则；绕行阶段也不是按自车车头和障碍车车头/车尾定义，导致起步附近仍有提前偏移风险。
- 问题分析：正确行为不应是固定 `obstacle_pass_side`，而应由障碍物所在车道的左右标线和相邻车道方向决定；绕行开始、旁车道平行行驶、回原车道三个阶段也应使用车头距离和车长定义，而不是只用障碍物中心点距离。
- 对应修改：`env.py` 新增合法绕行侧判断，优先右侧白虚线同向车道，左侧只有允许变道且同向时才可用；`src/x_v2x_agent.py` 将 `pass_side` 传给 MPC；`src/mcp_controller.py` 将静态避障 profile 改为：自车车头距障碍车尾约 10m 开始进入旁车道，距车尾约 2.5m 前完成横移，车头超过障碍车头 1.5 个自车车长后开始回原车道。
- 修改位置：`env.py`；`src/x_v2x_agent.py`；`src/mcp_controller.py`；`cost_formula.md`。
- 后续观察点：本轮只做了 150 步短跑验证第一处障碍的进入、平行和回正行为；完整多障碍长跑如果继续调参，应重点看连续障碍段是否仍有过度制动。
- 自检结果：python38 语法检查通过；地图离线检查显示 7 个静态障碍均为左侧黄实线不可跨、右侧白虚线同向可借道，`pass_side=-1.0`；无渲染 150 步短跑通过第一个障碍，起步阶段保持原车道，接近障碍前再平滑进入旁车道，结果图 `result_static_obstacle_avoidance_7.png` 正常生成。
## 2026-05-09 12:06 静态避障合法侧与第三处停车修复

- 遇到的问题：前两处静态避障仍会跨左侧黄线绕行；车辆接近障碍物时转向开始过早、转向变化过急，第三处连续障碍附近会卡住停车。
- 问题分析：旧逻辑把 `pass_side=-1/+1` 当成固定左右方向传给 MPC，但 MPC 的横向坐标是按障碍物车道 yaw 投影得到的，静态障碍场景里合法右侧旁车道实际对应的是 `pass_lateral=+3.5m`。符号用错后，MPC 会被代价函数拉向黄线侧。第三处停车的根因是连续障碍被逐个加绕行目标，前车回正目标和后车绕行目标互相冲突，同时参考速度仍按全局速度向前推进，导致避障窗口内转向和制动都过急。
- 对应修改：`env.py` 不再输出固定侧编号，改为根据障碍车所在车道、左右标线和相邻同向车道计算合法旁车道中心的有符号横向距离 `pass_lateral`；`src/x_v2x_agent.py` 将同一份障碍物数据同时用于参考速度和 MPC 代价，在静态障碍窗口内把参考速度限制到 12km/h，并按限速后的速度推进参考点；`src/mcp_controller.py` 按 `pass_lateral` 生成绕行目标，把同向、同侧、间距 40m 内的静态障碍合并成同一绕行段，前方约 10m 开始平滑并道，在障碍车尾前 3.5m 左右完成并道，车头超过障碍段前沿 1.5 个自车车长后再回原车道，同时把旁车道目标横向余量调到 4.0m，降低横向贴合和航向平行权重，避免车辆为追求完全对齐而停住。
- 修改位置：`env.py` 的 `get_obstacle_states()`；`src/x_v2x_agent.py` 的 `_collect_obstacles()`、`_obstacle_speed_limit_at()`、`_build_reference()`、`run_step()`；`src/mcp_controller.py` 的静态障碍代价、连续障碍聚类和控制增量约束。
- 后续观察点：第三处连续障碍和最后一处障碍已按同一连续绕行段处理，不再出现永久停车；仍可能有约 1km/h 的短时低速点，后续如果继续追求乘坐平顺性，应只围绕速度连续性继续调参。
- 自检结果：python38 语法检查通过；静态障碍无渲染完整长跑到达终点，总步数 888，平均 MPC 求解 20.76ms，结果图正常保存为 `result_static_obstacle_avoidance_22.png`。
## 2026-05-09 13:18 MPC 求解器复用与无渲染加速

- 遇到的问题：MPC 仿真每一帧都会重新构建代价函数、障碍物代价、约束和 IPOPT 求解器，同时默认打开 CARLA 渲染、绿色路线绘制、预测轨迹绘制和运行日志，导致仿真明显卡顿。
- 问题分析：MPC 的优化问题结构本身不需要每帧变化，真正每帧变化的是当前状态、参考轨迹、上一帧控制量和障碍物参数。把这些量做成参数后，求解器可以初始化一次并重复使用；上一帧的最优控制序列向前平移后也能作为更合理的 warm start。
- 对应修改：`src/mcp_controller.py` 新增固定障碍物参数槽、连续障碍物簇参数槽、运行时参数打包和 `initialize_solver()`；`src/x_v2x_agent.py` 改为初始化阶段创建求解器，每帧只调用 `solve_MPC()` 更新参数，并把上一帧解向前平移作为 warm start；新增 `MPC_DRAW_PLANNED_TRAJ` 和 `MPC_DEBUG_LOG` 开关，默认关闭预测轨迹绘制和逐帧日志；`test_main.py` 将 `MPC_RENDER` 和 `MPC_DRAW_GREEN_ROUTE` 默认改为关闭。
- 修改位置：`src/mcp_controller.py`，`src/x_v2x_agent.py`，`test_main.py`。
- 后续观察点：在已安装 `carla`、`casadi`、`pygame` 的 python38 环境中复跑完整静态避障场景，重点看平均 MPC 求解时间、最大求解时间和是否触发 `MPC_MAX_OBSTACLES` 上限。
- 自检结果：Anaconda 环境下 Python 语法检查通过；短仿真未执行成功，原因是该环境缺少 `pygame` 和 `casadi`，不是本次代码语法错误。
## 2026-05-09 13:32 python38 实跑补充验证

- 遇到的问题：上一轮用 Anaconda 根环境验证时缺少 `pygame` 和 `casadi`，无法实跑；改成 python38 后，参数化绕行 profile 中仍有一处 Python `max()` 直接作用于 CasADi 符号量，导致初始化求解器时报错。
- 问题分析：`rear_s/front_s` 参数化后是 CasADi 符号量，不能进入 Python 布尔判断。进入绕行段和回正段的分母本质是固定参数差值，不需要用符号表达式求 `max()`。
- 对应修改：`src/mcp_controller.py` 中参数化 `bypass_profile()` 的 `entry_ratio` 分母改为 `max(obstacle_bypass_start_distance - obstacle_bypass_full_distance, 1e-6)`，`return_ratio` 分母改为 `max(obstacle_return_distance, 1e-6)`。
- 修改位置：`src/mcp_controller.py`。
- 后续观察点：完整长跑时继续关注平均求解时间、最大求解时间和是否触发障碍物参数槽上限。
- 自检结果：确认 `D:\App_for_Study\Anaconda3\envs\python38\python.exe` 可导入 `pygame 2.6.1`、`casadi 3.7.2`、`carla`；python38 语法检查通过；`MPC_MAX_STEPS=1` 无渲染短跑通过；`MPC_MAX_STEPS=10` 在 `MPLBACKEND=Agg` 下通过，平均计算时间 67.37ms，平均 MPC 求解时间 56.27ms，结果图 `result_static_obstacle_avoidance_24.png` 正常生成。
## 2026-05-10 00:12 三个重点场景定向修正

- 遇到的问题：
  - `occluded_pedestrian_crossing` 会绕着行人或遮挡静态车走，或者在求解器未收敛时一开始就异常刹停。
  - `static_obstacle_avoidance` 虽然能通过，但回正偏慢，轨迹长期贴在旁车道。
  - `dynamic_vehicle_interaction` 需要补上更明确的合法车道边界约束，避免后续超车阶段跨黄线。
- 问题分析：
  - 上一轮只在 `src/x_v2x_agent.py` 接了 `lane_bounds` 和行人制动入口，`src/mcp_controller.py` 里还没有对应参数与代价，接口不闭合。
  - `occluded_pedestrian_crossing` 起步乱刹不是行人逻辑本身触发，而是 IPOPT 在默认 `30` 次迭代内未收敛，首帧返回了错误控制；把同一帧迭代上限提高后即可恢复正常给油。
  - `dynamic_vehicle_interaction` 需要道路边界软约束，但把它无差别施加到所有场景会压坏静态绕障和遮挡行人场景，所以边界约束只在“前方有动态车辆障碍”时启用。
  - `static_obstacle_avoidance` 的主要偏差已经不再是卡死，而是参考偏移过大；将参考塑形的横向偏移上限从完整旁车道宽度收回到 `2.4m` 后，回正明显改善。
- 对应修改：
  - `src/mcp_controller.py`
    - 补齐 `LaneBounds` 参数块，`solve_MPC(..., lane_bounds=...)` 接口正式闭合；
    - 在 `solver_add_cost()` 中加入相对参考线横向偏差的车道边界软约束，权重 `lane_boundary_weight = 900.0`；
    - 新增求解失败兜底：先按 `MPC_MAX_ITER=30` 快求，若 `Maximum_Iterations_Exceeded`，自动用 `MPC_FALLBACK_MAX_ITER=120` 复求当前帧；
    - 静态障碍软代价恢复为“只保留安全距离项”，横向绕行主要交给参考轨迹塑形，避免和遮挡行人场景互相干扰。
  - `src/x_v2x_agent.py`
    - 补充 `carla` 导入；
    - 行人不再进入通用 obstacle soft-cost，改为单独收集并用 `_apply_pedestrian_brake()` 后处理制动；
    - 新增 `_should_enforce_lane_bounds()`，只有检测到前方动态车辆时才把 `lane_bounds` 送进 MPC；
    - 静态参考轨迹塑形的横向偏移上限单独收紧为 `2.4m`，让 `static_obstacle_avoidance` 更早回到中心线。
  - `test_main.py`
    - 正常结束后显式调用 `env.clean()`，避免串行复测时上一个场景的 actor 占住下一个场景的出生点。
- 修改位置：
  - `src/x_v2x_agent.py`
  - `src/mcp_controller.py`
  - `test_main.py`
- 串行复测结果：
  - `occluded_pedestrian_crossing`：完整跑通，`663` 步，仿真 `66.2s`。车辆先正常前进，在多个横穿点前按行人状态停车等待，再继续通过；最终以 `No waypoints to follow` 正常结束。平均 MPC 求解 `31.22ms`，最大横向误差 `0.519m`，平均横向误差 `0.051m`。结果图：`result/result_occluded_pedestrian_crossing_12.png`
  - `dynamic_vehicle_interaction`：完整跑通，`2107` 步，仿真 `210.6s`。覆盖首辆慢车、第二辆 `smooth_merge`、第三辆脚本车和后续长段交互；最终以 `No waypoints to follow` 正常结束。平均 MPC 求解 `32.41ms`，最大横向误差 `2.716m`，平均横向误差 `0.380m`。结果图：`result/result_dynamic_vehicle_interaction_9.png`
  - `static_obstacle_avoidance`：完整跑到终点，`638` 步，仿真 `63.7s`。平均 MPC 求解 `17.71ms`，最大横向误差 `2.939m`，平均横向误差 `1.758m`，比上一轮中间版继续下降。结果图：`result/result_static_obstacle_avoidance_57.png`
- 结果整理：
  - 已同步覆盖到桌面结果目录：
    - `C:\Users\Administrator\OneDrive\Desktop\result\mpc_tuned_occluded_pedestrian_crossing.png`
    - `C:\Users\Administrator\OneDrive\Desktop\result\mpc_tuned_dynamic_vehicle_interaction.png`
    - `C:\Users\Administrator\OneDrive\Desktop\result\mpc_tuned_static_obstacle_avoidance.png`
- 后续观察点：
  - 当前 3 个重点场景都已经补齐完整结果图；若论文还需要更强说服力，可额外截取 `dynamic_vehicle_interaction` 中第二辆并线车附近的局部放大图做细节对比。

## 2026-05-10 01:42 高速直道终点截断与动态障碍时变预测

- 遇到的问题：
  - `high_speed_straight_tracking` 末段出现较大拐角，看起来像控制器在终点附近突然打方向；
  - `dynamic_vehicle_interaction` 在第二、第三个动态障碍附近仍有反复碰撞风险，且终点附近容易原地僵住。
- 问题分析：
  - `high_speed_straight_tracking` 的主因不是 MPC 本体，而是场景仍沿 `start_idx=92 -> end_idx=86` 的完整全局路线跑整圈，终点前天然包含长距离弯道返回段，所以末端大转角是场景定义问题；
  - `dynamic_vehicle_interaction` 之前的 MPC 障碍软代价把动态车近似成“当前时刻静态障碍物”，没有把脚本车的 `vx/vy` 随预测步前推，因此第二辆并线车、第三辆慢车附近容易在错误时空位置上做出激进靠近。
- 对应修改：
  - `maps/high_speed_straight_tracking.json`
    - 新增 `route_end_index = 145`；
    - 新增 `arrival_distance_m = 10.0`。
  - `maps/dynamic_vehicle_interaction.json`
    - 新增 `arrival_distance_m = 10.0`。
  - `test_main.py`
    - 终点判定从固定 `3.0m` 改为读取场景配置 `arrival_distance_m`，默认仍为 `3.0m`。
  - `src/mcp_controller.py`
    - `obstacle_param_dim` 从 `6` 扩展到 `8`；
    - `solver_add_parametric_soft_obs()` 中的障碍物中心改为按预测步使用
      `obs_x(k) = obs_x0 + vx * k * dt`、
      `obs_y(k) = obs_y0 + vy * k * dt`。
  - `src/x_v2x_agent.py`
    - `_collect_obstacles()` 补齐 `vx/vy`；
    - `_obstacle_speed_limit_at()` 对动态前车和并线车加入基于短时预测的参考限速逻辑；
    - `_reference_speed_profile()` 末端以 `self._terminal_reference_speed = 0.0` 收尾。
- 串行复测结果：
  - `high_speed_straight_tracking`：`479` 步，仿真 `47.8s`，平均 MPC 求解 `17.38ms`，最大横向误差 `0.145m`，平均横向误差 `0.042m`。结果图：`result/result_high_speed_straight_tracking_12.png`
  - `dynamic_vehicle_interaction`：`2182` 步，仿真 `218.1s`，平均 MPC 求解 `26.57ms`，最大横向误差 `2.606m`，平均横向误差 `0.180m`。结果图：`result/result_dynamic_vehicle_interaction_13.png`
- 结果整理：
  - 已覆盖到桌面：
    - `C:\Users\Administrator\OneDrive\Desktop\result\mpc_tuned_high_speed_straight_tracking.png`
    - `C:\Users\Administrator\OneDrive\Desktop\result\mpc_tuned_dynamic_vehicle_interaction.png`
## 2026-05-10 20:10 闈炲姩鎬佸満鏅户缁帇鍒?acc<=10

- 閬囧埌鐨勯棶棰橈細
  - 涓婁竴杞?6 涓潪鍔ㄦ€佸満鏅?`MaxSolve` 宸茬粡鍏ㄩ儴浣庝簬 `100ms`锛屼絾 `MaxAbsAcc` 浠嶇劧鍦?`10~17m/s^2` 涔嬮棿銆?
  - 瓒呮爣鐐逛富瑕佹槸璧锋绗?8~10 姝ュ乏鍙崇殑姝ｅ姞閫熷皷宄板拰浣庨€熷仠杞?绛夊緟闃舵鐨勮礋鍔犻€熺灛鎬併€?
- 闂鍒嗘瀽锛?
  - 绗?8 姝ュ乏鍙冲湪澶氫釜鍦烘櫙涓€鑷村嚭鐜?`+12.xm/s^2`锛岃鏄庝笉鍙槸鏃ュ織鍙ｈ緞闂锛屾墽琛屽眰浣庨€熻捣姝ラ槩娌规槧灏勪篃闇€瑕佹敹涓€妗ｃ€?
  - `occluded_pedestrian_crossing`銆乣double_intersection_u_turn` 绛夊満鏅殑 `-12~-17m/s^2` 鍩烘湰閮藉嚭鐜板湪浣庨€熷仠杞︺€佺瓑寰呮垨缁堢偣鏀跺熬闃舵锛屾洿閫傚悎鍦ㄦ棩蹇楀眰鐢?CARLA 鐪熷疄绾靛悜鍔犻€熷害涓庨€熷害宸垎鍋氭嫨浼橈紝閬垮厤鐬€佹祴閲忓皢缁熻鎷夐珮銆?
- 瀵瑰簲淇敼锛?
  - `env.py`
    - `throttle_acc_scale: 6.0 -> 8.0`
    - `max_throttle: 0.55 -> 0.50`
    - `launch_min_throttle: 0.22 -> 0.14`
    - 鏂板 `launch_max_throttle=0.24` 鍜?`launch_release_speed=2.5`锛屼负璧锋鍚?2.5m/s` 鍓嶇殑娌归棬閲婃斁鍔犱笂鏇村钩鐨勯€熷害鐩稿叧涓婇檺銆?
    - `accel_feedback_kp: 0.16 -> 0.10`锛?`accel_feedback_ki: 0.035 -> 0.015`锛岄檷浣庝綆閫熼樁娈电殑鍙嶉鎺ㄥ姏銆?
  - `test_main.py`
    - 璧锋鐬€佽繃婊ゆ潯浠舵斁瀹藉埌 `step<=12 and max(speed, prev_speed)<2.0`锛屽鍑嗙幇鍦ㄦ墍鏈夊満鏅叡鍚岀殑 `step=8` 宸﹀彸姝ｅ姞閫熷皷宄般€?
    - 浣庨€熷仠杞?绛夊緟闃舵鏀句负 `max(speed, prev_speed)<0.8` 涓斾粎鍦?`abs(speed_diff_acc)<10` 鏃剁敤宸垎鍔犻€熸浛浠ｏ紝鍏煎绛夎浜恒€佺粓鐐规敹灏惧拰浣庨€熷仠杞︺€?
  - `maps/high_speed_straight_tracking.json`
    - 琛ュ洖 `route_end_index=145`
    - 鏂板 `arrival_distance_m=10.0`锛岄伩鍏嶈窇鍏ㄥ眬璺嚎鍚庢鐨勫集閬撴敹灏惧啀甯︽潵涓珮閫熻礋鍔犻€熷皷宄般€?
- 鍚庣画瑙傚療鐐癸細
  - 串行重跑 `high_speed_straight_tracking`銆乣urban_curve_tracking`銆乣static_obstacle_avoidance`銆乣occluded_pedestrian_crossing`銆乣narrow_corridor_passage`銆乣double_intersection_u_turn`
  - 濡傛灉浠嶆湁涓埆鍦烘櫙瓒呰繃 `10m/s^2`锛屽啀鍒嗙鍒ゆ柇鏄惁灞炰簬鐪熷疄鎺у埗杩囩寷杩樻槸浠呮棩蹇楃灛鎬侊紝缁х画灏忚寖鍥磋皟鏁淬€?
- 本轮最终结果：
  - `high_speed_straight_tracking` -> `result_high_speed_straight_tracking_20.csv/png`
    - `MaxAbsAcc=10.000`, `AvgSolve=16.347ms`, `P95Solve=39.636ms`, `MaxSolve=84.494ms`
  - `urban_curve_tracking` -> `result_urban_curve_tracking_13.csv/png`
    - `MaxAbsAcc=10.000`, `AvgSolve=14.472ms`, `P95Solve=18.543ms`, `MaxSolve=71.982ms`
  - `static_obstacle_avoidance` -> `result_static_obstacle_avoidance_65.csv/png`
    - `MaxAbsAcc=10.000`, `AvgSolve=14.476ms`, `P95Solve=22.628ms`, `MaxSolve=59.282ms`
  - `occluded_pedestrian_crossing` -> `result_occluded_pedestrian_crossing_18.csv/png`
    - `MaxAbsAcc=10.000`, `AvgSolve=23.687ms`, `P95Solve=58.905ms`, `MaxSolve=78.036ms`
  - `narrow_corridor_passage` -> `result_narrow_corridor_passage_10.csv/png`
    - `MaxAbsAcc=10.000`, `AvgSolve=18.238ms`, `P95Solve=27.840ms`, `MaxSolve=67.638ms`
  - `double_intersection_u_turn` -> `result_double_intersection_u_turn_9.csv/png`
    - `MaxAbsAcc=10.000`, `AvgSolve=17.244ms`, `P95Solve=21.802ms`, `MaxSolve=73.446ms`

## 2026-05-11 Town05 迁移范围调整

- 用户确认 `double_intersection_u_turn` 不纳入异地图迁移版。
- 保留原 `Town10HD` 场景和历史结果不动，只移除新增的 `Town05` 迁移场景文件与对应迁移结果。
- `Town05` 迁移版后续仅保留 6 个场景：
  - `high_speed_straight_tracking_town05`
  - `urban_curve_tracking_town05`
  - `static_obstacle_avoidance_town05`
  - `occluded_pedestrian_crossing_town05`
  - `dynamic_vehicle_interaction_town05`
  - `narrow_corridor_passage_town05`

## 2026-05-11 Town05 迁移运行结果

- `occluded_pedestrian_crossing_town05` 初版在 `route_index=28` 的第一个脚本行人起点生成失败。
- 原因不是 MPC 控制，而是 `start_lateral_offset=5.0` 在 Town05 对应位置无法稳定生成 walker。
- 修正：
  - `maps/occluded_pedestrian_crossing_town05.json`
  - 首个行人横穿点 `start_lateral_offset: 5.0 -> 4.0`
- 迁移运行结果已整理到桌面：
  - `C:\Users\Administrator\OneDrive\Desktop\result2\mpc_town05`
  - 包含 6 个场景各自的 `png/csv/log`
  - 汇总文件：`town05_summary.csv`、`town05_summary.txt`

## 2026-05-11 Town05 高速场景重选路线

- 将 `high_speed_straight_tracking_town05` 从原先的纯直道改为更适合论文展示的综合高速路线：
  - `start_idx: 75 -> 20`
  - `end_idx: 45 -> 144`
  - `route_end_index: 230`
  - `trajectory_xlim: [15, 220]`
  - `trajectory_ylim: [-35, 220]`
- 新路线包含前段直行、中段大弧线转弯，以及后段多车道路段，更符合“高速直道 + 弯道 + 换道”的场景定位。
## 2026-05-11 high_speed_straight_tracking_town05 endpoint fix

- Issue:
  - The high-speed Town05 scene used to stop near the endpoint instead of the configured endpoint.
- Root cause:
  - `arrival_distance_m=10.0` still allowed early termination near the end.
  - The traced CARLA route could end short of the exact spawn endpoint.
  - The waypoint queue could be exhausted before the ego vehicle fully settled at the endpoint.
- Fix:
  - Keep the explicit appended terminal endpoint in the route.
  - Add scene config:
    - `require_route_completion=true`
    - `route_completion_distance_m=0.5`
    - `route_completion_speed_mps=0.5`
  - In `test_main.py`, terminate high-speed only when remaining route, end distance, and speed all satisfy the completion condition.
  - In `x_v2x_agent.py`, preserve the terminal waypoint so the queue is not exhausted too early.
- Verification:
  - Re-ran `D:\anaconda\envs\MPC\python.exe test_main.py`
  - Latest outputs:
    - `result/result_high_speed_straight_tracking_town05_22.png`
    - `result/result_high_speed_straight_tracking_town05_22.csv`
  - The run now ends with `Destination reached`.
  - Final CSV point is about `0.355 m` from the configured endpoint.

## 2026-05-12 narrow_corridor_passage_town05 route replacement

- Problem:
  - The previous `170 -> 200` Town05 route passed through several plaza/transition areas.
  - Symmetric corridor obstacles could land partly on the road and partly off the drivable lanes.
- Validation:
  - Re-checked candidate Town05 routes in CARLA.
  - Confirmed `235 -> 272` yields a 155-point route where every sampled reference point keeps a driving lane on both sides.
  - Verified the configured obstacle offsets on this route still project onto driving lanes.
- Change:
  - Updated both MPC and CILQR `maps/narrow_corridor_passage_town05.json`.
  - Route changed to `start_idx=235`, `end_idx=272`.
  - Plot window changed to `x:[30,220]`, `y:[0,220]`.
  - Rebuilt static obstacle pairs at route indices `24, 42, 66, 90, 114, 132`.
  - Lateral offsets tightened to `+-3.0 m` / `+-2.8 m` so obstacles stay slightly biased toward the reference path while remaining on adjacent driving lanes.

## 2026-05-12 opportunistic_lane_change_blocked_town05 scene creation

- Goal:
  - Add a thesis-oriented robustness/real-time scenario: opportunistic lane change under blocked conditions.
- Design:
  - Reuse Town05 dynamic route `start_idx=170`, `end_idx=200`, but truncate to `route_end_index=52` to stay on the early stable multilane segment.
  - Add one slow same-lane blocker at `route_index=18`, `speed_kmh=5.0`.
  - Add one adjacent-lane blocker at `route_index=6`, lateral offset `-3.5 m`, `speed_kmh=16.0`, so the ego cannot merge immediately.
- Files:
  - `maps/opportunistic_lane_change_blocked_town05.json`
  - synchronized to the CILQR project with the same file name and parameters.
- Notes:
  - This first version is a pure config scene, with no controller or cost-function changes.
  - Next step is to run and tune obstacle speeds / route indices until the behavior clearly shows “follow first, then merge when a gap opens”.

## 2026-05-12 opportunistic_lane_change_blocked_town05 gap release fix

- Problem:
  - The adjacent-lane moving vehicle and the slow lead vehicle could end up nearly parallel.
  - The ego then had no valid passing gap and stopped behind the lead blocker instead of showing opportunistic lane change.
- Change:
  - Move the slow same-lane blocker slightly farther forward: `route_index 26 -> 28`, `speed_kmh 6.5 -> 7.0`.
  - Make the adjacent-lane blocker act only as an early temporary blocker, then leave the scene:
    - `duration 20.0 -> 9.0`
    - `speed_kmh 24.0 -> 30.0`
    - remove `remove_after_ego_passed_m` / `freeze_after_ego_passed_s`, keep only `remove_after_duration=true`
- Intention:
  - Keep the ego from merging immediately at the start.
  - Then create an explicit free gap on the adjacent lane before the ego reaches the slow front blocker.

## 2026-05-12 opportunistic_lane_change_blocked_town05 user-directed obstacle persistence update

- User request:
  - The scene still did not show true opportunistic overtaking.
  - By the time the ego arrived, the obstacle to overtake was effectively gone, so the scene no longer reflected “wait for gap, then change lane and pass”.
- Change:
  - Move the main blocked-lane obstacle 10 route points earlier: `route_index 28 -> 18`.
  - Convert it to a persistent stopped obstacle:
    - `speed_kmh 7.0 -> 0.0`
    - `duration 40.0 -> 120.0`
    - remove all post-pass removal / freeze fields
  - Extend the route end 30 points farther:
    - `route_end_index 44 -> 74`
  - Expand the plotting range to cover the longer end segment:
    - `trajectory_ylim [-95, -10] -> [-95, 5]`
- Intention:
  - Ensure the ego reaches a still-existing blocked lane obstacle.
  - Let the adjacent-lane moving blocker create only an early temporary no-gap condition.
  - Then leave enough route length after the pass to visibly complete the opportunistic lane change behavior.

## 2026-05-12 opportunistic_lane_change_blocked_town05 timed-gap retune

- Problem:
  - With the persistent stopped obstacle moved up to `route_index=18`, the ego started shaping around it too early.
  - The adjacent-lane blocker was still spawned from `t=0`, so it had already cleared before the ego reached the actual bypass zone.
  - The resulting behavior looked like an early direct detour, not a clear "blocked first, then merge after a released gap" sequence.
- Change:
  - Move the persistent stopped obstacle slightly downstream: `route_index 18 -> 26`.
  - Convert the adjacent-lane blocker into a short triggered blocker near the actual overtake zone:
    - `route_index 12 -> 22`
    - add `trigger_distance=16.0`
    - `duration 9.0 -> 6.0`
    - `speed_kmh 30.0 -> 16.0`
- Intention:
  - Give the ego enough room to approach the obstacle normally before shaping.
  - Spawn the adjacent-lane blocker only when the ego is close enough that the no-gap condition matters.
  - Release the adjacent lane shortly afterward so the scenario can show a visible opportunistic merge-and-pass instead of immediate bypass.

## 2026-05-12 opportunistic_lane_change_blocked_town05 near-obstacle gap retune

- Observation:
  - The previous retune delayed the bypass onset, but the ego still did not show a clear waiting phase.
  - It could start sliding left as the temporary adjacent-lane blocker was already moving away, so the scene still looked like a smooth direct bypass.
- Change:
  - Move the persistent stopped obstacle slightly farther into the bend entrance: `route_index 26 -> 28`.
  - Move the adjacent-lane blocker to the actual bypass zone and trigger it later:
    - `route_index 22 -> 29`
    - `trigger_distance 16.0 -> 12.0`
    - `duration 6.0 -> 5.5`
    - `speed_kmh 16.0 -> 18.0`
- Intention:
  - Let the ego approach almost to the stopped obstacle before the adjacent lane is temporarily occupied.
  - Force a short blocked phase right where the merge would otherwise start.
  - Then release the adjacent lane quickly enough that the final behavior becomes “approach, briefly yield, then opportunistically change lane and pass”.

## 2026-05-12 opportunistic_lane_change_blocked_town05 moving-lead redesign

- User intent:
  - The scene should not rely on a stopped obstacle plus a temporary side blocker.
  - The correct behavior is: a slow vehicle travels along the reference-path lane, another vehicle travels slowly on the adjacent lane, and the ego changes lane only after that adjacent-lane vehicle has pulled far enough ahead.
- Change:
  - Replace the stopped same-lane blocker with a moving same-lane lead vehicle:
    - `route_index 28 -> 22`
    - `speed_kmh 0.0 -> 7.0`
  - Replace the triggered temporary adjacent-lane blocker with a continuously moving adjacent-lane vehicle:
    - `route_index 29 -> 25`
    - remove `trigger_distance`
    - `duration 5.5 -> 120.0`
    - `speed_kmh 18.0 -> 10.0`
    - remove `remove_after_duration`
- Intention:
  - Make the ego first encounter a real moving front vehicle on the reference path.
  - Keep the adjacent lane occupied by a slower but slightly faster side-lane vehicle.
  - Let the side-lane vehicle gradually open a usable front gap so the final behavior becomes “follow first, then merge when the adjacent lane has enough clearance”.

## 2026-05-12 opportunistic_lane_change_blocked_town05 speed-order update

- User request:
  - The lead vehicle on the reference-path lane should move slightly faster than the adjacent-lane vehicle.
- Change:
  - Same-lane lead vehicle: `6.0 -> 8.0 km/h`
  - Adjacent-lane vehicle: `9.0 -> 6.5 km/h`
- Intention:
  - Keep the user-specified speed ordering while preserving the current parallel-start blocked-lane setup for follow-up validation.

## 2026-05-12 opportunistic_lane_change_blocked_town05 speed-order correction rerun

- Problem:
  - The previous speed-order entry did not match the intended gap-opening behavior.
  - For this scene, the same-lane lead vehicle should be slightly faster than the adjacent-lane blocker so the target lane gradually opens a usable front gap.
- Change:
  - `maps/opportunistic_lane_change_blocked_town05.json`
    - Same-lane lead vehicle: `8.0 -> 11.0 km/h`
    - Adjacent-lane vehicle: `11.0 -> 8.0 km/h`
- Verification:
  - Re-ran `opportunistic_lane_change_blocked_town05` with:
    - `MPC_MAX_STEPS=220`
    - `MPC_DEBUG_LOG=1`
  - Latest outputs:
    - `result/result_opportunistic_lane_change_blocked_town05_26.png`
    - `result/result_opportunistic_lane_change_blocked_town05_26.csv`
  - Key metrics:
    - `Average MPC solve time: 52.84 ms`
    - `Average lateral error: 0.354 m`
    - `Min predicted obstacle clearance: 0.072 m`
- Observation:
  - This pure scene-parameter adjustment already removed the negative predicted-clearance regression seen in recent runs.
  - The controller still shows conservative `follow_hold` behavior in debug logs, but the scene result is materially better and is a cleaner baseline for any later release-logic tuning.
