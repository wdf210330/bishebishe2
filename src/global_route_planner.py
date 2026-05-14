"""
global_route_planner.py - 全局路径规划器

功能说明：
    基于A*算法的全局路径规划器
    将CARLA道路网络转换为图结构，通过A*搜索找到从起点到终点的最优路径

核心算法：
    1. 从CARLA获取道路拓扑（路段、车道）
    2. 构建NetworkX有向图
    3. A*搜索找到最短路径
    4. 生成路径点序列

参考文献：
    - CARLA官方GlobalRoutePlanner实现
    - 版权：Copyright (c) 2018-2020 CVC
    - 协议：MIT License

作者：CARLA Team (修改自官方版本)
"""

# Copyright (c) 2018-2020 CVC.
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import sys
import os

# 添加父目录路径
try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'official'))
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'utils'))
except IndexError:
    pass

import math
import numpy as np
import networkx as nx  # 网络图库，用于A*搜索

import carla
from src.local_planner import RoadOption
from src.misc import vector


class GlobalRoutePlanner(object):
    """
    全局路径规划器类
    
    将CARLA的道路网络转换为图结构，通过A*算法搜索最短路径
    
    图结构：
        - 节点：道路段端点 (x, y, z) 坐标
        - 边：道路段，包含entry/exit waypoint、path、长度等信息
        - 节点属性：vertex (x, y, z)
        - 边属性：entry_vector, exit_vector, net_vector, intersection, type等
    
    使用方法：
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
        route = grp.trace_route(start_location, end_location)
    """
    
    def __init__(self, wmap, sampling_resolution):
        """
        初始化全局路径规划器
        
        参数：
            wmap: CARLA地图对象（carla.Map）
            sampling_resolution: 路径采样分辨率（米），越小路径越精确
        
        初始化流程：
            1. 构建道路拓扑
            2. 构建图结构
            3. 处理断头路
            4. 建立换道连接
        """
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None  # 道路拓扑列表
        self._graph = None     # NetworkX有向图
        self._id_map = None    # 坐标->节点ID映射
        self._road_id_to_edge = None  # 道路ID->边的映射
        
        # 交叉口处理相关
        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID
        
        # 执行初始化
        self._build_topology()      # 构建拓扑
        self._build_graph()          # 构建图
        self._find_loose_ends()      # 处理断头路
        self._lane_change_link()     # 建立换道连接
    
    def trace_route(self, origin, destination):
        """
        计算从起点到终点的全局路径
        
        使用A*算法搜索最短路径，并生成路径点序列
        
        参数：
            origin: 起点CARLA Location
            destination: 终点CARLA Location
        
        返回：
            route_trace: [(carla.Waypoint, RoadOption), ...] 路径点及拓扑选项列表
        """
        route_trace = []
        
        # 1. A*搜索得到节点序列
        route = self._path_search(origin, destination)
        
        # 2. 获取起点和终点对应的Waypoint
        current_waypoint = self._wmap.get_waypoint(origin)
        destination_waypoint = self._wmap.get_waypoint(destination)
        
        # 3. 遍历路径节点，生成Waypoint序列
        for i in range(len(route) - 1):
            # 判断道路拓扑（直行、左转、右转等）
            road_option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i+1]]
            path = []
            
            # -------------------------------------------
            # 处理非车道保持的情况（转弯、换道等）
            # -------------------------------------------
            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                # 记录当前waypoint和决策
                route_trace.append((current_waypoint, road_option))
                
                # -------------------------------------------
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._graph.edges[n1, n2]

                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path']) - 1, closest_index + 5)
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))
            
            # -------------------------------------------
            # 车道保持：收集路径上的所有Waypoint
            # -------------------------------------------
            else:
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self._find_closest_in_list(current_waypoint, path)
                
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    
                    # 检查是否到达终点附近
                    if (len(route) - i <= 2 and 
                        waypoint.transform.location.distance(destination) < 2 * self._sampling_resolution):
                        break
                    # 检查是否到达目标车道
                    elif (len(route) - i <= 2 and 
                          current_waypoint.road_id == destination_waypoint.road_id and
                          current_waypoint.section_id == destination_waypoint.section_id and
                          current_waypoint.lane_id == destination_waypoint.lane_id):
                        destination_index = self._find_closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break
        
        return route_trace
    
    def _build_topology(self):
        """
        从CARLA获取道路拓扑并构建路段列表
        
        每个路段（segment）是一个字典，包含：
            - entry: 入口Waypoint
            - exit: 出口Waypoint
            - entryxyz: 入口坐标 (x, y, z)
            - exitxyz: 出口坐标 (x, y, z)
            - path: 入口到出口之间的路径点列表
        
        采样分辨率决定path中点的密度
        """
        self._topology = []
        
        # 获取原始拓扑（相邻Waypoint对）
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            
            # 四舍五入避免浮点误差
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            
            # 构建路段字典
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            
            # 在入口和出口之间按分辨率采样
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
            
            self._topology.append(seg_dict)
    
    def _build_graph(self):
        """
        将道路拓扑转换为NetworkX有向图
        
        图结构：
            - 节点：道路段端点 (x, y, z)
            - 边：道路段
        
        节点属性：
            - vertex: (x, y, z) 位置
        
        边属性：
            - length: 路径点数量
            - path: Waypoint列表
            - entry_waypoint: 入口Waypoint
            - exit_waypoint: 出口Waypoint
            - entry_vector: 入口方向向量
            - exit_vector: 出口方向向量
            - net_vector: 弦向量
            - intersection: 是否在交叉口
            - type: 道路类型
        """
        self._graph = nx.DiGraph()
        self._id_map = dict()           # {(x,y,z): node_id}
        self._road_id_to_edge = dict()  # {road_id: {section_id: {lane_id: (n1, n2)}}}
        
        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id
            
            # 添加节点
            for vertex in entry_xyz, exit_xyz:
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            
            # 建立道路ID到边的映射
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
            
            # 获取方向向量
            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()
            
            # 添加边
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1,
                path=path,
                entry_waypoint=entry_wp,
                exit_waypoint=exit_wp,
                entry_vector=np.array([
                    entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z
                ]),
                exit_vector=np.array([
                    exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z
                ]),
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection,
                type=RoadOption.LANEFOLLOW
            )
    
    def _find_loose_ends(self):
        """
        处理图中的断头路（悬空端点）
        
        有些道路段在交叉口或道路尽头没有连接到其他路段
        这些端点需要延伸连接到有效节点
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            
            # 检查是否已经有连接
            if (road_id in self._road_id_to_edge and
                section_id in self._road_id_to_edge[road_id] and
                lane_id in self._road_id_to_edge[road_id][section_id]):
                pass
            else:
                # 添加断头路连接
                count_loose_ends += 1
                
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                
                n1 = self._id_map[exit_xyz]
                n2 = -1 * count_loose_ends  # 负数ID表示虚拟节点
                
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                
                # 延伸路径
                next_wp = end_wp.next(hop_resolution)
                path = []
                while (next_wp is not None and next_wp and
                       next_wp[0].road_id == road_id and
                       next_wp[0].section_id == section_id and
                       next_wp[0].lane_id == lane_id):
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                
                if path:
                    n2_xyz = (
                        path[-1].transform.location.x,
                        path[-1].transform.location.y,
                        path[-1].transform.location.z
                    )
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1,
                        path=path,
                        entry_waypoint=end_wp,
                        exit_waypoint=path[-1],
                        entry_vector=None,
                        exit_vector=None,
                        net_vector=None,
                        intersection=end_wp.is_junction,
                        type=RoadOption.LANEFOLLOW
                    )
    
    def _lane_change_link(self):
        """
        在图中添加换道连接（零成本边）
        
        当道路支持换道时，在拓扑图中添加连接相邻车道的边
        实现自动换道功能
        """
        for segment in self._topology:
            left_found, right_found = False, False
            
            for waypoint in segment['path']:
                if not segment['entry'].is_junction:
                    # 检查是否可以向右换道
                    if (waypoint.right_lane_marking and
                        waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and
                        not right_found):
                        
                        next_waypoint = waypoint.get_right_lane()
                        if (next_waypoint is not None and
                            next_waypoint.lane_type == carla.LaneType.Driving and
                            waypoint.road_id == next_waypoint.road_id):
                            
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']],
                                    next_segment[0],
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint,
                                    intersection=False,
                                    exit_vector=None,
                                    path=[],
                                    length=0,
                                    type=next_road_option,
                                    change_waypoint=next_waypoint
                                )
                                right_found = True
                    
                    # 检查是否可以向左换道
                    if (waypoint.left_lane_marking and
                        waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and
                        not left_found):
                        
                        next_waypoint = waypoint.get_left_lane()
                        if (next_waypoint is not None and
                            next_waypoint.lane_type == carla.LaneType.Driving and
                            waypoint.road_id == next_waypoint.road_id):
                            
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(next_waypoint.transform.location)
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']],
                                    next_segment[0],
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint,
                                    intersection=False,
                                    exit_vector=None,
                                    path=[],
                                    length=0,
                                    type=next_road_option,
                                    change_waypoint=next_waypoint
                                )
                                left_found = True
                
                if left_found and right_found:
                    break
    
    def _localize(self, location):
        """
        根据位置找到对应的图中的边
        
        参数：
            location: CARLA Location
        
        返回：
            edge: (n1, n2) 边的两个端点ID，或None
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass
        return edge
    
    def _distance_heuristic(self, n1, n2):
        """
        A*搜索的距离启发函数
        
        计算两节点之间的欧几里得距离作为启发值
        
        参数：
            n1, n2: 两个节点的ID
        
        返回：
            两节点之间的直线距离
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1 - l2)
    
    def _path_search(self, origin, destination):
        """
        使用A*算法搜索从起点到终点的最短路径
        
        参数：
            origin: 起点CARLA Location
            destination: 终点CARLA Location
        
        返回：
            route: 节点ID列表 [n1, n2, n3, ...]
        """
        # 定位起点和终点所在的边
        start = self._localize(origin)
        end = self._localize(destination)
        
        # A*搜索
        route = nx.astar_path(
            self._graph,
            source=start[0],
            target=end[0],
            heuristic=self._distance_heuristic,
            weight='length'
        )
        route.append(end[1])  # 添加终点节点
        
        return route
    
    def _successive_last_intersection_edge(self, index, route):
        """
        找到连续交叉口路段的最后一个边
        
        用于在交叉口内部正确计算转向决策
        
        参数：
            index: 当前在route中的索引
            route: 节点ID列表
        
        返回：
            (last_node, last_intersection_edge): 最后的节点和边
        """
        last_intersection_edge = None
        last_node = None
        
        for node1, node2 in [(route[i], route[i+1]) for i in range(index, len(route) - 1)]:
            candidate_edge = self._graph.edges[node1, node2]
            
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            
            if (candidate_edge['type'] == RoadOption.LANEFOLLOW and
                candidate_edge['intersection']):
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break
        
        return last_node, last_intersection_edge
    
    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        判断路口转向决策（RoadOption）
        
        基于进入交叉口前和出口的方向向量夹角判断左转/右转/直行
        
        参数：
            index: 当前在route中的索引
            route: 节点ID列表
            threshold: 直行判断阈值（默认35度）
        
        返回：
            RoadOption: LEFT/RIGHT/STRAIGHT/LANEFOLLOW
        """
        decision = None
        previous_node = route[index - 1]
        current_node = route[index]
        next_node = route[index + 1]
        next_edge = self._graph.edges[current_node, next_node]
        
        if index > 0:
            # 处理交叉口内部的连续决策
            if (self._previous_decision != RoadOption.VOID and
                self._intersection_end_node > 0 and
                self._intersection_end_node != previous_node and
                next_edge['type'] == RoadOption.LANEFOLLOW and
                next_edge['intersection']):
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                
                # 判断是否需要计算转向
                calculate_turn = (
                    current_edge['type'] == RoadOption.LANEFOLLOW and
                    not current_edge['intersection'] and
                    next_edge['type'] == RoadOption.LANEFOLLOW and
                    next_edge['intersection']
                )
                
                if calculate_turn:
                    # 找到交叉口出口边
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node
                    
                    if tail_edge is not None:
                        next_edge = tail_edge
                    
                    cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                    
                    if cv is None or nv is None:
                        return next_edge['type']
                    
                    # 计算转向角度
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        if select_edge['type'] == RoadOption.LANEFOLLOW:
                            if neighbor != route[index + 1]:
                                sv = select_edge['net_vector']
                                cross_list.append(np.cross(cv, sv)[2])
                    
                    next_cross = np.cross(cv, nv)[2]
                    deviation = math.acos(np.clip(
                        np.dot(cv, nv) / (np.linalg.norm(cv) * np.linalg.norm(nv)),
                        -1.0, 1.0
                    ))
                    
                    if not cross_list:
                        cross_list.append(0)
                    
                    # 根据叉积符号判断左右转
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']
        else:
            decision = next_edge['type']
        
        self._previous_decision = decision
        return decision
    
    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        """
        在路径点列表中找到距离车辆最近的点
        
        参数：
            current_waypoint: 当前Waypoint
            waypoint_list: Waypoint列表
        
        返回：
            closest_index: 最近点的索引
        """
        min_distance = float('inf')
        closest_index = -1
        
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location
            )
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index
