#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机双目视觉避障系统
功能：检测障碍物距离和方位，提供避障决策
适用：无人机实时避障导航
"""

import cv2
import numpy as np
import sys
import os
import time
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 导入相机配置
sys.path.append(r'two_vision_calibration/calibration_code/calibration_results')
try:
    import camera_config as config
    print("✅ 成功导入相机配置")
    print(f"   基线距离: {config.BASELINE_MM:.2f}mm")
    print(f"   焦距: {config.FOCAL_LENGTH:.2f}pixels")
except ImportError as e:
    print("❌ 错误：无法导入相机配置文件")
    print("   请确保已运行 stereo_calibration.py 生成配置文件")
    sys.exit(1)


class ObstacleLevel(Enum):
    """障碍物威胁等级"""
    SAFE = "SAFE"           # 安全
    CAUTION = "CAUTION"     # 注意
    WARNING = "WARNING"     # 警告  
    DANGER = "DANGER"       # 危险
    CRITICAL = "CRITICAL"   # 极危险


class Direction(Enum):
    """飞行方向建议"""
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    BACK = "BACK"
    STOP = "STOP"


@dataclass
class ObstacleInfo:
    """障碍物信息"""
    distance: float         # 距离（mm）
    position_x: float       # X坐标（mm，相对相机中心）
    position_y: float       # Y坐标（mm，相对相机中心）
    threat_level: ObstacleLevel
    region: str            # 所属区域


@dataclass
class AvoidanceDecision:
    """避障决策"""
    safe_direction: Direction
    threat_level: ObstacleLevel
    closest_obstacle: float
    region_distances: Dict[str, float]
    action_confidence: float


class DroneObstacleAvoidance:
    """无人机障碍物避障系统"""
    
    def __init__(self):
        """初始化避障系统"""
        # 避障参数设置
        self.safe_distance = 2000      # 安全距离（mm）
        self.warning_distance = 1500   # 警告距离（mm）
        self.danger_distance = 1000    # 危险距离（mm）
        self.critical_distance = 500   # 极危险距离（mm）
        
        # 视野分区设置（将视野分成9个区域）
        self.regions = {
            'top_left': (0, 0, 1, 1),      # (x_start_ratio, y_start_ratio, x_end_ratio, y_end_ratio)
            'top_center': (1, 0, 2, 1),
            'top_right': (2, 0, 3, 1),
            'middle_left': (0, 1, 1, 2),
            'center': (1, 1, 2, 2),
            'middle_right': (2, 1, 3, 2),
            'bottom_left': (0, 2, 1, 3),
            'bottom_center': (1, 2, 2, 3),
            'bottom_right': (2, 2, 3, 3)
        }
        
        # 初始化立体视觉组件
        self._init_stereo_vision()
        
        print("🚁 无人机避障系统初始化完成")
        
    def _init_stereo_vision(self):
        """初始化立体视觉组件"""
        # 创建立体匹配器
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # 创建WLS滤波器
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_matcher)
        self.wls_filter.setLambda(80000)
        self.wls_filter.setSigmaColor(1.2)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
        
        # 创建校正映射表
        self._create_rectify_maps()
        
    def _create_rectify_maps(self):
        """创建图像校正映射表"""
        image_size = (config.CAMERA_WIDTH//2, config.CAMERA_HEIGHT)
        
        # 左相机校正映射
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            config.LEFT_CAMERA_MATRIX,
            config.LEFT_DIST_COEFFS,
            config.RECTIFY_R1,
            config.RECTIFY_P1,
            image_size,
            cv2.CV_16SC2
        )
        
        # 右相机校正映射
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            config.RIGHT_CAMERA_MATRIX,
            config.RIGHT_DIST_COEFFS,
            config.RECTIFY_R2,
            config.RECTIFY_P2,
            image_size,
            cv2.CV_16SC2
        )
        
    def compute_3d_points(self, disparity: np.ndarray) -> np.ndarray:
        """计算3D点云"""
        # 使用Q矩阵重投影到3D
        points_3d = cv2.reprojectImageTo3D(disparity, config.RECTIFY_Q)
        
        # 过滤无效点
        mask = disparity > 0
        points_3d[~mask] = [0, 0, 10000]  # 无效点设为远处
        
        return points_3d
        
    def analyze_regions(self, points_3d: np.ndarray) -> Dict[str, ObstacleInfo]:
        """分析各区域的障碍物情况"""
        h, w = points_3d.shape[:2]
        region_info = {}
        
        for region_name, (x_start, y_start, x_end, y_end) in self.regions.items():
            # 计算区域边界
            x1 = int(w * x_start / 3)
            y1 = int(h * y_start / 3)
            x2 = int(w * x_end / 3)
            y2 = int(h * y_end / 3)
            
            # 提取区域3D点
            region_points = points_3d[y1:y2, x1:x2]
            
            # 获取有效深度点
            valid_mask = (region_points[:, :, 2] > 0) & (region_points[:, :, 2] < 10000)
            
            if np.any(valid_mask):
                valid_points = region_points[valid_mask]
                
                # 找到最近的障碍物
                distances = valid_points[:, 2]  # Z坐标就是距离
                min_distance = np.min(distances)
                min_idx = np.argmin(distances)
                closest_point = valid_points[min_idx]
                
                # 确定威胁等级
                threat_level = self._get_threat_level(min_distance)
                
                region_info[region_name] = ObstacleInfo(
                    distance=min_distance,
                    position_x=closest_point[0],
                    position_y=closest_point[1],
                    threat_level=threat_level,
                    region=region_name
                )
            else:
                # 无有效障碍物
                region_info[region_name] = ObstacleInfo(
                    distance=float('inf'),
                    position_x=0,
                    position_y=0,
                    threat_level=ObstacleLevel.SAFE,
                    region=region_name
                )
        
        return region_info
    
    def _get_threat_level(self, distance: float) -> ObstacleLevel:
        """根据距离确定威胁等级"""
        if distance < self.critical_distance:
            return ObstacleLevel.CRITICAL
        elif distance < self.danger_distance:
            return ObstacleLevel.DANGER
        elif distance < self.warning_distance:
            return ObstacleLevel.WARNING
        elif distance < self.safe_distance:
            return ObstacleLevel.CAUTION
        else:
            return ObstacleLevel.SAFE
    
    def make_avoidance_decision(self, region_info: Dict[str, ObstacleInfo]) -> AvoidanceDecision:
        """做出避障决策"""
        # 提取各区域距离
        region_distances = {name: info.distance for name, info in region_info.items()}
        
        # 找到最近障碍物
        closest_distance = min(region_distances.values())
        overall_threat = self._get_threat_level(closest_distance)
        
        # 分析前方中央区域
        center_distance = region_distances['center']
        center_threat = self._get_threat_level(center_distance)
        
        # 决策逻辑
        if center_threat == ObstacleLevel.CRITICAL:
            # 紧急停止
            safe_direction = Direction.STOP
            confidence = 1.0
        elif center_threat in [ObstacleLevel.DANGER, ObstacleLevel.WARNING]:
            # 需要避障
            left_distance = min(region_distances['middle_left'], region_distances['top_left'], region_distances['bottom_left'])
            right_distance = min(region_distances['middle_right'], region_distances['top_right'], region_distances['bottom_right'])
            up_distance = min(region_distances['top_left'], region_distances['top_center'], region_distances['top_right'])
            down_distance = min(region_distances['bottom_left'], region_distances['bottom_center'], region_distances['bottom_right'])
            
            # 选择最安全的方向
            directions = {
                Direction.LEFT: left_distance,
                Direction.RIGHT: right_distance,
                Direction.UP: up_distance,
                Direction.DOWN: down_distance
            }
            
            safe_direction = max(directions, key=directions.get)
            confidence = min(1.0, directions[safe_direction] / self.safe_distance)
            
        elif center_threat == ObstacleLevel.CAUTION:
            # 谨慎前进
            safe_direction = Direction.FORWARD
            confidence = 0.6
        else:
            # 安全前进
            safe_direction = Direction.FORWARD
            confidence = 1.0
        
        return AvoidanceDecision(
            safe_direction=safe_direction,
            threat_level=overall_threat,
            closest_obstacle=closest_distance,
            region_distances=region_distances,
            action_confidence=confidence
        )
    
    def create_avoidance_visualization(self, image: np.ndarray, region_info: Dict[str, ObstacleInfo], 
                                     decision: AvoidanceDecision) -> np.ndarray:
        """创建避障可视化界面"""
        vis_img = image.copy()
        h, w = vis_img.shape[:2]
        
        # 绘制区域网格和障碍物信息
        for region_name, (x_start, y_start, x_end, y_end) in self.regions.items():
            x1 = int(w * x_start / 3)
            y1 = int(h * y_start / 3)
            x2 = int(w * x_end / 3)
            y2 = int(h * y_end / 3)
            
            info = region_info[region_name]
            
            # 根据威胁等级选择颜色
            color_map = {
                ObstacleLevel.SAFE: (0, 255, 0),      # 绿色
                ObstacleLevel.CAUTION: (0, 255, 255), # 黄色
                ObstacleLevel.WARNING: (0, 165, 255), # 橙色
                ObstacleLevel.DANGER: (0, 0, 255),    # 红色
                ObstacleLevel.CRITICAL: (255, 0, 255) # 紫色
            }
            
            color = color_map[info.threat_level]
            
            # 绘制区域边框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 显示距离信息
            if info.distance != float('inf'):
                distance_text = f"{info.distance:.0f}mm"
                cv2.putText(vis_img, distance_text, (x1 + 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 显示决策信息
        decision_y = 30
        cv2.putText(vis_img, f"Decision: {decision.safe_direction.value}", (10, decision_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        decision_y += 30
        cv2.putText(vis_img, f"Threat: {decision.threat_level.value}", (10, decision_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        decision_y += 30
        cv2.putText(vis_img, f"Closest: {decision.closest_obstacle:.0f}mm", (10, decision_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        decision_y += 30
        cv2.putText(vis_img, f"Confidence: {decision.action_confidence:.2f}", (10, decision_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return vis_img
    
    def process_frame(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[AvoidanceDecision, np.ndarray, np.ndarray]:
        """处理单帧图像，返回避障决策和可视化"""
        # 图像校正
        rectified_left = cv2.remap(left_img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        # 计算视差
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        
        disparity_left = self.stereo_matcher.compute(gray_left, gray_right)
        disparity_right = self.right_matcher.compute(gray_right, gray_left)
        
        disparity_left = disparity_left.astype(np.float32) / 16.0
        disparity_right = disparity_right.astype(np.float32) / 16.0
        
        # WLS滤波
        disparity = self.wls_filter.filter(disparity_left, gray_left, None, disparity_right)
        
        # 计算3D点云
        points_3d = self.compute_3d_points(disparity)
        
        # 分析各区域
        region_info = self.analyze_regions(points_3d)
        
        # 做出避障决策
        decision = self.make_avoidance_decision(region_info)
        
        # 创建可视化
        visualization = self.create_avoidance_visualization(rectified_left, region_info, decision)
        
        return decision, visualization, disparity
    
    def run_real_time_avoidance(self, camera_id: int = 20):
        """运行实时避障系统"""
        print("🚁 启动无人机避障系统...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            return False
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        print("✅ 避障系统运行中")
        print("📖 操作说明:")
        print("   - 绿色区域：安全")
        print("   - 黄色区域：注意")
        print("   - 橙色区域：警告")
        print("   - 红色区域：危险")
        print("   - 紫色区域：极危险")
        print("   - 按 'q'：退出")
        print("   - 按 'l'：保存避障日志")
        
        cv2.namedWindow('Drone Obstacle Avoidance', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        decisions_log = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                # 分离左右图像
                height, width = frame.shape[:2]
                left_img = frame[:, :width//2]
                right_img = frame[:, width//2:]
                
                # 处理避障
                decision, visualization, disparity = self.process_frame(left_img, right_img)
                
                # 记录决策日志
                log_entry = {
                    'frame': frame_count,
                    'timestamp': time.time(),
                    'decision': decision.safe_direction.value,
                    'threat_level': decision.threat_level.value,
                    'closest_obstacle': decision.closest_obstacle,
                    'confidence': decision.action_confidence
                }
                decisions_log.append(log_entry)
                
                # 创建深度图可视化
                depth_colored = cv2.applyColorMap(
                    cv2.convertScaleAbs(disparity, alpha=255/disparity.max()), 
                    cv2.COLORMAP_JET
                )
                
                # 组合显示
                combined_display = np.hstack([visualization, depth_colored])
                cv2.imshow('Drone Obstacle Avoidance', combined_display)
                
                frame_count += 1
                
                # 控制台输出决策
                if frame_count % 30 == 0:  # 每秒输出一次
                    print(f"🎯 Frame {frame_count}: {decision.safe_direction.value} | "
                          f"Threat: {decision.threat_level.value} | "
                          f"Closest: {decision.closest_obstacle:.0f}mm | "
                          f"Confidence: {decision.action_confidence:.2f}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    # 保存决策日志
                    log_filename = f"avoidance_log_{int(time.time())}.json"
                    with open(log_filename, 'w') as f:
                        json.dump(decisions_log, f, indent=2)
                    print(f"📝 避障日志已保存: {log_filename}")
                
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 避障系统已退出")
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("🚁 无人机双目视觉避障系统")
    print("实时检测障碍物方位和距离")
    print("=" * 60)
    
    try:
        avoidance_system = DroneObstacleAvoidance()
        avoidance_system.run_real_time_avoidance(config.CAMERA_INDEX)
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()