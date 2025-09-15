#!/usr/bin/env python3
"""
双目摄像头标定程序
功能：对采集的左右图像对进行双目标定，生成标定结果文件
适用于：HBVCAM-W2307-2双目摄像头，6米测距应用
作者：Claude AI Assistant
版本：1.0
"""

import cv2
import numpy as np
import os
import glob
import json
import time
from datetime import datetime
# import matplotlib.pyplot as plt


class StereoCalibrator:
    """双目摄像头标定器"""
    
    def __init__(self, chessboard_size=(11, 8), square_size=40.0):
        """
        初始化标定器
        
        参数：
        - chessboard_size: 棋盘格内部角点数量 (width, height)
        - square_size: 棋盘格方格实际尺寸（毫米）
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size  # 40mm方格
        
        # 生成棋盘格的3D坐标点
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 存储所有图像的角点
        self.obj_points = []    # 3D点
        self.left_img_points = []   # 左图像2D点
        self.right_img_points = []  # 右图像2D点
        
        # 标定结果
        self.left_camera_matrix = None
        self.left_dist_coeffs = None
        self.right_camera_matrix = None
        self.right_dist_coeffs = None
        self.R = None  # 旋转矩阵
        self.T = None  # 平移向量
        self.E = None  # 本质矩阵
        self.F = None  # 基础矩阵
        
        # 立体矫正结果
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi1 = None
        self.roi2 = None
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None
        
    def load_image_pairs(self, left_dir, right_dir):
        """
        加载左右图像对
        
        参数：
        - left_dir: 左图像目录
        - right_dir: 右图像目录
        
        返回：
        - 成功加载的图像对数量
        """
        print("🔍 加载图像对...")
        
        left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
        right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))
        
        if len(left_images) != len(right_images):
            print(f"❌ 错误：左图像{len(left_images)}张，右图像{len(right_images)}张，数量不匹配！")
            return 0
            
        print(f"📁 找到 {len(left_images)} 对图像")
        
        successful_pairs = 0
        
        for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            print(f"处理第 {i+1}/{len(left_images)} 对图像...", end=' ')
            
            # 读取图像
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if left_img is None or right_img is None:
                print("❌ 读取失败")
                continue
            
            # 转换为灰度图
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 检测棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_left and ret_right:
                # 亚像素精度优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
                
                # 存储角点
                self.obj_points.append(self.objp)
                self.left_img_points.append(corners_left)
                self.right_img_points.append(corners_right)
                
                successful_pairs += 1
                print("✅ 成功")
            else:
                print(f"❌ 棋盘格检测失败 (L:{ret_left}, R:{ret_right})")
        
        print(f"\n📊 总结: {successful_pairs}/{len(left_images)} 对图像成功处理")
        return successful_pairs
    
    def calibrate_single_camera(self, img_points, img_size, camera_name):
        """标定单个相机"""
        print(f"🔧 标定{camera_name}相机...")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, img_points, img_size, None, None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO
        )
        
        if ret:
            print(f"✅ {camera_name}相机标定成功，重投影误差: {ret:.4f} 像素")
            
            # 显示相机参数
            print(f"📷 {camera_name}相机内参矩阵:")
            print(f"   fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
            print(f"   cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
            print(f"🔍 {camera_name}畸变系数:")
            print(f"   k1={dist_coeffs[0,0]:.6f}, k2={dist_coeffs[0,1]:.6f}")
            print(f"   p1={dist_coeffs[0,2]:.6f}, p2={dist_coeffs[0,3]:.6f}")
            print(f"   k3={dist_coeffs[0,4]:.6f}")
        else:
            print(f"❌ {camera_name}相机标定失败！")
        
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs
    
    def calibrate_stereo(self, img_size):
        """双目标定"""
        print("🎯 开始双目标定...")
        
        # 首先分别标定左右相机
        ret_left, self.left_camera_matrix, self.left_dist_coeffs, _, _ = \
            self.calibrate_single_camera(self.left_img_points, img_size, "左")
            
        ret_right, self.right_camera_matrix, self.right_dist_coeffs, _, _ = \
            self.calibrate_single_camera(self.right_img_points, img_size, "右")
        
        if not ret_left or not ret_right:
            print("❌ 单相机标定失败，无法进行双目标定！")
            return False
        
        print("🔄 执行双目标定...")
        
        # 双目标定
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.obj_points,
            self.left_img_points,
            self.right_img_points,
            self.left_camera_matrix,
            self.left_dist_coeffs,
            self.right_camera_matrix,
            self.right_dist_coeffs,
            img_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        if ret:
            print(f"✅ 双目标定成功，重投影误差: {ret:.4f} 像素")
            
            # 计算基线距离
            baseline = np.linalg.norm(self.T)  # 基线距离(mm)
            print(f"📏 基线距离: {baseline:.2f} mm")
            
            # 显示外参
            print("🔄 旋转矩阵 R:")
            print(self.R)
            print("📐 平移向量 T (mm):")
            print(self.T.flatten())
            
            return True
        else:
            print("❌ 双目标定失败！")
            return False
    
    def stereo_rectify(self, img_size):
        """立体矫正"""
        print("📐 计算立体矫正映射...")
        
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            img_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9  # 保留更多图像内容
        )
        
        # 生成矫正映射表
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_dist_coeffs, self.R1, self.P1, img_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_dist_coeffs, self.R2, self.P2, img_size, cv2.CV_32FC1)
        
        print("✅ 立体矫正映射计算完成")
        
        # 计算测距范围
        focal_length = self.P1[0, 0]  # 矫正后的焦距
        baseline = np.linalg.norm(self.T)  # 基线距离
        
        print(f"🎯 测距参数:")
        print(f"   矫正后焦距: {focal_length:.2f} 像素")
        print(f"   基线距离: {baseline:.2f} mm")
        print(f"   最小视差: 1 像素 -> 最大测距: {focal_length * baseline / 1000:.1f} 米")
        print(f"   推荐视差: 10 像素 -> 推荐最大测距: {focal_length * baseline / 10000:.1f} 米")
    
    def save_calibration_results(self, output_dir="calibration_results"):
        """保存标定结果"""
        print(f"💾 保存标定结果到 {output_dir}/")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备保存数据
        calibration_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chessboard_size": self.chessboard_size,
            "square_size_mm": self.square_size,
            "image_pairs_used": len(self.obj_points),
            
            "left_camera_matrix": self.left_camera_matrix.tolist(),
            "left_dist_coeffs": self.left_dist_coeffs.tolist(),
            "right_camera_matrix": self.right_camera_matrix.tolist(),
            "right_dist_coeffs": self.right_dist_coeffs.tolist(),
            
            "rotation_matrix": self.R.tolist(),
            "translation_vector": self.T.tolist(),
            "essential_matrix": self.E.tolist(),
            "fundamental_matrix": self.F.tolist(),
            
            "rectify_R1": self.R1.tolist(),
            "rectify_R2": self.R2.tolist(),
            "rectify_P1": self.P1.tolist(),
            "rectify_P2": self.P2.tolist(),
            "rectify_Q": self.Q.tolist(),
            "roi1": self.roi1,
            "roi2": self.roi2,
            
            "baseline_mm": float(np.linalg.norm(self.T)),
            "focal_length_rectified": float(self.P1[0, 0]),
        }
        
        # 保存JSON文件
        json_path = os.path.join(output_dir, "stereo_calibration.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        # 保存numpy文件（用于Python程序直接加载）
        npz_path = os.path.join(output_dir, "stereo_calibration.npz")
        np.savez(npz_path,
                 left_camera_matrix=self.left_camera_matrix,
                 left_dist_coeffs=self.left_dist_coeffs,
                 right_camera_matrix=self.right_camera_matrix,
                 right_dist_coeffs=self.right_dist_coeffs,
                 R=self.R, T=self.T, E=self.E, F=self.F,
                 R1=self.R1, R2=self.R2, P1=self.P1, P2=self.P2, Q=self.Q,
                 map1x=self.map1x, map1y=self.map1y,
                 map2x=self.map2x, map2y=self.map2y)
        
        # 生成camera_config.py文件
        config_py_path = self._generate_camera_config_py(output_dir, calibration_data)
        
        print(f"✅ 标定结果已保存:")
        print(f"   📄 {json_path} (人类可读)")
        print(f"   📦 {npz_path} (程序加载)")
        print(f"   🐍 {config_py_path} (Python配置文件)")
        
        return json_path, npz_path, config_py_path
    
    def _generate_camera_config_py(self, output_dir, calibration_data):
        """生成camera_config.py配置文件"""
        config_py_path = os.path.join(output_dir, "camera_config.py")
        
        config_content = f'''#!/usr/bin/env python3
"""
双目摄像头标定配置文件
自动生成时间: {calibration_data["timestamp"]}
摄像头型号: HBVCAM-W2307-2
基线距离: {calibration_data["baseline_mm"]:.2f} mm
焦距: {calibration_data["focal_length_rectified"]:.2f} 像素
使用图像对: {calibration_data["image_pairs_used"]} 对
"""

import numpy as np

# 基本标定参数
CHESSBOARD_SIZE = {calibration_data["chessboard_size"]}
SQUARE_SIZE_MM = {calibration_data["square_size_mm"]}
BASELINE_MM = {calibration_data["baseline_mm"]}
FOCAL_LENGTH = {calibration_data["focal_length_rectified"]}

# 左摄像头参数
LEFT_CAMERA_MATRIX = np.array({calibration_data["left_camera_matrix"]})
LEFT_DIST_COEFFS = np.array({calibration_data["left_dist_coeffs"]})

# 右摄像头参数  
RIGHT_CAMERA_MATRIX = np.array({calibration_data["right_camera_matrix"]})
RIGHT_DIST_COEFFS = np.array({calibration_data["right_dist_coeffs"]})

# 双目关系参数
ROTATION_MATRIX = np.array({calibration_data["rotation_matrix"]})
TRANSLATION_VECTOR = np.array({calibration_data["translation_vector"]})
ESSENTIAL_MATRIX = np.array({calibration_data["essential_matrix"]})
FUNDAMENTAL_MATRIX = np.array({calibration_data["fundamental_matrix"]})

# 立体矫正参数
RECTIFY_R1 = np.array({calibration_data["rectify_R1"]})
RECTIFY_R2 = np.array({calibration_data["rectify_R2"]})
RECTIFY_P1 = np.array({calibration_data["rectify_P1"]})
RECTIFY_P2 = np.array({calibration_data["rectify_P2"]})
RECTIFY_Q = np.array({calibration_data["rectify_Q"]})

# ROI区域
ROI_LEFT = {calibration_data["roi1"]}
ROI_RIGHT = {calibration_data["roi2"]}

def get_distance_from_disparity(disparity):
    """
    根据视差计算距离
    
    参数:
        disparity: 视差值（像素）
        
    返回:
        distance: 距离（毫米）
    """
    if disparity <= 0:
        return float('inf')
    
    return (FOCAL_LENGTH * BASELINE_MM) / disparity

def get_max_distance():
    """获取最大有效测距距离（以1像素视差为基准）"""
    return get_distance_from_disparity(1.0)

def get_recommended_max_distance():
    """获取推荐最大测距距离（以10像素视差为基准）"""
    return get_distance_from_disparity(10.0)

# 摄像头配置
CAMERA_INDEX = 20  # HBVCAM-W2307-2摄像头索引
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# 测距配置
MIN_DISPARITY = 1    # 最小视差（像素）
MAX_DISPARITY = 128  # 最大视差（像素）
DISPARITY_SEARCH_RANGE = 64  # 视差搜索范围

# 距离测量精度配置
DISTANCE_FILTER_WINDOW = 5  # 距离滤波窗口大小
MIN_VALID_DISTANCE_MM = 500   # 最小有效距离（毫米）
MAX_VALID_DISTANCE_MM = 6000  # 最大有效距离（毫米）

print(f"摄像头配置已加载:")
print(f"  基线距离: {{BASELINE_MM:.1f}} mm")
print(f"  焦距: {{FOCAL_LENGTH:.1f}} 像素")
print(f"  最大理论测距: {{get_max_distance()/1000:.1f}} 米")
print(f"  推荐最大测距: {{get_recommended_max_distance()/1000:.1f}} 米")
'''
        
        with open(config_py_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return config_py_path
    
    def test_rectification(self, left_dir, right_dir, output_dir="calibration_results"):
        """测试立体矫正效果"""
        print("🔍 测试立体矫正效果...")
        
        test_images = glob.glob(os.path.join(left_dir, "*.png"))[:3]  # 测试前3对
        
        for i, left_path in enumerate(test_images):
            right_path = left_path.replace("left", "right")
            
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if left_img is None or right_img is None:
                continue
            
            # 应用立体矫正
            left_rectified = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)
            
            # 创建并排显示
            combined = np.hstack((left_rectified, right_rectified))
            
            # 绘制水平线辅助对齐检查
            h = combined.shape[0]
            for y in range(0, h, 50):
                cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
            
            # 保存测试图像
            test_path = os.path.join(output_dir, f"rectification_test_{i+1}.jpg")
            cv2.imwrite(test_path, combined)
            print(f"📸 矫正测试图 {i+1} 已保存: {test_path}")
        
        print("✅ 立体矫正测试完成，请检查绿线是否对齐")


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 双目摄像头标定程序")
    print("适用于：HBVCAM-W2307-2 双目摄像头")
    print("功能：6米距离测量标定")
    print("=" * 60)
    
    # 初始化标定器
    # 棋盘格规格：12x9外部格子 = 11x8内部角点
    # 方格大小：40mm (根据你的实际棋盘格调整)
    calibrator = StereoCalibrator(chessboard_size=(11, 8), square_size=40.0)
    
    # 图像目录
    left_dir = "left"
    right_dir = "right"
    
    # 检查目录是否存在
    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        print(f"❌ 错误：找不到图像目录 {left_dir} 或 {right_dir}")
        return
    
    # 加载图像对
    num_pairs = calibrator.load_image_pairs(left_dir, right_dir)
    
    if num_pairs < 10:
        print(f"❌ 错误：有效图像对太少 ({num_pairs})，建议至少10对")
        return
    
    # 获取图像尺寸
    sample_img = cv2.imread(glob.glob(os.path.join(left_dir, "*.png"))[0])
    img_size = (sample_img.shape[1], sample_img.shape[0])
    print(f"📐 图像尺寸: {img_size[0]} x {img_size[1]}")
    
    # 执行标定
    if calibrator.calibrate_stereo(img_size):
        # 立体矫正
        calibrator.stereo_rectify(img_size)
        
        # 保存结果
        calibrator.save_calibration_results()
        
        # 测试矫正效果
        calibrator.test_rectification(left_dir, right_dir)
        
        print("\n🎉 双目标定完成！")
        print("📝 下一步：")
        print("   1. 检查 calibration_results/ 目录中的结果文件")
        print("   2. 查看 rectification_test_*.jpg 验证矫正效果")
        print("   3. 如果效果满意，可以开始实现距离测量功能")
        
    else:
        print("❌ 标定失败，请检查图像质量和棋盘格检测")


if __name__ == "__main__":
    main()
