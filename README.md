# StereoVision_Ranging

一个生产级别的立体视觉实时测距解决方案，支持**无人机避障**、**实时距离测量**、**机器人导航**等工业应用。开箱即用，开发者友好。

## ✨ **核心优势**：高性能 SGBM 算法 + WLS 滤波 + 实时处理 | 完整的标定→校正→测距工作流 | 针对 HBVCAM-W2307-2 双目摄像头优化

## 主要功能
- 相机标定（单应/棋盘格或圆点标定）
- 立体图像校正（rectification）
- 视差图生成（基于块匹配、Semi-Global Matching 等算法）
- 从视差图计算深度/距离（结合相机基线与焦距）
- 可视化视差与深度图
- 支持离线图像与实时相机流（视实现而定）

## 🏗️ 项目结构

```plaintext
StereoVision_Ranging/
├── README.md                              # 项目说明文档
├── avoidance_config.json                  # 无人机避障配置文件
├── real_time_distance_measurement.py      # ⭐ 实时测距主程序
├── drone_obstacle_avoidance.py            # ⭐ 无人机避障系统
├── universal_stereo_distance. py           # ⭐ 通用测距库
│
└── two_vision_calibration/                # 相机标定模块
    └── calibration_code/
        ├── stereo_calibration.py          # 标定程序
        └── calibration_results/
            └── camera_config.py           # 【标定输出】相机参数配置
```

### 文件说明

| 文件 | 说明 |
|------|------|
| `real_time_distance_measurement.py` | 实时双目测距程序，支持鼠标交互 |
| `drone_obstacle_avoidance.py` | 无人机实时避障系统 |
| `universal_stereo_distance.py` | 通用测距 API 库，可集成到其他项目 |
| `avoidance_config.json` | 避障系统配置（安全距离、告警参数等） |
| `two_vision_calibration/stereo_calibration.py` | 双目相机标定工具 |
| `two_vision_calibration/calibration_results/camera_config.py` | 标定后生成的相机参数（焦距、基线、畸变等） |


## 环境依赖
推荐使用 Python 3.8+，常见依赖包括：
- opencv-python (cv2) 或 opencv-contrib-python
- numpy
- matplotlib（用于可视化）
- scikit-image（可选）
- jupyter（可选，若要运行 notebooks）

安装示例（使用 pip）：
```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

pip install -r requirements.txt
# 或者手动安装
pip install opencv-contrib-python numpy matplotlib
```

## 快速开始（示例流程）

1. 相机标定（若已标定可跳过）
   - 准备棋盘格/圆点格图片（左右相机分别或同时拍摄）
   - 运行标定脚本生成相机内参与畸变参数，例如：
     ```bash
     python src/calibrate.py --images calib/images/ --pattern 9x6 --output calib/calib.yml
     ```
   - 输出：`calib/calib.yml`（包含相机矩阵和畸变系数，以及立体标定得到的 R、T 等）

2. 立体校正（Rectification）
   - 使用标定结果对左右图像进行校正：
     ```bash
     python src/rectify.py --left data/left.png --right data/right.png --calib calib/calib.yml --out_dir results/rectified
     ```
   - 输出：校正后的左右图像，便于后续视差计算。

3. 视差图计算
   - 使用块匹配（BM）或半全局匹配（SGBM）生成视差图：
     ```bash
     python src/compute_disparity.py --left results/rectified/left.png --right results/rectified/right.png --method sgbm --out results/disparity.png
     ```
   - 参数示例：最小/最大视差、窗口大小、惩罚项 P1/P2 等，可在脚本中调整以改善质量。

4. 从视差到深度/距离
   - 使用公式： depth = (focal_length * baseline) / disparity
     ```bash
     python src/estimate_depth.py --disparity results/disparity.png --calib calib/calib.yml --out results/depth.png
     ```
   - 输出：深度图（通常以灰度或伪彩图显示），也可以输出点云（XYZ）文件。

5. 可视化与评估
   - 可视化视差和深度，或将点云导入 MeshLab / CloudCompare 检查三维结构。

## 算法与原理要点（简述）
- 相机标定：通过多张棋盘格图片估计相机内参（焦距、主点）、畸变系数以及左右相机间的旋转 R 和平移 T。
- 立体校正：对图像进行极线对齐（epipolar alignment），使得对应点在同一行上，简化视差搜索为一维。
- 视差计算：在校正图像中搜索对应点的水平视差，常用方法有 StereoBM、StereoSGBM，或深度学习方法（若有）。
- 从视差到深度：依据基础几何关系使用相机焦距和基线（baseline）计算深度；需要处理无穷大视差（0/无效值）和噪声。

## 常见问题与调优建议
- 视差噪声大：尝试调大窗口、使用 SGBM、加入左右一致性检查（left-right check）。
- 距离不准确：检查标定准确性（标定板数量、拍摄角度），确认单位一致（焦距像素单位 vs 基线单位）。
- 近距离与远距离：算法参数对不同距离敏感，可能需要针对场景调参。

## 示例命令（汇总）
以下命令为示例，请根据仓库中实际脚本名替换：
```bash
# 标定
python src/calibrate.py --images calib/images/ --pattern 9x6 --output calib/calib.yml

# 校正
python src/rectify.py --left data/left.png --right data/right.png --calib calib/calib.yml --out_dir results/rectified

# 计算视差
python src/compute_disparity.py --left results/rectified/left.png --right results/rectified/right.png --method sgbm --out results/disparity.png

# 从视差估计深度
python src/estimate_depth.py --disparity results/disparity.png --calib calib/calib.yml --out results/depth.png
```

## 测试数据与评价
- 建议使用公开立体数据集（例如 Middlebury、KITTI）进行算法效果对比。
- 常用评估指标：视差错误率（% pixels > threshold）、平均绝对误差（MAE），深度 RMSE 等。

## 贡献
欢迎提交 issue 或 pull request 来修复 bug、增加新算法（深度学习模型、滤波器、后处理）、或扩展对新相机/传感器的支持。请遵循以下流程：
1. Fork 本仓库并创建 feature 分支
2. 提交代码并发起 Pull Request，描述所做修改与测试结果
3. 在问题中提供复现步骤与示例数据（若可能）

## 许可证
本项目请在此处补充许可证类型（例如 MIT、Apache 2.0 等）。如果未指定许可证，则默认不可商用或修改，请尽快补充 LICENSE 文件以明确许可。

## 联系与致谢
作者 / 维护者: zhiqianzheng  
Email / 联系方式：在仓库主页或个人资料中查看（如有需要可在 README 补充）

---

如果你希望我根据仓库中的实际脚本和文件生成更精确的 README（包含具体命令、参数和示例输出），请允许我查看仓库的文件结构或告诉我主要脚本名称与用途。我可以基于真实代码把 README 调整为可直接拷贝使用的版本。
