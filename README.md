[Chinese Version](./README_ZH.md)
# 🎯 StereoVision_Ranging

**Production-grade Real-time Stereo Vision Ranging Solution**

Supports **UAV Obstacle Avoidance** | **Real-time Ranging** | **Robot Navigation** | **Industrial Applications**

[Quick Start](https://www.google.com/search?q=%23-quick-start) • [Features](https://www.google.com/search?q=%23-core-features) • [Documentation](https://www.google.com/search?q=%23-detailed-docs) • [Examples](https://www.google.com/search?q=%23-usage-examples)

---

## ✨ Highlights

* 🚀 **High-Performance Algorithm**: SGBM (Semi-Global Block Matching) + WLS Filtering, improving accuracy by 30-50%.
* ⚡ **Real-time Processing**: 25+ FPS @ 2560x720 resolution, meeting the latency requirements for drone obstacle avoidance.
* 🎯 **Complete Workflow**: Calibration → Rectification → Ranging → Decision Making, ready for deployment out-of-the-box.
* 🛠️ **Highly Configurable**: JSON-based parameter management—no code changes required for tuning.
* 📊 **Rich Visualization**: Real-time depth maps, distance heatmaps, and threat level indicators.
* 🔌 **Modular Design**: A universal library that can be easily integrated into other computer vision projects.
* 🎓 **Developer Friendly**: Detailed comments, comprehensive examples, and a robust calibration toolchain.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Stereo Image Acquisition                  │
│             (HBVCAM-W2307-2 / Universal Camera)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Camera Calibration Module                   │
│ • Checkerboard Detection • Intrinsic/Extrinsic • Stereo Calib│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Image Rectification                       │
│  • Epipolar Alignment  • Distortion Removal  • ROI Cropping  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Disparity Calculation (SGBM + WLS)               │
│ • Semi-Global Matching  • Edge Preservation  • Noise Filter │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Depth Estimation & 3D Point Cloud                │
│       depth = (focal_length × baseline) / disparity         │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│ Real-time Ranging│   │ UAV Avoidance    │
│ • Mouse Interaction│   │ • 9-Zone Analysis│
│ • Distance Display│   │ • Threat Eval    │
│ • Visualization  │   │ • Flight Decision│
└──────────────────┘   └──────────────────┘

```

---

## 🎯 Core Features

### 1️⃣ Calibration Toolkit

* Automatic checkerboard corner detection with sub-pixel accuracy.
* Supports Monocular and Stereo calibration.
* Multi-format output: JSON, NPZ, and Python Config.
* Calibration quality assessment reports and rectification previews.

### 2️⃣ Real-time Ranging

* Click-to-measure: Query distance at any point via mouse (5x5 Median Filtering).
* Real-time Depth Mapping (JET Color Map).
* Color-coded status indicators (Near/Mid/Far).
* Effective range filtering (500mm - 6000mm).

### 3️⃣ UAV Obstacle Avoidance

* **9-Zone Analysis**: Full field-of-view coverage.
* **5-Level Threat Assessment**: From "Safe" to "Critical".
* **Intelligent Decision Logic**: Commands for Forward/Backward/Left/Right/Up/Down/Stop.
* **Configurable Thresholds**: All parameters tuned via `avoidance_config.json`.

---

## 🚀 Quick Start

### Requirements

* **OS**: Windows / Linux / macOS
* **Python**: 3.8+
* **Hardware**: Stereo Camera (Recommended: HBVCAM-W2307-2)

### Installation

```bash
# Clone the repository
git clone https://github.com/zhiqianzheng/StereoVision_Ranging.git
cd StereoVision_Ranging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install opencv-contrib-python numpy matplotlib scikit-image

```

### Usage in 3 Steps

#### **Step 1: Calibration** (First-time use)

```bash
cd two_vision_calibration/calibration_code
# 1. Capture 30-40 pairs of images
python capture.py  # Press 's' to save, 'q' to quit

# 2. Run Calibration
python stereo_calibration.py

```

#### **Step 2: Real-time Ranging Test**

```bash
cd ../..
python real_time_distance_measurement.py

```

#### **Step 3: UAV Avoidance System**

```bash
python drone_obstacle_avoidance.py

```

---

## ⚙️ Configuration (`avoidance_config.json`)

### Distance Thresholds

```json
"distance_thresholds": {
  "safe_distance_mm": 2000,      // Green: Safe
  "warning_distance_mm": 1500,   // Yellow: Caution
  "danger_distance_mm": 1000,    // Orange: Warning
  "critical_distance_mm": 500    // Red: Critical Danger
}

```

---

## 📊 Performance Metrics

| Operation | CPU Latency | GPU Latency* | FPS (CPU) |
| --- | --- | --- | --- |
| Rectification | ~5ms | ~1ms | 200+ |
| SGBM Calculation | ~25ms | ~5ms | 40 |
| WLS Filtering | ~5ms | ~2ms | 200+ |
| **Total Pipeline** | **~40ms** | **~9ms** | **25 FPS** |

> *GPU acceleration requires `opencv-contrib-python` compiled with CUDA support.

---

## 🚁 UAV Avoidance Logic

### Field of View Partitioning (9 Zones)

```
┌──────────┬──────────┬──────────┐
│ Top Left │  Top Mid │ Top Right│  Independent distance
│   (1/9)  │   (1/9)  │   (1/9)  │  calculation per zone
├──────────┼──────────┼──────────┤
│ Mid Left │  CENTER  │ Mid Right│  Center Zone Weight: 1.5x
│   (1/9)  │   (1/9)  │   (1/9)  │  (Main Flight Direction)
├──────────┼──────────┼──────────┤
│ Bot Left │  Bot Mid │ Bot Right│  Top/Bot Weight: 0.8x
│   (1/9)  │   (1/9)  │   (1/9)  │
└──────────┴──────────┴──────────┘

```

---

## ❓ FAQ

**Q: Camera cannot open (Error 20).**
A: Check the camera index in the code. On Linux, ensure you have permissions: `sudo chmod 666 /dev/video*`.

**Q: High ranging error (> 10%).**
A: Ensure you have 30+ high-quality calibration images. Check if the `baseline_mm` in JSON matches your hardware.

**Q: High noise in disparity map.**
A: Increase `P1` and `P2` in the SGBM configuration to enhance smoothness, or increase `blockSize`.

---

## 🤝 Contribution

Contributions are what make the open-source community an amazing place to learn, inspire, and create.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Author

**zhiqianzheng** - [GitHub Profile](https://github.com/zhiqianzheng)

---

<div align="center">
If this project helped you, please give it a ⭐ Star!
</div>

---

这样你的项目看起来就非常国际化且专业了！需要我再针对某些具体的技术细节做微调吗？
