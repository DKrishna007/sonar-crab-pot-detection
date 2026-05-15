# Sonar Crab Pot Detection

**Real-time crab pot detection | Humminbird side-scan sonar + UDP reconstruction + YOLOv8 | 80-88% accuracy | Jetson Nano**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange.svg)](https://ros.org)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano-76b900.svg)](https://developer.nvidia.com/jetson-nano)

## Overview

End-to-end pipeline for real-time crab pot detection using Humminbird HELIX side-scan sonar. 
Captures raw UDP packets from the sonar, reconstructs sonar waterfall imagery, and runs 
YOLOv8 object detection to identify crab pot structures in the sonar imagery.

**Research context**: University of Delaware MS Robotics — underwater survey automation for 
commercial crab pot localization in the Chesapeake and Delaware Bays.

## Key Results

| Metric | Value |
|--------|-------|
| Detection accuracy (mAP@0.5) | **80-88%** |
| End-to-end latency (TensorRT) | **50-65ms** |
| End-to-end latency (PyTorch) | **120-180ms** |
| False positive rate | **< 12%** |
| Platform | NVIDIA Jetson Nano 4GB |
| Sonar | Humminbird HELIX 9 MEGA SI+ GPS G3 |
| Training data | 2,847 annotated sonar frames (field-collected) |

## System Architecture

```
Humminbird HELIX → UDP (port 5200) → Packet Capture → Fragment Reassembly
→ Waterfall Visualization (CLAHE + JET colormap) → YOLOv8n Detection
→ GPS-tagged CSV log / ROS 2 topics
```

## Repository Structure

```
sonar-crab-pot-detection/
├── src/
│   ├── packet_capture.py      # Humminbird UDP packet capture & reconstruction
│   ├── sonar_visualizer.py    # Side-scan waterfall display (OpenCV)
│   ├── crab_pot_detector.py   # YOLOv8 detection + coordinate conversion
│   └── pipeline.py            # End-to-end orchestration
├── config/
│   └── sonar_params.yaml      # Sonar & detection configuration
├── docs/
│   ├── architecture.md        # System architecture with ASCII diagrams
│   ├── evaluation.md          # Performance metrics and benchmarks
│   └── proof_guide.md         # Field validation evidence guide
├── results/
│   └── detections_sample.csv  # Sample detection log with GPS coordinates
├── sample_data/
│   └── README.md              # Sample data access instructions
├── videos/
│   └── README.md              # Video demonstration guide
├── models/                    # Model weights (not committed, see below)
└── requirements.txt
```

## Installation

```bash
# Clone repository
git clone https://github.com/DKrishna007/sonar-crab-pot-detection.git
cd sonar-crab-pot-detection

# Install Python dependencies
pip install -r requirements.txt

# For Jetson Nano, install PyTorch first:
# See: https://developer.nvidia.com/embedded/pytorch
```

## Quick Start

```bash
# Run full pipeline (real sonar hardware)
python src/pipeline.py --host 192.168.1.100 --model models/crab_pot_yolov8n.pt

# Test with mock detector (no hardware required)
python src/pipeline.py --host 127.0.0.1 --no-display

# Test individual components
python src/crab_pot_detector.py   # Detector demo
python src/sonar_visualizer.py    # Visualizer demo
```

## Hardware Setup

1. **Sonar**: Humminbird HELIX 9 MEGA SI+ GPS G3 (side-scan capable)
2. **Compute**: NVIDIA Jetson Nano 4GB (JetPack 4.6.4)
3. **Network**: Direct Ethernet between sonar hotspot and Jetson
4. **IP Config**: Sonar at 192.168.1.100, Jetson at 192.168.1.x

Enable sonar broadcasting: Humminbird menu → Network → Enable Ethernet Broadcast

## Detection Configuration

Key parameters in `config/sonar_params.yaml`:
- `sonar.side_scan.range_m`: Sonar range (10-80m)
- `detection.thresholds.confidence`: YOLOv8 confidence (default: 0.45)
- `detection.thresholds.iou`: NMS IoU threshold (default: 0.50)

## Model Weights

The trained YOLOv8n model (`crab_pot_yolov8n.pt`) is not included due to file size.

Training details:
- Base model: YOLOv8n (3.2M parameters, ultralytics)
- Training data: 2,847 labeled frames (2,562 train / 285 test)
- Augmentation: flip, rotation ±15°, mosaic, Gaussian noise
- Epochs: 150, batch size: 16, input: 640×640

Contact @DKrishna007 for model access or training data.

## Citation

```bibtex
@misc{digamarthi2024sonar,
  title={Real-time Crab Pot Detection with Side-Scan Sonar and YOLOv8},
  author={Krishna Digamarthi},
  year={2024},
  institution={University of Delaware, Department of Electrical and Computer Engineering},
  url={https://github.com/DKrishna007/sonar-crab-pot-detection}
}
```

## Author

**Krishna Digamarthi** — MS Robotics, University of Delaware  
Specialization: Perception, Autonomous Systems, Edge AI  
GitHub: [@DKrishna007](https://github.com/DKrishna007)
