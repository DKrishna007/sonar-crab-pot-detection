# System Architecture: Sonar Crab Pot Detection

## Overview

Real-time crab pot detection pipeline using Humminbird HELIX side-scan sonar,
UDP packet reconstruction, waterfall image generation, and YOLOv8 object detection
deployed on NVIDIA Jetson Nano for underwater survey operations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    VESSEL / SURVEY BOAT                          │
│                                                                   │
│  ┌─────────────────┐    UDP/Ethernet     ┌───────────────────┐  │
│  │  Humminbird     │ ─────────────────► │   Jetson Nano 4GB │  │
│  │  HELIX 9 MEGA   │  Port 5200          │                   │  │
│  │  SI GPS G3      │  ~5 Hz, 1.2MHz      │  ┌─────────────┐ │  │
│  │                 │                     │  │  Packet     │ │  │
│  │  Side-scan:     │                     │  │  Capture    │ │  │
│  │  ±30m range     │                     │  │  (UDP)      │ │  │
│  │  1024px/side    │                     │  └──────┬──────┘ │  │
│  └─────────────────┘                     │         │        │  │
│                                           │  ┌──────▼──────┐ │  │
│  ┌─────────────────┐                     │  │  Sonar      │ │  │
│  │  GPS/IMU        │                     │  │  Visualizer │ │  │
│  │  (via sonar)    │                     │  │  (Waterfall)│ │  │
│  │  Lat/Lon/Hdg    │                     │  └──────┬──────┘ │  │
│  └─────────────────┘                     │         │        │  │
│                                           │  ┌──────▼──────┐ │  │
│                                           │  │  YOLOv8n   │ │  │
│                                           │  │  Detector  │ │  │
│                                           │  │  ~45ms     │ │  │
│                                           │  └──────┬──────┘ │  │
│                                           │         │        │  │
│                                           │  ┌──────▼──────┐ │  │
│                                           │  │  Detection  │ │  │
│                                           │  │  Logger     │ │  │
│                                           │  │  (CSV/ROS2) │ │  │
│                                           │  └─────────────┘ │  │
│                                           └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Humminbird Packet Capture (src/packet_capture.py)

**Input**: UDP broadcast from Humminbird HELIX 9 MEGA SI+ GPS G3
**Protocol**: Proprietary binary UDP, port 5200
**Output**: Reconstructed SonarPacket objects

Key features:
- UDP socket capture with 1MB receive buffer
- 64-byte header parsing (magic, channel ID, ping number, GPS, depth)
- Fragment reassembly: large pings split across multiple UDP packets
- Side-scan channel filtering (channel ID = 0x02)
- Ping reconstruction buffer keyed by ping number

**Header structure (64 bytes)**:
```
Bytes  0- 3: Magic number (0x42494E00)
Bytes  4- 7: Channel ID (0x02 = side-scan)
Bytes  8-11: Ping number (sequential)
Bytes 12-15: Fragment offset
Bytes 16-19: Total ping size
Bytes 20-23: Timestamp (ms since epoch)
Bytes 24-27: Sonar range (cm)
Bytes 28-31: Speed × 10 (knots)
Bytes 32-35: Heading × 10 (degrees)
Bytes 36-39: Depth × 10 (meters)
Bytes 40-47: Latitude (double, decimal degrees)
Bytes 48-55: Longitude (double, decimal degrees)
Bytes 56-63: Reserved/checksum
```

### 2. Sonar Visualizer (src/sonar_visualizer.py)

**Input**: SonarPacket data arrays (uint8, 0-255)
**Output**: BGR waterfall image (1024×600 pixels)

Processing pipeline per ping:
1. Split data into port and starboard arrays
2. Time-Varying Gain (TVG) compensation: amplify far returns
3. Gamma correction (γ = 1.2)
4. Linear contrast enhancement (α = 1.5, β = -20)
5. Port side reversal: near returns at center, far at left edge
6. Water column blank zone: 20px center gap
7. CLAHE adaptive contrast equalization
8. Colormap application (JET colormap for false-color display)

**Memory buffer**: Rolling deque of 600 ping rows (~3.7MB)

### 3. YOLOv8 Detector (src/crab_pot_detector.py)

**Input**: BGR waterfall image (1024×600 pixels)
**Output**: DetectionResult with bounding boxes, confidence, physical coordinates

Model specifications:
- Architecture: YOLOv8n (nano) — optimized for Jetson
- Input size: 640×640 pixels
- Parameters: 3.2M
- Classes: {0: crab_pot, 1: debris}
- Training data: 2,847 annotated sonar frames (field-collected)
- Augmentations: horizontal flip, rotation ±15°, mosaic, Gaussian noise

Coordinate conversion:
- X-axis: lateral offset from vessel track (meters)
- Y-axis: slant range from transducer (meters)
- Calibration: range_m / image_half_width = m/pixel

### 4. Pipeline Orchestrator (src/pipeline.py)

Integration of all components with:
- Graceful shutdown handling (SIGINT/SIGTERM)
- CSV detection logging with GPS coordinates
- Optional ROS 2 topic publishing
- Statistics tracking (FPS, detection rate)

## Data Flow

```
Humminbird Sonar → UDP Packets → Fragment Reassembly → SonarPacket
SonarPacket → Port/Starboard Split → TVG → Gamma → Contrast
→ Waterfall Buffer (600 lines) → CLAHE → JET Colormap
→ BGR Image → YOLOv8n Inference → Bounding Boxes
→ Coordinate Conversion → Detection CSV Log
```

## Performance

| Component | Time (Jetson Nano) |
|-----------|-------------------|
| UDP Capture | < 1ms |
| Waterfall Update | 8-12ms |
| YOLOv8n Inference | 40-50ms (TensorRT) / 120ms (PyTorch) |
| CSV Logging | < 1ms |
| **Total Latency** | **50-65ms (TensorRT)** |

**End-to-end**: 80-120ms (including sonar acquisition at ~4-6 Hz)

## Deployment Configuration

- Platform: NVIDIA Jetson Nano 4GB (Jetpack 4.6.4)
- OS: Ubuntu 18.04 (L4T R32.7.4)
- Framework: PyTorch 1.11.0, TorchVision 0.12.0
- Python: 3.8.10
- Storage: 64GB SD card (results logging)
- Network: Direct Ethernet to Humminbird hotspot
