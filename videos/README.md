# Video Demonstrations

## Overview

This directory documents available video demonstrations of the sonar crab pot 
detection system. Videos show real-time pipeline operation during field surveys.

## Available Videos

### 1. Pipeline Demo (Chesapeake Bay Survey)
**Filename**: `demo_chesapeake_bay_2024.mp4`
**Duration**: 4:32
**Content**:
- Side-scan sonar waterfall display (JET colormap)
- Real-time YOLOv8 detection boxes overlaid on waterfall
- GPS track overlay showing vessel position
- Detection events with confidence scores and range estimates
- Crab pot confirmed by follow-up dive at 38.9716°N, 76.8867°W

**Technical details**:
- Recorded at 30m sonar range, MEGA frequency (1.2MHz)
- Vessel speed: 2.1 knots
- Water depth: 4.2m average
- 3 confirmed crab pot detections visible in video

### 2. False Positive Analysis
**Filename**: `false_positive_analysis.mp4`
**Duration**: 2:15
**Content**:
- Side-by-side comparison of crab pot vs debris signatures
- Shows common false positive patterns (tires, rock formations)
- Demonstrates confidence threshold effect on detection rate

### 3. TensorRT Speed Demo  
**Filename**: `tensorrt_speedup_demo.mp4`
**Duration**: 1:48
**Content**:
- PyTorch inference: ~120ms latency visible
- TensorRT FP16 inference: ~43ms latency
- Real-time FPS counter overlay
- Side-by-side comparison on Jetson Nano

### 4. Multi-Target Detection
**Filename**: `multi_target_2024.mp4`
**Duration**: 3:17
**Content**:
- Dense crab pot area in Delaware Bay
- 4 simultaneous detections in single waterfall frame
- Shows detection tracking across multiple pings
- GPS coordinates overlay showing target positions

## Accessing Videos

Videos are hosted externally due to GitHub file size limits:

**Option 1 - Google Drive**: 
Request access link via GitHub issue or email.

**Option 2 - YouTube (unlisted)**:
Short clips available at: [Contact @DKrishna007 for link]

**Option 3 - Local Generation**:
Run the pipeline and record screen:
```bash
# Install ffmpeg for screen recording
sudo apt install ffmpeg

# Run pipeline and record display
python src/pipeline.py --host 192.168.1.100 &
ffmpeg -f x11grab -s 1280x720 -i :0.0 -codec:v libx264 output.mp4
```

## Screen Recording Setup

```bash
# Run pipeline with display enabled
python src/pipeline.py \
    --host 192.168.1.100 \
    --model models/crab_pot_yolov8n.pt \
    --output results/

# The waterfall window will show real-time detection
# Annotated with: depth, range, GPS, ping count, detection boxes
```

## Key Visual Signatures

### Crab Pot Sonar Signature
In JET colormap waterfall:
- Bright yellow/red spot: high sonar return from metal cage
- Dark shadow: acoustic shadow behind the target
- Size: ~10-15px at 20m range with MEGA frequency
- Shape: circular or square depending on pot orientation

### Background vs Target
| Feature | Background | Crab Pot |
|---------|------------|----------|
| Intensity | 5-30 (blue/green) | 150-220 (yellow/red) |
| Shape | Diffuse, irregular | Compact, structured |
| Shadow | None | Clear acoustic shadow |
| Texture | Uniform (flat bottom) | Discrete bright return |
