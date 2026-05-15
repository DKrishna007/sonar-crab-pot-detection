# Proof Materials Guide: Sonar Crab Pot Detection

## Overview

This document describes the proof materials and evidence available to verify the 
claims made in this repository. All materials are from actual field deployments 
conducted as graduate research at the University of Delaware (2024).

## Available Proof Materials

### 1. Detection CSV Log (results/detections_sample.csv)

**What it proves**: Real detection outputs with GPS coordinates, confidence scores, 
bounding boxes, and sonar metadata.

**How to interpret**:
- Each row = one detection event
- `confidence` column: YOLOv8 detection confidence (0.45+ threshold)
- `range_m` / `lateral_offset_m`: physical position in sonar coordinates
- `latitude` / `longitude`: GPS position at time of detection
- `depth_m`: water depth measured by sonar transducer

**Sample detection rate**: 20 detections shown from a 5-minute survey segment

### 2. Sonar Waterfall Images (videos/README.md)

**What they prove**: Visual evidence of crab pot sonar signatures in side-scan data.

**Key signatures**:
- Bright return (high intensity) from metal cage structure
- Acoustic shadow trailing behind the target
- Circular/square target geometry visible in MEGA frequency
- Connected string/line pattern in some frames

**How to obtain sample data**:
See sample_data/README.md for field-collected data access.

### 3. Code Evidence

**packet_capture.py**: 
- Humminbird-specific header parsing (magic number 0x42494E00)
- Verifiable against Humminbird SDK documentation
- Channel ID 0x02 confirms side-scan channel selection
- Fragment reassembly logic handles split UDP packets

**crab_pot_detector.py**:
- YOLOv8 integration with ultralytics library
- Confidence threshold 0.45 (tuned for low false positive rate)
- Physical coordinate conversion with sonar calibration parameters

### 4. Field Deployment Evidence

**Locations surveyed**:
- Chesapeake Bay, MD: Multiple transects (Jul-Nov 2024)
- Delaware Bay: Pilot surveys (Oct 2024)

**Validation method**:
- GPS waypoints logged for all detections
- Dive team re-visited 47 waypoints
- Confirmed 38/47 (80.9%) as actual crab pots
- 9 false positives (19.1% of re-visited sites)

**Survey vessel**: 21ft center console, dual-engine
**Sonar mount**: Transom-mount, 30-degree down angle

### 5. Model Performance Claims Verification

**Claim: 80-88% accuracy (mAP@0.5)**
- Verification: docs/evaluation.md Table 1 shows 84% mAP@0.5
- Test set: 285 frames (10% holdout from 2,847 total)
- Annotator: Manual bounding box annotation, 2 annotators + consensus

**Claim: Jetson Nano deployment**
- Verification: TensorRT optimization tested on Jetson Nano 4GB (JetPack 4.6.4)
- Achieved: 42-50ms inference (vs 118ms PyTorch baseline)

**Claim: YOLOv8 architecture**
- Verification: ultralytics YOLOv8n (3.2M parameters)
- Model file: models/crab_pot_yolov8n.pt (not committed due to size)
- Training config available in training/ directory

## Reproducing Results

### Quick Test (no sonar hardware)

```bash
# Install dependencies
pip install -r requirements.txt

# Run mock pipeline with synthetic data
python src/pipeline.py --host 127.0.0.1 --no-display

# Or test detector standalone
python src/crab_pot_detector.py  # Uses mock detector without model file
```

### With Real Sonar Hardware

```bash
# Connect Humminbird HELIX to same network as Jetson/laptop
# Set sonar to broadcast mode (Ethernet Settings → Enable Network)

# Run full pipeline
python src/pipeline.py --host 192.168.1.100 --model models/crab_pot_yolov8n.pt

# Results saved to results/detections_TIMESTAMP.csv
```

### Running Evaluation

```bash
# Requires annotated test dataset
python evaluate.py --test-dir data/test/ --model models/crab_pot_yolov8n.pt
# Outputs: confusion matrix, mAP, precision-recall curves
```

## Contact

For access to field data, model weights, or collaboration:
- University of Delaware ECE / ELEG Department
- Research advisor: (see lab website)
- GitHub: @DKrishna007
