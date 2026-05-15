# Performance Evaluation: Sonar Crab Pot Detection

## Evaluation Overview

Evaluation conducted across multiple field deployments in Chesapeake Bay (MD) and 
Delaware Bay (DE) during 2024. Dataset: 2,847 annotated sonar frames from 23 survey 
transects totaling ~87 km.

## Detection Accuracy

### Overall Performance

| Metric | Value | Notes |
|--------|-------|-------|
| mAP@0.5 | 0.84 | Primary detection metric |
| mAP@0.5:0.95 | 0.71 | Stricter IoU threshold |
| Precision | 0.88 | True positive rate |
| Recall | 0.82 | Detection sensitivity |
| F1 Score | 0.85 | Harmonic mean |
| False Positive Rate | 11.2% | Per survey transect |
| False Negative Rate | 18.0% | Missed detections |

### Per-Class Performance

| Class | mAP@0.5 | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| crab_pot | 0.88 | 0.91 | 0.85 | 0.88 |
| debris | 0.72 | 0.76 | 0.71 | 0.73 |

### Performance by Range

| Range Zone | mAP@0.5 | Notes |
|------------|---------|-------|
| 0-10m | 0.91 | Near field, high resolution |
| 10-20m | 0.87 | Optimal detection zone |
| 20-30m | 0.78 | Reduced signal intensity |

### Performance by Water Depth

| Depth | mAP@0.5 | Notes |
|-------|---------|-------|
| 0-3m | 0.76 | Shallow, surface clutter |
| 3-8m | 0.88 | Optimal depth range |
| 8-15m | 0.82 | Deep, some signal attenuation |

## Latency Analysis

### Jetson Nano (PyTorch FP32)

| Component | Mean (ms) | Std (ms) | P95 (ms) |
|-----------|-----------|----------|----------|
| Packet capture | 0.8 | 0.2 | 1.2 |
| Waterfall update | 10.2 | 1.8 | 13.5 |
| YOLOv8n inference | 118.4 | 12.3 | 142.0 |
| CSV logging | 0.3 | 0.1 | 0.5 |
| **Total** | **129.7** | **13.5** | **157.2** |

### Jetson Nano (TensorRT FP16)

| Component | Mean (ms) | Std (ms) | P95 (ms) |
|-----------|-----------|----------|----------|
| Packet capture | 0.8 | 0.2 | 1.2 |
| Waterfall update | 10.2 | 1.8 | 13.5 |
| YOLOv8n inference | 42.6 | 4.1 | 51.3 |
| CSV logging | 0.3 | 0.1 | 0.5 |
| **Total** | **53.9** | **5.0** | **66.5** |

**Sonar frame rate**: 4.2 Hz average → 238ms between frames → pipeline runs well within budget

## Comparison with Baseline Methods

| Method | mAP@0.5 | Inference (ms) | Notes |
|--------|---------|----------------|-------|
| Manual review (human) | ~0.93 | N/A | Reference upper bound |
| Template matching (OpenCV) | 0.51 | 5ms | Classical baseline |
| YOLOv5n | 0.79 | 135ms | Previous model |
| **YOLOv8n (ours)** | **0.84** | **43ms** | Current deployment |
| YOLOv8s (larger) | 0.87 | 89ms | Not used (memory) |

## Field Validation

### Survey Statistics
- Total transects: 23
- Total frames: 2,847
- True crab pot detections: 1,423
- Detected by model: 1,165 (81.9% recall)
- False positives: 149 (11.4% of positive predictions)
- Ground truth confirmed via GPS re-visit with dive team

### Representative Results

| Survey | Transect Length | Pots Found (GT) | Pots Detected | Recall | FP |
|--------|----------------|-----------------|---------------|--------|-----|
| CB-2024-07 | 4.2km | 38 | 32 | 84.2% | 4 |
| CB-2024-09 | 6.8km | 61 | 51 | 83.6% | 8 |
| DB-2024-10 | 3.5km | 29 | 23 | 79.3% | 3 |
| CB-2024-11 | 7.1km | 74 | 62 | 83.8% | 9 |

## System Resource Usage (Jetson Nano)

| Resource | Usage |
|----------|-------|
| GPU Memory | 1.8GB / 4GB |
| CPU Load | ~35% (2 cores) |
| RAM | 2.1GB / 4GB |
| Power Draw | 8.2W average |
| Storage I/O | ~12 MB/min (CSV + optional frames) |

## Error Analysis

### Common False Positives
1. Circular debris (tires, crab pot floats without string) — 42%
2. Rock formations with similar sonar signature — 28%  
3. Shadow artifacts from steep bottom features — 18%
4. Other — 12%

### Common False Negatives
1. Occluded by heavy vegetation — 35%
2. Very small pots at far range (>25m) — 31%
3. Unusual pot orientation — 22%
4. Low sonar signal strength — 12%

## Conclusions

- **Primary accuracy target** (80%+ mAP@0.5): **Achieved** (84%)
- **Latency target** (<200ms end-to-end): **Achieved** (54ms TensorRT, 130ms PyTorch)
- **Platform requirement** (Jetson Nano deployable): **Achieved** (8.2W, 2.1GB RAM)
- **False positive rate** (<15%): **Achieved** (11.2%)
