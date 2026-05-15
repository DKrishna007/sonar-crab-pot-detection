# Sample Data

## Overview

This directory contains sample sonar data files for testing and demonstration.
Full datasets from field deployments are available upon request for research purposes.

## Available Sample Files

Due to file size constraints, raw sonar data is not stored in this repository.
Sample data is available through the following means:

### PCAP Captures (Wireshark format)
- `sample_chesapeake_bay_30m.pcap` — 5-minute capture, 30m range setting
- `sample_delaware_bay_20m.pcap` — 3-minute capture, 20m range setting
- Files contain ~2,400 UDP packets demonstrating the protocol structure

**Download**: Contact @DKrishna007 or email for access to anonymized field data.

**Size**: ~45MB per 5-minute capture

### Synthetic Test Data (Included)
The following synthetic data files are included for pipeline testing:

```
sample_data/
├── synthetic_ping_port.npy     # 512-sample port-side ping
├── synthetic_ping_stbd.npy     # 512-sample starboard-side ping
├── synthetic_target_close.npy  # Ping with simulated close-range target
├── synthetic_target_far.npy    # Ping with simulated far-range target
└── synthetic_sequence.npy      # 50-ping sequence for waterfall testing
```

## Data Format

### Raw UDP Packet Data
- Format: Binary UDP packets captured via Wireshark/tcpdump
- Header: 64-byte Humminbird proprietary format
- Data: uint8 sonar intensity values (0-255)
- Rate: ~4-6 Hz (depends on range setting)
- Size per ping: 512-1024 bytes (side-scan data)

### Reconstructed Numpy Arrays
- Shape: (N_pings, 512) — N pings, 512 samples per side
- dtype: uint8
- Range: [0, 255] — 0=minimum return, 255=maximum return
- Each ping: starboard side (port reversed separately)

## Annotation Format

Annotations follow YOLO format (class cx cy width height normalized):
```
# class center_x center_y width height (all normalized 0-1)
0 0.487 0.423 0.024 0.038   # crab_pot detection
1 0.213 0.568 0.031 0.045   # debris detection
```

## Sample PCAP Parsing

```python
from src.packet_capture import HumminbirdPacketCapture
import pyshark  # For offline PCAP reading

# Parse PCAP file offline
cap = pyshark.FileCapture('sample_data/sample_chesapeake_bay_30m.pcap',
                          display_filter='udp.port==5200')

for pkt in cap:
    raw_bytes = bytes.fromhex(pkt.data.data.replace(':', ''))
    # Process with existing parser...
```

## Generating Synthetic Data

```python
import numpy as np

# Generate a synthetic sonar ping with a target
n_samples = 512
ping = np.random.randint(5, 25, n_samples, dtype=np.uint8)  # background noise

# Add target at position 200 (about 12m range, 30m total)
target_pos = 200
ping[target_pos-10:target_pos+10] = np.random.randint(180, 220, 20)

# Save
np.save('sample_data/synthetic_target_close.npy', ping)
```

## Citation

If using this data for research, please cite:

```
Digamarthi, K. (2024). Real-time Crab Pot Detection using Side-Scan Sonar 
and YOLOv8. University of Delaware MS Robotics Research.
```
