# Sonar Perception – Crab Pot Detection

> **Research Project** | University of Delaware | Jan 2025 – Dec 2025
> > Graduate Robotics Research | Sonar Perception & Underwater Target Detection
> >
> > ---
> >
> > ## Overview
> >
> > End-to-end **side-scan sonar perception pipeline** for detecting submerged crab pots in open-water environments. Built and field-deployed using Python, OpenCV, and ROS 2 with GNSS/IMU sensor fusion for accurate underwater localization.
> >
> > ---
> >
> > ## Key Results
> >
> > | Metric | Value |
> > |---|---|
> > | Detection Accuracy | 80–88% across live field deployments |
> > | Localization Accuracy | 1.5–3.0 m for submerged underwater targets |
> > | Detection Stability Improvement | +15–25% via packet-stitching algorithm |
> > | False Positive Suppression | Contour + geometry-based signature filtering |
> >
> > ---
> >
> > ## System Architecture
> >
> > ```
> > Side-Scan Sonar → Packet Stitching → Preprocessing (OpenCV)
> >        ↓
> > Contour & Geometry Filtering → Signature Matching → Detection Output
> >        ↓
> > GNSS + IMU/Odometry Fusion → Geo-referenced Target Localization (ROS 2)
> > ```
> >
> > ---
> >
> > ## Features
> >
> > - **Packet-stitching algorithm** to reconstruct fragmented sonar streams, improving detection stability by 15–25%
> > - - **GNSS + IMU/Odometry sensor fusion** achieving 1.5–3.0 m localization accuracy for submerged targets
> >   - - **Contour and geometry-based signature filtering** to isolate high-confidence detections and suppress false positives
> >     - - **ROS 2 integration** with custom topics for sonar data ingestion and detection publishing
> >       - - **Field-deployable** pipeline tested across live open-water environments
> >        
> >         - ---
> >
> > ## Tech Stack
> >
> > | Category | Tools / Libraries |
> > |---|---|
> > | Middleware | ROS 2 (Humble) |
> > | Perception | Python, OpenCV, NumPy |
> > | Sensor Fusion | GNSS, IMU, Odometry, EKF |
> > | Sonar Hardware | Side-Scan Sonar (custom interface) |
> > | Localization | GNSS-referenced coordinate transforms |
> >
> > ---
> >
> > ## ROS 2 Package Structure
> >
> > ```
> > sonar_perception_ws/
> > ├── src/
> > │   ├── sonar_perception/
> > │   │   ├── sonar_perception/
> > │   │   │   ├── __init__.py
> > │   │   │   ├── packet_stitcher.py       # Sonar stream reconstruction
> > │   │   │   ├── sonar_preprocessor.py   # OpenCV-based image processing
> > │   │   │   ├── detector.py             # Contour & geometry filtering
> > │   │   │   └── gnss_imu_fusion.py      # Sensor fusion for localization
> > │   │   ├── launch/
> > │   │   │   └── sonar_pipeline.launch.py
> > │   │   ├── config/
> > │   │   │   └── detection_params.yaml
> > │   │   ├── package.xml
> > │   │   └── setup.py
> > ```
> >
> > ---
> >
> > ## Installation
> >
> > ```bash
> > # Clone the repository
> > git clone https://github.com/DKrishna007/sonar-perception-crab-pot-detection.git
> > cd sonar-perception-crab-pot-detection
> >
> > # Install ROS 2 dependencies
> > rosdep install --from-paths src --ignore-src -r -y
> >
> > # Build the workspace
> > colcon build --symlink-install
> >
> > # Source the workspace
> > source install/setup.bash
> > ```
> >
> > ---
> >
> > ## Usage
> >
> > ```bash
> > # Launch the full sonar perception pipeline
> > ros2 launch sonar_perception sonar_pipeline.launch.py
> >
> > # Run individual nodes
> > ros2 run sonar_perception packet_stitcher
> > ros2 run sonar_perception detector
> > ```
> >
> > ---
> >
> > ## Research Context
> >
> > This work was conducted as part of graduate research at the **University of Delaware** focused on underwater robotics and marine environment monitoring. The pipeline aims to assist fisheries management by enabling automated, GPS-referenced crab pot detection from survey vessels.
> >
> > ---
> >
> > ## Author
> >
> > **Krishna Digamarthi** | Graduate Robotics Researcher | University of Delaware
> > 📧 shivasaikrishna23@gmail.com | [GitHub](https://github.com/DKrishna007)
