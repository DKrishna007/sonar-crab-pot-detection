#!/usr/bin/env python3
"""
gnss_imu_fusion.py
==================
Fuses GNSS (GPS) and IMU/wheel-odometry data with an Extended Kalman
Filter to produce geo-referenced, drift-corrected pose estimates for
mapping submerged crab-pot detections.

State vector: [lat, lon, heading, v_forward, v_lateral]
Measurements:
  - GNSS fix  → [lat, lon]
    - IMU       → heading_rate
      - Odometry  → [v_forward, v_lateral]

      Author : Krishna Digamarthi  <shivasaikrishna23@gmail.com>
      Project: Sonar Perception – Crab Pot Detection (University of Delaware)
      """

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── Optional ROS 2 ────────────────────────────────────────────────────────────
try:
      import rclpy
      from rclpy.node import Node
      from sensor_msgs.msg import NavSatFix, Imu
      from nav_msgs.msg import Odometry
      from geometry_msgs.msg import PoseWithCovarianceStamped
      _ROS_AVAILABLE = True
except ImportError:
      _ROS_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0        # metres
DEG_TO_RAD     = math.pi / 180.0
RAD_TO_DEG     = 180.0 / math.pi


# ── Helpers ───────────────────────────────────────────────────────────────────

def lat_lon_to_metres(lat: float, lon: float,
                                             ref_lat: float, ref_lon: float) -> Tuple[float, float]:
                                                   """Convert lat/lon to local ENU metres relative to a reference origin."""
                                                   dlat = (lat - ref_lat) * DEG_TO_RAD
                                                   dlon = (lon - ref_lon) * DEG_TO_RAD
                                                   x = EARTH_RADIUS_M * dlon * math.cos(ref_lat * DEG_TO_RAD)
                                                   y = EARTH_RADIUS_M * dlat
                                                   return x, y   # East, North


def metres_to_lat_lon(x: float, y: float,
                                             ref_lat: float, ref_lon: float) -> Tuple[float, float]:
                                                   """Inverse of lat_lon_to_metres."""
                                                   dlat = y / EARTH_RADIUS_M
                                                   dlon = x / (EARTH_RADIUS_M * math.cos(ref_lat * DEG_TO_RAD))
                                                   return ref_lat + dlat * RAD_TO_DEG, ref_lon + dlon * RAD_TO_DEG


# ── EKF State ──────────────────────────────────────────────────────────────────

@dataclass
class EKFState:
      """
          EKF state for sonar-platform localisation.

              State:  [x_E, y_N, heading_rad, v_fwd, v_lat]
                          x_E   – East  displacement from origin (m)
                                      y_N   – North displacement from origin (m)
                                                  heading – vessel heading (rad, 0=North CW+)
                                                              v_fwd  – forward speed (m/s)
                                                                          v_lat  – lateral speed (m/s)
                                                                              """
      x:   np.ndarray = field(default_factory=lambda: np.zeros(5))
      P:   np.ndarray = field(default_factory=lambda: np.eye(5) * 1.0)
      ref_lat: Optional[float] = None
      ref_lon: Optional[float] = None
      last_t:  float = 0.0
      initialized: bool = False


class GNSSIMUFusion:
      """
          Extended Kalman Filter fusing GNSS and IMU/odometry for sonar platforms.

              Usage
                  -----
                      >>> fuse = GNSSIMUFusion()
                          >>> fuse.update_gnss(lat=38.897, lon=-77.036, t=time.time())
                              >>> fuse.update_imu(heading_rate_rads=0.01, t=time.time())
                                  >>> pose = fuse.get_pose()
                                      """

    # Process noise covariance
      Q_DIAG = np.array([0.01, 0.01, 0.005, 0.1, 0.1])
    # GNSS measurement noise (metres)
    R_GNSS = np.diag([2.0, 2.0])
    # IMU heading-rate noise (rad/s)
    R_IMU  = np.array([[0.01]])
    # Odometry velocity noise (m/s)
    R_ODOM = np.diag([0.05, 0.05])

    def __init__(self):
              self._state = EKFState()

    # ── Feed measurements ─────────────────────────────────────────────────────

    def update_gnss(self, lat: float, lon: float, t: float,
                                        h_acc_m: float = 3.0) -> None:
                                                  """Incorporate a GNSS fix."""
                                                  s = self._state
                                                  if not s.initialized:
                                                                s.ref_lat, s.ref_lon = lat, lon
                                                                s.x[:2] = [0.0, 0.0]
                                                                s.initialized = True
                                                                s.last_t = t
                                                                return

                                                  self._predict(t)
                                                  x_m, y_m = lat_lon_to_metres(lat, lon, s.ref_lat, s.ref_lon)
                                                  z = np.array([x_m, y_m])
                                                  H = np.zeros((2, 5));  H[0, 0] = 1.0;  H[1, 1] = 1.0
                                                  R = np.diag([h_acc_m**2, h_acc_m**2])
                                                  self._correct(z, H, R)

    def update_imu(self, heading_rate_rads: float, t: float) -> None:
              """Incorporate an IMU heading-rate measurement."""
              if not self._state.initialized:
                            return
                        self._predict(t)
        z = np.array([heading_rate_rads])
        # heading_rate ~ 0 in state, so measurement = heading_rate ≈ Δheading/dt
        H = np.zeros((1, 5));  H[0, 2] = 0.0   # not directly observable
        # Use pseudo-measurement: update heading
        self._state.x[2] += heading_rate_rads * max(t - self._state.last_t, 0.001)
        self._state.x[2] = (self._state.x[2] + math.pi) % (2 * math.pi) - math.pi

    def update_odometry(self, v_fwd: float, v_lat: float, t: float) -> None:
              """Incorporate velocity from wheel/DVL odometry."""
        if not self._state.initialized:
                      return
                  self._predict(t)
        z = np.array([v_fwd, v_lat])
        H = np.zeros((2, 5));  H[0, 3] = 1.0;  H[1, 4] = 1.0
        self._correct(z, H, self.R_ODOM)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_pose(self) -> Optional[dict]:
              """Return current pose estimate or None if not initialised."""
        s = self._state
        if not s.initialized:
                      return None
                  lat, lon = metres_to_lat_lon(s.x[0], s.x[1], s.ref_lat, s.ref_lon)
        return {
                      "lat":         lat,
                      "lon":         lon,
                      "x_m":         float(s.x[0]),
                      "y_m":         float(s.x[1]),
                      "heading_deg": float(s.x[2] * RAD_TO_DEG) % 360,
                      "v_fwd":       float(s.x[3]),
                      "v_lat":       float(s.x[4]),
                      "cov_pos_m":   float(math.sqrt((s.P[0, 0] + s.P[1, 1]) / 2.0)),
        }

    def geo_reference(self, pixel_x: float, pixel_y: float,
                                             sonar_range_m: float, sonar_width_px: int) -> Optional[dict]:
                                                       """
                                                               Project a pixel-space sonar detection to real-world lat/lon.

                                                                       Parameters
                                                                               ----------
                                                                                       pixel_x      : pixel column of detection centre
                                                                                               pixel_y      : pixel row  of detection centre (0 = near side)
                                                                                                       sonar_range_m: total sonar range in metres (swath half-width)
                                                                                                               sonar_width_px: image width in pixels
                                                                                                               
                                                                                                                       Returns
                                                                                                                               -------
                                                                                                                                       dict with keys lat, lon, range_m, bearing_deg
                                                                                                                                               """
                                                       pose = self.get_pose()
                                                       if pose is None:
                                                                     return None

                                                       # Lateral offset from nadir (metres)
                                                       lateral_m = (pixel_x / sonar_width_px - 0.5) * 2 * sonar_range_m
                                                       # Along-track: row index relative to image centre
                                                       # (positive = ahead of vessel)
                                                       range_m   = sonar_range_m * pixel_y / sonar_width_px

        heading_rad = pose["heading_deg"] * DEG_TO_RAD
        # Rotate offsets from sonar frame to ENU
        dx =  lateral_m * math.cos(heading_rad) + range_m * math.sin(heading_rad)
        dy = -lateral_m * math.sin(heading_rad) + range_m * math.cos(heading_rad)

        lat, lon = metres_to_lat_lon(
                      pose["x_m"] + dx, pose["y_m"] + dy,
                      self._state.ref_lat, self._state.ref_lon,
        )
        bearing = math.degrees(math.atan2(dx, dy)) % 360
        return {"lat": lat, "lon": lon,
                                "range_m": math.hypot(dx, dy),
                                "bearing_deg": bearing}

    # ── EKF internals ─────────────────────────────────────────────────────────

    def _predict(self, t: float) -> None:
              s = self._state
        dt = max(t - s.last_t, 0.0)
        s.last_t = t
        if dt <= 0:
                      return

        x, h = s.x, s.x[2]
        vf, vl = x[3], x[4]

        # State transition
        F = np.eye(5)
        F[0, 3] =  math.cos(h) * dt
        F[0, 4] = -math.sin(h) * dt
        F[1, 3] =  math.sin(h) * dt
        F[1, 4] =  math.cos(h) * dt

        # Predict state
        x_new = x.copy()
        x_new[0] += (vf * math.cos(h) - vl * math.sin(h)) * dt
        x_new[1] += (vf * math.sin(h) + vl * math.cos(h)) * dt
        s.x = x_new

        # Predict covariance
        Q = np.diag(self.Q_DIAG * dt)
        s.P = F @ s.P @ F.T + Q

    def _correct(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
              s = self._state
        y  = z - H @ s.x
        S  = H @ s.P @ H.T + R
        K  = s.P @ H.T @ np.linalg.inv(S)
        s.x = s.x + K @ y
        s.P = (np.eye(5) - K @ H) @ s.P


# ── Standalone demo ───────────────────────────────────────────────────────────

def _demo():
      """Simulate a vessel track and fuse synthetic GNSS + IMU data."""
    import random
    fuse = GNSSIMUFusion()
    t = time.time()

    # Simulate 30 seconds of data at 1 Hz
    lat, lon = 38.897, -77.036
    for i in range(30):
              t += 1.0
        # Add GPS noise
        fuse.update_gnss(lat + random.gauss(0, 3e-5),
                                                  lon + random.gauss(0, 3e-5), t)
        fuse.update_imu(heading_rate_rads=random.gauss(0, 0.02), t=t + 0.05)
        fuse.update_odometry(v_fwd=2.0 + random.gauss(0, 0.1),
                                                           v_lat=random.gauss(0, 0.05), t=t + 0.1)
        lat += 2.0 / EARTH_RADIUS_M * RAD_TO_DEG  # move north ~2 m/s

        pose = fuse.get_pose()
        print(f"[{i+1:02d}s] lat={pose['lat']:.6f}  lon={pose['lon']:.6f}"
                            f"  hdg={pose['heading_deg']:.1f}deg"
                            f"  cov={pose['cov_pos_m']:.2f}m")

    # Test geo-referencing a detection at pixel (256, 100) in a 512-px wide image
    geo = fuse.geo_reference(pixel_x=200, pixel_y=80,
                                                           sonar_range_m=50.0, sonar_width_px=512)
    print(f"\nGeo-referenced detection: lat={geo['lat']:.6f}  lon={geo['lon']:.6f}"
                    f"  range={geo['range_m']:.1f}m  bearing={geo['bearing_deg']:.1f}deg")


if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="GNSS/IMU EKF fusion demo")
    parser.add_argument("--demo", action="store_true", help="Run simulation demo")
    args = parser.parse_args()
    if args.demo:
              _demo()
else:
        print("Run with --demo to see the EKF fusion in action.")
        print("In ROS 2 mode, use: ros2 run sonar_perception gnss_imu_fusion")
