#!/usr/bin/env python3
  """
  detector.py
  ===========
  Contour- and geometry-based crab-pot signature detector for side-scan
  sonar waterfall images.

  Pipeline
  --------
  1. Adaptive thresholding / CLAHE enhancement
  2. Morphological noise removal
  3. Contour extraction
  4. Geometry filter  (area, aspect-ratio, solidity, extent)
  5. High-confidence detection output  →  ROS 2 topic + OpenCV visualisation

  Author : Krishna Digamarthi  <shivasaikrishna23@gmail.com>
  Project: Sonar Perception – Crab Pot Detection (University of Delaware)
    """

    import argparse
    import logging
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import List, Optional, Tuple

      import cv2
      import numpy as np

      logger = logging.getLogger(__name__)

      # ── Try to import ROS 2; fall back gracefully for standalone use ─────────────
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from geometry_msgs.msg import PointStamped
            from cv_bridge import CvBridge
            _ROS_AVAILABLE = True
        except ImportError:
            _ROS_AVAILABLE = False
            logger.warning("rclpy not found – running in standalone (no-ROS) mode.")


        # ── Detection data structure ─────────────────────────────────────────────────

        @dataclass
        class Detection:
            """Single detected crab-pot candidate on a sonar scan line."""
            x: int            # Column centre (pixels)
                  y: int            # Row centre (pixels)
                        width: int        # Bounding-box width
                              height: int       # Bounding-box height
                                    area: float       # Contour area (pixels²)
                                          confidence: float # Normalised score in [0, 1]
                                                contour: np.ndarray = None  # Raw OpenCV contour (optional)

                                                def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) bounding box."""
                  return (self.x - self.width // 2, self.y - self.height // 2,
                                          self.width, self.height)


          # ── Geometry filter ───────────────────────────────────────────────────────────

          class GeometryFilter:
              """
              Rejects contours that are unlikely to be crab-pot reflections based
              on simple shape statistics.

              Parameters  (all tunable via config YAML)
              ----------
              min_area, max_area      : pixel² bounds for contour area
                    min_aspect, max_aspect  : width/height ratio bounds
                          min_solidity            : contour_area / convex_hull_area
                                min_extent              : contour_area / bounding_rect_area
                                      """

                                      def __init__(
                                          self,
                                          min_area: float = 40.0,
                                          max_area: float = 4000.0,
                                          min_aspect: float = 0.3,
                                          max_aspect: float = 5.0,
                                          min_solidity: float = 0.45,
                                          min_extent: float = 0.25,
                                      ):
        self.min_area     = min_area
                  self.max_area     = max_area
                  self.min_aspect   = min_aspect
                  self.max_aspect   = max_aspect
                  self.min_solidity = min_solidity
                  self.min_extent   = min_extent

              def accept(self, contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
                  if not (self.min_area <= area <= self.max_area):
                      return False

                  x, y, w, h = cv2.boundingRect(contour)
                            if h == 0:
                                return False
                            aspect = w / h
                            if not (self.min_aspect <= aspect <= self.max_aspect):
                                return False

                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            if hull_area == 0:
                                return False
                            solidity = area / hull_area
                            if solidity < self.min_solidity:
                                return False

                            extent = area / (w * h)
                            if extent < self.min_extent:
                                return False

                            return True

                        def score(self, contour: np.ndarray) -> float:
        """Return a [0,1] confidence score for an already-accepted contour."""
                  area = cv2.contourArea(contour)
                  x, y, w, h = cv2.boundingRect(contour)
                            hull_area = cv2.contourArea(cv2.convexHull(contour)) or 1.0

                            # Prefer compact, moderately-sized blobs
                                      area_score    = 1.0 - abs(np.log(area / 200.0)) / 5.0
                                      solidity_score = cv2.contourArea(contour) / hull_area
                                      aspect_score  = 1.0 - abs(np.log(max(w, h) / max(min(w, h), 1))) / 3.0

                                      raw = (area_score + solidity_score + aspect_score) / 3.0
                                      return float(np.clip(raw, 0.0, 1.0))


                              # ── Main detector ─────────────────────────────────────────────────────────────

                              class CrabPotDetector:
                                  """
                                  Full sonar-image processing and crab-pot detection pipeline.

                                  Parameters
                                  ----------
                                  config : dict
                                            Override any default parameters.  Keys mirror the constructor args.

                                        Example
                                        -------
                                        >>> det = CrabPotDetector()
                                        >>> img = cv2.imread("waterfall.png", cv2.IMREAD_GRAYSCALE)
                                        >>> dets, vis = det.detect(img, return_vis=True)
                                              >>> cv2.imwrite("result.png", vis)
                                              """

                                              DEFAULT_CONFIG = {
                                                  "clahe_clip":       2.0,
                                                  "clahe_grid":       (8, 8),
                                                            "blur_ksize":       5,
                                                  "thresh_block":     21,
                                                  "thresh_c":         4,
                                                  "morph_ksize":      3,
                                                  "morph_open_iter":  1,
                                                  "morph_close_iter": 2,
                                                  "min_confidence":   0.35,
                                                  # Geometry filter defaults
                                                            "min_area":         40.0,
                                                  "max_area":         4000.0,
                                                  "min_aspect":       0.3,
                                                  "max_aspect":       5.0,
                                                  "min_solidity":     0.45,
                                                  "min_extent":       0.25,
                                        }

    def __init__(self, config: Optional[dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self._cfg = cfg

                  self._clahe = cv2.createCLAHE(
                      clipLimit=cfg["clahe_clip"],
                      tileGridSize=tuple(cfg["clahe_grid"]),
                  )
                  self._geo   = GeometryFilter(
                      min_area=cfg["min_area"],    max_area=cfg["max_area"],
                      min_aspect=cfg["min_aspect"], max_aspect=cfg["max_aspect"],
                      min_solidity=cfg["min_solidity"], min_extent=cfg["min_extent"],
                  )
                  self._morph_kernel = cv2.getStructuringElement(
                      cv2.MORPH_ELLIPSE, (cfg["morph_ksize"], cfg["morph_ksize"])
                  )

              # ── Public ───────────────────────────────────────────────────────────────

              def detect(
                  self,
                  image: np.ndarray,
                  return_vis: bool = False,
              ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
                  Run the full detection pipeline on a grayscale sonar image.

                  Parameters
                  ----------
                  image      : H×W uint8 grayscale sonar waterfall image.
                            return_vis : If True, also return an annotated BGR visualisation.

                                      Returns
                                      -------
                                      detections : List of Detection objects.
                                                vis_image  : Annotated BGR image (or None if return_vis=False).
                                                          """
                                                          if image.dtype != np.uint8:
                                                              image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                                                          # 1. Enhance contrast
                                                          enhanced = self._clahe.apply(image)

                                                          # 2. Blur
                                                          blurred = cv2.GaussianBlur(
                                                              enhanced,
                                                              (self._cfg["blur_ksize"], self._cfg["blur_ksize"]),
                                                              0,
                                                          )

                                                          # 3. Adaptive threshold
                                                          binary = cv2.adaptiveThreshold(
                                                              blurred,
                                                              255,
                                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                              cv2.THRESH_BINARY,
                                                              self._cfg["thresh_block"],
                                                              self._cfg["thresh_c"],
                                                          )

                                                          # 4. Morphological clean-up
                                                          binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                                                                                                      self._morph_kernel,
                                                                                                                      iterations=self._cfg["morph_open_iter"])
                                                          binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                                                                                                      self._morph_kernel,
                                                                                                                      iterations=self._cfg["morph_close_iter"])

                                                          # 5. Find contours
                                                          contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                                                                                                 cv2.CHAIN_APPROX_SIMPLE)

                                                                    # 6. Geometry filter + score
                                                                    detections: List[Detection] = []
                                                                              for cnt in contours:
                                                                                  if not self._geo.accept(cnt):
                                                                                      continue
                                                                                  conf = self._geo.score(cnt)
                                                                                  if conf < self._cfg["min_confidence"]:
                                                                                      continue
                                                                                  x, y, w, h = cv2.boundingRect(cnt)
                                                                                                det = Detection(
                                                                                                    x=x + w // 2,
                                                                                                    y=y + h // 2,
                                                                                                    width=w,
                                                                                                    height=h,
                                                                                                    area=float(cv2.contourArea(cnt)),
                                                                                                    confidence=conf,
                                                                                                    contour=cnt,
                                                                                                )
                                                                                                detections.append(det)

                                                                                            # Sort by confidence descending
                                                                                            detections.sort(key=lambda d: d.confidence, reverse=True)

                                                                                            vis = None
                                                                                            if return_vis:
                                                                                                vis = self._draw(image, binary, detections)

                                                                                            return detections, vis

                                                                                                  # ── Visualisation ─────────────────────────────────────────────────────────

                                                                                                  def _draw(
                                                                                                      self,
                                                                                                      original: np.ndarray,
                                                                                                      binary:   np.ndarray,
                                                                                                      detections: List[Detection],
                                                                                                  ) -> np.ndarray:
        """Produce a side-by-side annotated visualisation."""
                  orig_bgr   = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                  binary_bgr = cv2.cvtColor(binary,   cv2.COLOR_GRAY2BGR)

                  for i, det in enumerate(detections):
                      bx, by, bw, bh = det.bbox()
                                    colour = (0, 255, 0) if det.confidence >= 0.6 else (0, 200, 200)

                                    # Draw on original
                                    cv2.rectangle(orig_bgr, (bx, by), (bx + bw, by + bh), colour, 2)
                                    cv2.circle(orig_bgr, (det.x, det.y), 4, (0, 0, 255), -1)
                                    label = f"#{i+1} {det.confidence:.2f}"
                                    cv2.putText(orig_bgr, label, (bx, by - 6),
                                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

                                    # Draw contour on binary
                                    if det.contour is not None:
                                        cv2.drawContours(binary_bgr, [det.contour], -1, colour, 2)

                                info = f"Detections: {len(detections)}"
                                cv2.putText(orig_bgr, info, (8, 20),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                                return np.hstack([orig_bgr, binary_bgr])


                        # ── Optional ROS 2 node wrapper ───────────────────────────────────────────────

                        if _ROS_AVAILABLE:
                            class DetectorNode(Node):
                                """
                                ROS 2 node wrapping CrabPotDetector.

                                Subscriptions
                                -------------
                                /sonar/scan_image  (sensor_msgs/Image)  – grayscale waterfall

                                Publications
                                ------------
                                /sonar/detections  (geometry_msgs/PointStamped)  – one msg per detection
                                /sonar/debug_image (sensor_msgs/Image)            – annotated BGR
                                """

                                def __init__(self):
                                    super().__init__("crab_pot_detector")
                                    self.declare_parameter("min_confidence", 0.35)
                                    self.declare_parameter("min_area",       40.0)
                                    self.declare_parameter("max_area",       4000.0)

                                    cfg = {
                                        "min_confidence": self.get_parameter("min_confidence").value,
                                        "min_area":       self.get_parameter("min_area").value,
                                        "max_area":       self.get_parameter("max_area").value,
                      }
            self._det    = CrabPotDetector(config=cfg)
                          self._bridge = CvBridge()

                          self._sub = self.create_subscription(
                              Image, "/sonar/scan_image", self._image_cb, 5
                          )
                          self._pub_det = self.create_publisher(PointStamped, "/sonar/detections", 50)
                          self._pub_vis = self.create_publisher(Image, "/sonar/debug_image", 5)
                          self.get_logger().info("CrabPotDetector node ready.")

                      def _image_cb(self, msg: Image) -> None:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                          dets, vis = self._det.detect(frame, return_vis=True)

                                        for det in dets:
                                            pt = PointStamped()
                                            pt.header = msg.header
                                            pt.point.x = float(det.x)
                                            pt.point.y = float(det.y)
                                            pt.point.z = float(det.confidence)
                                            self._pub_det.publish(pt)

                                        if vis is not None:
                                            vis_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                                            vis_msg.header = msg.header
                                            self._pub_vis.publish(vis_msg)

                                        self.get_logger().debug("Detected %d crab pots", len(dets))


                            # ── Standalone CLI ────────────────────────────────────────────────────────────

                            def main():
                                parser = argparse.ArgumentParser(
                                    description="Run crab-pot detector on a sonar image file."
                                )
                                parser.add_argument("image", help="Path to a grayscale sonar image (PNG/TIFF/BMP)")
                                parser.add_argument("--output", "-o", default="result.png",
                                                                            help="Path to save annotated result (default: result.png)")
                                parser.add_argument("--min-conf",  type=float, default=0.35)
                                parser.add_argument("--min-area",  type=float, default=40.0)
                                parser.add_argument("--max-area",  type=float, default=4000.0)
                                parser.add_argument("--show",      action="store_true",
                                                                            help="Display result with cv2.imshow()")
                                args = parser.parse_args()

                                img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
                                if img is None:
                                    print(f"ERROR: Cannot read '{args.image}'")
                                    raise SystemExit(1)

                                cfg = {"min_confidence": args.min_conf,
                                                  "min_area": args.min_area,
                                                  "max_area": args.max_area}
    detector = CrabPotDetector(config=cfg)

          t0 = time.perf_counter()
          detections, vis = detector.detect(img, return_vis=True)
                elapsed = (time.perf_counter() - t0) * 1000

                print(f"Found {len(detections)} detection(s) in {elapsed:.1f} ms")
                for i, d in enumerate(detections):
                    print(f"  [{i+1}] centre=({d.x},{d.y})  size={d.width}x{d.height}"
                                        f"  area={d.area:.0f}  confidence={d.confidence:.3f}")

                cv2.imwrite(args.output, vis)
                print(f"Saved annotated result → {args.output}")

                if args.show:
                    cv2.imshow("Crab Pot Detections", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


            if __name__ == "__main__":
                main()
            
