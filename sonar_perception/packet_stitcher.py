#!/usr/bin/env python3
"""
packet_stitcher.py
==================
Reconstructs fragmented side-scan sonar UDP/serial data streams into
complete scan lines. Handles out-of-order packets, packet loss, and
stream re-synchronisation.

Author : Krishna Digamarthi  <shivasaikrishna23@gmail.com>
Project: Sonar Perception – Crab Pot Detection (University of Delaware)
"""

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SonarPacket:
      """Single UDP/serial sonar data packet."""
      line_id: int          # Scan-line index (wraps at 65535)
    fragment_id: int      # Fragment index within the scan line
    total_fragments: int  # Total number of fragments for this line
    timestamp: float      # Receive timestamp (time.time())
    payload: bytes        # Raw intensity bytes for this fragment


@dataclass
class ScanLineBuffer:
      """Accumulates fragments for one complete scan line."""
      line_id: int
      total_fragments: int
      fragments: Dict[int, bytes] = field(default_factory=dict)
      first_seen: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
      def add_fragment(self, frag_id: int, payload: bytes) -> None:
                self.fragments[frag_id] = payload

      def is_complete(self) -> bool:
                return len(self.fragments) == self.total_fragments

      def assemble(self) -> np.ndarray:
                """Concatenate fragments in order and return as uint8 array."""
                ordered = b"".join(self.fragments[i] for i in sorted(self.fragments))
                return np.frombuffer(ordered, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core stitcher logic
# ---------------------------------------------------------------------------

class PacketStitcher:
      """
          Reconstructs complete sonar scan lines from fragmented packets.

              Parameters
                  ----------
                      max_buffer_age_s : float
                              Seconds before an incomplete buffer is discarded (default 2.0).
                                  max_buffers : int
                                          Maximum number of concurrent scan-line buffers (default 64).
                                              """

    def __init__(self, max_buffer_age_s: float = 2.0, max_buffers: int = 64):
              self.max_buffer_age_s = max_buffer_age_s
              self.max_buffers = max_buffers
              self._buffers: Dict[int, ScanLineBuffer] = {}
              self._completed: collections.deque = collections.deque(maxlen=256)
              self._stats = {"received": 0, "completed": 0, "dropped": 0, "duplicates": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, packet: SonarPacket) -> Optional[np.ndarray]:
              """
                      Feed one packet into the stitcher.

                              Returns
                                      -------
                                              np.ndarray or None
                                                          The reassembled scan line when all fragments have arrived,
                                                                      otherwise None.
                                                                              """
              self._stats["received"] += 1
              self._evict_stale()

        buf = self._buffers.get(packet.line_id)

        if buf is None:
                      if len(self._buffers) >= self.max_buffers:
                                        oldest_id = min(self._buffers, key=lambda k: self._buffers[k].first_seen)
                                        logger.warning("Buffer full – dropping line %d", oldest_id)
                                        self._stats["dropped"] += 1
                                        del self._buffers[oldest_id]

                      buf = ScanLineBuffer(
                          line_id=packet.line_id,
                          total_fragments=packet.total_fragments,
                      )
                      self._buffers[packet.line_id] = buf

        # Duplicate check
        if packet.fragment_id in buf.fragments:
                      self._stats["duplicates"] += 1
                      return None

        buf.add_fragment(packet.fragment_id, packet.payload)

        if buf.is_complete():
                      scan_line = buf.assemble()
                      del self._buffers[packet.line_id]
                      self._completed.append(packet.line_id)
                      self._stats["completed"] += 1
                      logger.debug("Line %d complete (%d bytes)", packet.line_id, len(scan_line))
                      return scan_line

        return None

    def stats(self) -> Dict[str, int]:
              return dict(self._stats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_stale(self) -> None:
              now = time.time()
              stale = [lid for lid, buf in self._buffers.items()
                       if now - buf.first_seen > self.max_buffer_age_s]
              for lid in stale:
                            logger.warning("Evicting stale buffer for line %d (%d/%d fragments)",
                                                                      lid, len(self._buffers[lid].fragments),
                                                                      self._buffers[lid].total_fragments)
                            self._stats["dropped"] += 1
                            del self._buffers[lid]


# ---------------------------------------------------------------------------
# ROS 2 Node wrapper
# ---------------------------------------------------------------------------

class PacketStitcherNode(Node):
      """
          ROS 2 node that subscribes to raw sonar packets, stitches them, and
              publishes complete scan lines as sensor_msgs/Image (mono8).

                  Subscriptions
                      -------------
                          /sonar/raw_packets  (std_msgs/UInt8MultiArray)  – raw binary packets

                              Publications
                                  ------------
                                      /sonar/scan_line    (sensor_msgs/Image)          – stitched scan lines
                                          /sonar/scan_image   (sensor_msgs/Image)          – accumulated waterfall
                                              """

    HEADER_SIZE = 8   # bytes: [line_id(2), frag_id(2), total_frags(2), reserved(2)]
    WATERFALL_ROWS = 500

    def __init__(self):
              super().__init__("packet_stitcher")

        # Parameters
              self.declare_parameter("max_buffer_age_s", 2.0)
        self.declare_parameter("max_buffers", 64)
        self.declare_parameter("scan_width", 1024)

        age = self.get_parameter("max_buffer_age_s").value
        bufs = self.get_parameter("max_buffers").value
        self._width = self.get_parameter("scan_width").value

        self._stitcher = PacketStitcher(max_buffer_age_s=age, max_buffers=bufs)
        self._bridge = CvBridge()
        self._waterfall = np.zeros((self.WATERFALL_ROWS, self._width), dtype=np.uint8)
        self._wf_row = 0

        from std_msgs.msg import UInt8MultiArray
        self._sub = self.create_subscription(
                      UInt8MultiArray, "/sonar/raw_packets", self._packet_cb, 10
        )
        self._pub_line = self.create_publisher(Image, "/sonar/scan_line", 10)
        self._pub_wf   = self.create_publisher(Image, "/sonar/scan_image", 10)

        self._stat_timer = self.create_timer(5.0, self._log_stats)
        self.get_logger().info("PacketStitcherNode ready (width=%d)", self._width)

    # ------------------------------------------------------------------

    def _packet_cb(self, msg) -> None:
              raw = bytes(msg.data)
              if len(raw) < self.HEADER_SIZE:
                            return

              line_id      = int.from_bytes(raw[0:2], "big")
              frag_id      = int.from_bytes(raw[2:4], "big")
              total_frags  = int.from_bytes(raw[4:6], "big")
              payload      = raw[self.HEADER_SIZE:]

        pkt = SonarPacket(
                      line_id=line_id,
                      fragment_id=frag_id,
                      total_fragments=total_frags,
                      timestamp=time.time(),
                      payload=payload,
        )

        scan_line = self._stitcher.ingest(pkt)
        if scan_line is not None:
                      self._publish_scan_line(scan_line)

    def _publish_scan_line(self, scan_line: np.ndarray) -> None:
              # Resize/pad to expected width
              if len(scan_line) != self._width:
                            from PIL import Image as PILImage
                            import cv2
                            scan_line = cv2.resize(scan_line.reshape(1, -1), (self._width, 1),
                                                   interpolation=cv2.INTER_LINEAR)[0]

              # Publish single line
              line_img = scan_line.reshape(1, self._width)
        ros_img = self._bridge.cv2_to_imgmsg(line_img, encoding="mono8")
        ros_img.header.stamp = self.get_clock().now().to_msg()
        self._pub_line.publish(ros_img)

        # Update waterfall
        self._waterfall[self._wf_row % self.WATERFALL_ROWS] = scan_line
        self._wf_row += 1
        wf_img = self._bridge.cv2_to_imgmsg(self._waterfall, encoding="mono8")
        wf_img.header.stamp = ros_img.header.stamp
        self._pub_wf.publish(wf_img)

    def _log_stats(self) -> None:
              s = self._stitcher.stats()
              self.get_logger().info(
                  "Stats | recv=%d  complete=%d  dropped=%d  dup=%d",
                  s["received"], s["completed"], s["dropped"], s["duplicates"],
              )


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

def _simulate_fragmented_stream(n_lines: int = 20, fragments_per_line: int = 4,
                                                                 width: int = 512, drop_rate: float = 0.05):
                                                                       """Generates synthetic fragmented sonar packets for unit testing."""
                                                                       import random
                                                                       stitcher = PacketStitcher()
                                                                       frag_size = width // fragments_per_line
                                                                       completed = 0

    for line in range(n_lines):
              intensity = np.random.randint(0, 256, width, dtype=np.uint8)
              frags = [intensity[i * frag_size:(i + 1) * frag_size].tobytes()
                       for i in range(fragments_per_line)]

        # Shuffle to simulate out-of-order delivery
              indices = list(range(fragments_per_line))
              random.shuffle(indices)

        for frag_id in indices:
                      if random.random() < drop_rate:
                                        continue  # Simulate packet loss
            pkt = SonarPacket(line_id=line, fragment_id=frag_id,
                                                            total_fragments=fragments_per_line,
                                                            timestamp=time.time(), payload=frags[frag_id])
            result = stitcher.ingest(pkt)
            if result is not None:
                              completed += 1

    print(f"Simulated {n_lines} lines | Completed: {completed} | Stats: {stitcher.stats()}")


def main(args=None):
      rclpy.init(args=args)
      node = PacketStitcherNode()
      try:
                rclpy.spin(node)
except KeyboardInterrupt:
        pass
finally:
        node.destroy_node()
          rclpy.shutdown()


if __name__ == "__main__":
      import argparse
    parser = argparse.ArgumentParser(description="Standalone packet stitcher test")
    parser.add_argument("--simulate", action="store_true", help="Run simulation demo")
    parser.add_argument("--lines", type=int, default=20)
    parser.add_argument("--drop-rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.simulate:
              _simulate_fragmented_stream(n_lines=args.lines, drop_rate=args.drop_rate)
else:
        main()
