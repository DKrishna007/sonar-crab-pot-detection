#!/usr/bin/env python3
"""
Humminbird Side-Scan Sonar Packet Capture and UDP Reconstruction
"""
import socket
import struct
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUMMINBIRD_PORT = 5200
PACKET_HEADER_SIZE = 64
SONAR_DATA_OFFSET = 64
MAX_PACKET_SIZE = 65535
SIDE_SCAN_CHANNEL = 0x02

@dataclass
class SonarPacket:
    timestamp: float
    channel: int
    ping_number: int
    range_meters: float
    speed_knots: float
    heading_degrees: float
    depth_meters: float
    latitude: float
    longitude: float
    data: object
    packet_size: int
    valid: bool = True

class HumminbirdPacketCapture:
    """
    Captures and reconstructs UDP packets from Humminbird HELIX side-scan sonar.
    Handles packet fragmentation reassembly and side-scan channel filtering.
    Tested with Humminbird HELIX 7, 9, 12 MEGA SI+ models.
    Connection: Direct Ethernet or WiFi hotspot on port 5200.
    """
    def __init__(self, host='0.0.0.0', port=HUMMINBIRD_PORT, buffer_size=100, timeout=5.0):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.socket = None
        self.capture_active = False
        self.reconstruction_buffers = {}
        self.completed_pings = deque(maxlen=buffer_size)
        self.packets_received = 0
        self.packets_dropped = 0
        self.pings_completed = 0
        self.line_width_pixels = 512
        logger.info(f"HumminbirdPacketCapture initialized: {host}:{port}")

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(self.timeout)
            self.capture_active = True
            logger.info(f"Bound to {self.host}:{self.port}")
            return True
        except socket.error as e:
            logger.error(f"Failed to bind socket: {e}")
            return False

    def disconnect(self):
        self.capture_active = False
        if self.socket:
            self.socket.close()
            self.socket = None

    def _parse_header(self, data):
        if len(data) < PACKET_HEADER_SIZE:
            return None
        try:
            magic = struct.unpack_from('<I', data, 0)[0]
            if magic not in (0x42494E00, 0x42494E01):
                return None
            return {
                'channel': struct.unpack_from('<I', data, 4)[0] & 0xFF,
                'ping_number': struct.unpack_from('<I', data, 8)[0],
                'fragment_offset': struct.unpack_from('<I', data, 12)[0],
                'total_size': struct.unpack_from('<I', data, 16)[0],
                'timestamp': struct.unpack_from('<I', data, 20)[0] / 1000.0,
                'range_m': struct.unpack_from('<I', data, 24)[0] / 100.0,
                'speed_knots': struct.unpack_from('<H', data, 28)[0] / 10.0,
                'heading_deg': struct.unpack_from('<H', data, 30)[0] / 10.0,
                'depth_m': struct.unpack_from('<H', data, 36)[0] / 10.0,
                'latitude': struct.unpack_from('<d', data, 40)[0],
                'longitude': struct.unpack_from('<d', data, 48)[0],
            }
        except struct.error:
            return None

    def _add_fragment(self, header, fragment_data):
        ping_num = header['ping_number']
        if ping_num not in self.reconstruction_buffers:
            self.reconstruction_buffers[ping_num] = {
                'expected_size': header['total_size'],
                'fragments': [],
                'offsets': [],
                'timestamp': header['timestamp'],
                'complete': False
            }
        buf = self.reconstruction_buffers[ping_num]
        buf['fragments'].append(fragment_data)
        buf['offsets'].append(header['fragment_offset'])
        if sum(len(f) for f in buf['fragments']) >= buf['expected_size']:
            buf['complete'] = True

    def _reconstruct_ping(self, ping_num, header):
        buf = self.reconstruction_buffers.get(ping_num)
        if not buf or not buf['complete']:
            return None
        sorted_pairs = sorted(zip(buf['offsets'], buf['fragments']))
        raw_data = b''.join(f for _, f in sorted_pairs)
        sonar_array = np.frombuffer(raw_data[:buf['expected_size']], dtype=np.uint8)
        if len(sonar_array) >= self.line_width_pixels:
            sonar_array = sonar_array[:self.line_width_pixels]
        del self.reconstruction_buffers[ping_num]
        return SonarPacket(
            timestamp=buf['timestamp'], channel=header['channel'],
            ping_number=ping_num, range_meters=header['range_m'],
            speed_knots=header['speed_knots'], heading_degrees=header['heading_deg'],
            depth_meters=header['depth_m'], latitude=header['latitude'],
            longitude=header['longitude'], data=sonar_array,
            packet_size=buf['expected_size']
        )

    def capture_packet(self):
        if not self.socket or not self.capture_active:
            return None
        try:
            raw_data, addr = self.socket.recvfrom(MAX_PACKET_SIZE)
            self.packets_received += 1
            header = self._parse_header(raw_data)
            if not header or header['channel'] != SIDE_SCAN_CHANNEL:
                self.packets_dropped += 1
                return None
            self._add_fragment(header, raw_data[SONAR_DATA_OFFSET:])
            ping_num = header['ping_number']
            buf = self.reconstruction_buffers.get(ping_num)
            if buf and buf['complete']:
                packet = self._reconstruct_ping(ping_num, header)
                if packet:
                    self.pings_completed += 1
                    self.completed_pings.append(packet)
                    return packet
            return None
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    def capture_batch(self, n_pings=10, timeout=30.0):
        pings = []
        start = time.time()
        while len(pings) < n_pings and (time.time() - start) < timeout:
            p = self.capture_packet()
            if p:
                pings.append(p)
        logger.info(f"Captured {len(pings)}/{n_pings} pings")
        return pings

    def get_stats(self):
        return {
            'packets_received': self.packets_received,
            'packets_dropped': self.packets_dropped,
            'pings_completed': self.pings_completed,
            'drop_rate': self.packets_dropped / max(1, self.packets_received),
        }


if __name__ == '__main__':
    capture = HumminbirdPacketCapture(host='192.168.1.100')
    if capture.connect():
        pings = capture.capture_batch(n_pings=50)
        capture.disconnect()
        print(f"Captured {len(pings)} pings")
