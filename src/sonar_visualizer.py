#!/usr/bin/env python3
"""
Side-Scan Sonar Waterfall Visualization
Generates real-time waterfall display of Humminbird side-scan sonar data
"""
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, List
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ColormapConfig:
    """Configuration for sonar colormap display"""
    colormap_id: int = cv2.COLORMAP_JET  # Sonar visualization colormap
    gamma: float = 1.2                   # Gamma correction
    contrast_alpha: float = 1.5          # Contrast enhancement
    brightness_beta: float = -20         # Brightness adjustment
    water_column_width: int = 20         # Water column blank zone pixels

class SonarVisualizer:
    """
    Real-time waterfall visualization of side-scan sonar data.
    
    Generates a scrolling waterfall display where each new sonar ping
    adds a row at the top of the image. Applies standard sonar image
    processing: gamma correction, contrast enhancement, colormap mapping.
    
    Supports:
    - Real-time display with OpenCV
    - Waterfall image export
    - Annotation overlay (depth, range, GPS)
    - Multiple colormaps (JET, INFERNO, MAGMA, GRAY)
    - Contrast limited adaptive histogram equalization (CLAHE)
    """
    
    def __init__(self, 
                 width: int = 1024,
                 height: int = 600,
                 config: Optional[ColormapConfig] = None):
        self.width = width           # Waterfall image width (pixels)
        self.height = height         # Waterfall image height (lines)
        self.config = config or ColormapConfig()
        
        # Waterfall buffer: rows stored oldest to newest
        self.waterfall_buffer = deque(maxlen=height)
        self.waterfall_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Processing components
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Display window
        self.window_name = "Side-Scan Sonar Waterfall"
        self.display_active = False
        
        # Annotation data
        self.latest_depth = 0.0
        self.latest_range = 0.0
        self.latest_position = (0.0, 0.0)
        self.ping_count = 0
        
        logger.info(f"SonarVisualizer initialized: {width}x{height}")
    
    def _preprocess_line(self, raw_line: np.ndarray) -> np.ndarray:
        """Apply signal processing to raw sonar intensity line"""
        if len(raw_line) == 0:
            return np.zeros(self.width // 2, dtype=np.uint8)
        
        # Resize to half-width (one side of waterfall)
        line = raw_line.astype(np.float32)
        if len(line) != self.width // 2:
            line = cv2.resize(line.reshape(1, -1), 
                            (self.width // 2, 1)).flatten()
        
        # Time-varying gain (TVG) compensation: amplify far returns
        tvg_curve = np.linspace(0.7, 1.3, len(line))
        line = line * tvg_curve
        
        # Gamma correction
        line = np.power(line / 255.0, 1.0 / self.config.gamma) * 255.0
        
        # Contrast and brightness
        line = self.config.contrast_alpha * line + self.config.brightness_beta
        
        return np.clip(line, 0, 255).astype(np.uint8)
    
    def _build_waterfall_row(self, 
                              port_data: np.ndarray, 
                              stbd_data: np.ndarray) -> np.ndarray:
        """
        Build one waterfall row from port and starboard sonar data.
        Layout: [port (reversed) | water column | starboard]
        """
        half_w = self.width // 2
        water_w = self.config.water_column_width
        side_w = (self.width - water_w) // 2
        
        # Preprocess each side
        port_proc = self._preprocess_line(port_data)[:side_w]
        stbd_proc = self._preprocess_line(stbd_data)[:side_w]
        
        # Port side: reversed (near = center, far = left edge)
        port_disp = port_proc[::-1]
        
        # Water column blank zone
        water_col = np.zeros(water_w, dtype=np.uint8)
        
        # Combine: [port | water | starboard]
        row = np.concatenate([port_disp, water_col, stbd_proc])
        
        # Ensure correct width
        if len(row) != self.width:
            row = cv2.resize(row.reshape(1, -1), (self.width, 1)).flatten()
        
        return row.astype(np.uint8)
    
    def _apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for adaptive contrast enhancement"""
        return self.clahe.apply(gray_image)
    
    def _apply_colormap(self, gray_image: np.ndarray) -> np.ndarray:
        """Convert grayscale sonar to false-color display"""
        return cv2.applyColorMap(gray_image, self.config.colormap_id)
    
    def _annotate_image(self, image: np.ndarray) -> np.ndarray:
        """Add metadata annotations to waterfall image"""
        annotated = image.copy()
        
        # Status bar background
        cv2.rectangle(annotated, (0, 0), (self.width, 30), (0, 0, 0), -1)
        
        # Depth annotation
        depth_text = f"Depth: {self.latest_depth:.1f}m"
        cv2.putText(annotated, depth_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Range annotation
        range_text = f"Range: {self.latest_range:.0f}m"
        cv2.putText(annotated, range_text, (150, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # GPS annotation
        lat, lon = self.latest_position
        gps_text = f"GPS: {lat:.5f}, {lon:.5f}"
        cv2.putText(annotated, gps_text, (300, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Ping counter
        ping_text = f"Ping: {self.ping_count}"
        cv2.putText(annotated, ping_text, (700, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Range scale markers on sides
        n_markers = 5
        for i in range(1, n_markers + 1):
            y_pos = 30 + (self.height - 30) * i // n_markers
            range_val = self.latest_range * i / n_markers
            cv2.line(annotated, (0, y_pos), (5, y_pos), (255, 255, 255), 1)
            cv2.line(annotated, (self.width-5, y_pos), (self.width, y_pos), (255, 255, 255), 1)
            cv2.putText(annotated, f"{range_val:.0f}m", (7, y_pos+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        return annotated
    
    def add_ping(self, 
                 port_data: np.ndarray,
                 stbd_data: np.ndarray,
                 depth_m: float = 0.0,
                 range_m: float = 0.0,
                 lat: float = 0.0,
                 lon: float = 0.0):
        """
        Add a new sonar ping to the waterfall display.
        Port and starboard arrays are intensity values (uint8, 0-255).
        """
        self.latest_depth = depth_m
        self.latest_range = range_m
        self.latest_position = (lat, lon)
        self.ping_count += 1
        
        # Build waterfall row
        row = self._build_waterfall_row(port_data, stbd_data)
        self.waterfall_buffer.append(row)
        
        # Rebuild waterfall image from buffer (newest at top)
        n_rows = len(self.waterfall_buffer)
        gray_wf = np.zeros((self.height, self.width), dtype=np.uint8)
        
        buffer_list = list(self.waterfall_buffer)
        for i, r in enumerate(reversed(buffer_list)):
            if i >= self.height:
                break
            gray_wf[i] = r
        
        # Apply CLAHE for adaptive contrast
        enhanced = self._apply_clahe(gray_wf)
        
        # Convert to color
        self.waterfall_image = self._apply_colormap(enhanced)
    
    def get_display_image(self, annotate: bool = True) -> np.ndarray:
        """Get current waterfall image for display or export"""
        if annotate:
            return self._annotate_image(self.waterfall_image)
        return self.waterfall_image.copy()
    
    def show(self, annotate: bool = True):
        """Display waterfall in OpenCV window"""
        if not self.display_active:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            self.display_active = True
        
        display_img = self.get_display_image(annotate)
        cv2.imshow(self.window_name, display_img)
        return cv2.waitKey(1) & 0xFF
    
    def save(self, filepath: str, annotate: bool = True):
        """Save current waterfall image to file"""
        display_img = self.get_display_image(annotate)
        cv2.imwrite(filepath, display_img)
        logger.info(f"Waterfall saved: {filepath}")
    
    def close(self):
        """Close display window"""
        if self.display_active:
            cv2.destroyWindow(self.window_name)
            self.display_active = False
    
    def set_colormap(self, colormap_id: int):
        """Change visualization colormap"""
        self.config.colormap_id = colormap_id
        logger.info(f"Colormap changed to: {colormap_id}")
    
    def reset(self):
        """Clear waterfall buffer"""
        self.waterfall_buffer.clear()
        self.waterfall_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.ping_count = 0
    
    def get_statistics(self) -> dict:
        """Return display statistics"""
        return {
            'ping_count': self.ping_count,
            'buffer_fill': len(self.waterfall_buffer) / self.height,
            'width': self.width,
            'height': self.height,
        }


def demo_visualization():
    """Demo with synthetic sonar data"""
    viz = SonarVisualizer(width=1024, height=600)
    
    for i in range(200):
        # Synthetic side-scan data with Gaussian targets
        x = np.arange(512)
        port_data = np.zeros(512, dtype=np.float32)
        stbd_data = np.zeros(512, dtype=np.float32)
        
        # Add random targets
        for _ in range(3):
            pos = np.random.randint(50, 450)
            port_data += 200 * np.exp(-((x - pos)**2) / (2 * 20**2))
        for _ in range(3):
            pos = np.random.randint(50, 450)
            stbd_data += 200 * np.exp(-((x - pos)**2) / (2 * 20**2))
        
        port_data = np.clip(port_data + np.random.normal(10, 5, 512), 0, 255).astype(np.uint8)
        stbd_data = np.clip(stbd_data + np.random.normal(10, 5, 512), 0, 255).astype(np.uint8)
        
        viz.add_ping(port_data, stbd_data, depth_m=5.2, range_m=30.0, 
                    lat=38.9716, lon=-76.8867)
        
        key = viz.show(annotate=True)
        if key == ord('q'):
            break
    
    viz.save('waterfall_demo.png')
    viz.close()


if __name__ == '__main__':
    demo_visualization()
