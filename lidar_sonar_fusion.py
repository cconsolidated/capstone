#!/usr/bin/env python3
"""
LIDAR-Sonar Data Fusion System
Combines HYPACK LIDAR (HSX) and NORBIT WBMS sonar (S7K) data for water surface detection
"""

import numpy as np
import pandas as pd
import struct
import datetime
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
import re
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.interpolate import griddata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LidarPoint:
    """Single LIDAR point with spatial and temporal data"""
    x: float
    y: float
    z: float
    intensity: float
    timestamp: datetime.datetime
    
@dataclass
class SonarPing:
    """Single sonar ping with bathymetric data"""
    x: float
    y: float
    depth: float
    intensity: float
    timestamp: datetime.datetime
    beam_angle: float

@dataclass
class WaterSurfacePoint:
    """Detected water surface point combining LIDAR and sonar data"""
    x: float
    y: float
    surface_elevation: float  # From LIDAR (above water)
    bottom_depth: float       # From sonar (below water)
    water_depth: float        # Calculated difference
    timestamp: datetime.datetime
    confidence: float

class HSXParser:
    """Parser for HYPACK HSX LIDAR data files"""
    
    def __init__(self, hsx_file: str):
        self.hsx_file = Path(hsx_file)
        self.raw_file = self.hsx_file.with_suffix('.RAW')
        self.header_info = {}
        self.devices = {}
        
    def parse_header(self) -> Dict:
        """Parse HSX header information"""
        logger.info(f"Parsing HSX header from {self.hsx_file}")
        
        with open(self.hsx_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Parse basic info
                if line.startswith('TND'):
                    parts = line.split()
                    self.header_info['time'] = parts[1]
                    self.header_info['date'] = parts[2]
                
                # Parse device information
                elif line.startswith('DEV'):
                    parts = line.split()
                    device_id = int(parts[1])
                    device_type = int(parts[2])
                    device_name = ' '.join(parts[3:]).strip('"')
                    self.devices[device_id] = {
                        'type': device_type,
                        'name': device_name
                    }
                
                # Parse offset information
                elif line.startswith('OF2'):
                    parts = line.split()
                    device_id = int(parts[1])
                    offset_type = int(parts[2])
                    if device_id in self.devices:
                        self.devices[device_id]['offsets'] = {
                            'x': float(parts[3]),
                            'y': float(parts[4]),
                            'z': float(parts[5]),
                            'roll': float(parts[6]),
                            'pitch': float(parts[7]),
                            'yaw': float(parts[8])
                        }
        
        return self.header_info
    
    def parse_lidar_data(self) -> List[LidarPoint]:
        """Parse LIDAR point cloud data from RAW file"""
        logger.info(f"Parsing LIDAR data from {self.raw_file}")
        
        if not self.raw_file.exists():
            logger.error(f"RAW file not found: {self.raw_file}")
            return []
        
        lidar_points = []
        
        # Find LIDAR device (Velodyne VLP-16)
        lidar_device_id = None
        for dev_id, dev_info in self.devices.items():
            if 'Velodyne' in dev_info.get('name', ''):
                lidar_device_id = dev_id
                break
        
        if lidar_device_id is None:
            logger.warning("No Velodyne LIDAR device found in header")
            return []
        
        try:
            with open(self.raw_file, 'rb') as f:
                while True:
                    # Read record header (simplified - actual format may vary)
                    header = f.read(20)
                    if len(header) < 20:
                        break
                    
                    # Parse basic record structure (this is a simplified example)
                    record_type, device_id, timestamp, data_length = struct.unpack('<HHIQ', header)
                    
                    if device_id == lidar_device_id:
                        # Read LIDAR point data
                        point_data = f.read(data_length)
                        points = self._parse_velodyne_points(point_data, timestamp)
                        lidar_points.extend(points)
                    else:
                        # Skip non-LIDAR data
                        f.seek(data_length, 1)
        
        except Exception as e:
            logger.error(f"Error parsing RAW file: {e}")
        
        logger.info(f"Parsed {len(lidar_points)} LIDAR points")
        return lidar_points
    
    def _parse_velodyne_points(self, data: bytes, timestamp: int) -> List[LidarPoint]:
        """Parse Velodyne VLP-16 point data"""
        points = []
        
        # Velodyne VLP-16 data structure (simplified)
        # Each point: distance (2 bytes), intensity (1 byte), angle info
        point_size = 3  # Simplified
        num_points = len(data) // point_size
        
        for i in range(num_points):
            offset = i * point_size
            if offset + point_size > len(data):
                break
            
            # Parse point data (this is highly simplified)
            distance_raw = struct.unpack('<H', data[offset:offset+2])[0]
            intensity = data[offset+2]
            
            # Convert to XYZ coordinates (simplified conversion)
            distance = distance_raw * 0.002  # Convert to meters
            angle = (i / num_points) * 2 * np.pi  # Simplified angle calculation
            
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            z = 0  # Would need proper elevation calculation
            
            dt = datetime.datetime.fromtimestamp(timestamp / 1000000.0)
            
            points.append(LidarPoint(x, y, z, intensity, dt))
        
        return points

class S7KParser:
    """Parser for NORBIT WBMS S7K sonar data files"""
    
    def __init__(self, s7k_file: str):
        self.s7k_file = Path(s7k_file)
        
    def parse_sonar_data(self) -> List[SonarPing]:
        """Parse sonar bathymetric data from S7K file"""
        logger.info(f"Parsing sonar data from {self.s7k_file}")
        
        sonar_pings = []
        
        try:
            with open(self.s7k_file, 'rb') as f:
                while True:
                    # Read S7K record header
                    header = f.read(64)  # S7K header is 64 bytes
                    if len(header) < 64:
                        break
                    
                    # Parse S7K header (simplified)
                    sync_pattern = struct.unpack('<H', header[0:2])[0]
                    if sync_pattern != 0x0000:  # S7K sync pattern
                        continue
                    
                    record_type = struct.unpack('<I', header[8:12])[0]
                    timestamp = struct.unpack('<Q', header[16:24])[0]
                    record_size = struct.unpack('<I', header[32:36])[0]
                    
                    # Read record data
                    data = f.read(record_size - 64)
                    
                    # Parse bathymetric data (record type 7027 for multibeam)
                    if record_type == 7027:
                        pings = self._parse_bathymetric_data(data, timestamp)
                        sonar_pings.extend(pings)
        
        except Exception as e:
            logger.error(f"Error parsing S7K file: {e}")
        
        logger.info(f"Parsed {len(sonar_pings)} sonar pings")
        return sonar_pings
    
    def _parse_bathymetric_data(self, data: bytes, timestamp: int) -> List[SonarPing]:
        """Parse multibeam bathymetric data"""
        pings = []
        
        if len(data) < 32:
            return pings
        
        # Parse bathymetric record header (simplified)
        num_beams = struct.unpack('<I', data[0:4])[0]
        
        beam_size = 32  # Simplified beam data size
        
        for i in range(min(num_beams, (len(data) - 32) // beam_size)):
            offset = 32 + i * beam_size
            
            # Parse beam data (simplified)
            depth = struct.unpack('<f', data[offset:offset+4])[0]
            x = struct.unpack('<f', data[offset+4:offset+8])[0]
            y = struct.unpack('<f', data[offset+8:offset+12])[0]
            intensity = struct.unpack('<f', data[offset+12:offset+16])[0]
            beam_angle = struct.unpack('<f', data[offset+16:offset+20])[0]
            
            dt = datetime.datetime.fromtimestamp(timestamp / 1000000.0)
            
            pings.append(SonarPing(x, y, depth, intensity, dt, beam_angle))
        
        return pings

class DataFusion:
    """Main class for fusing LIDAR and sonar data"""
    
    def __init__(self, spatial_tolerance: float = 1.0, temporal_tolerance: float = 10.0):
        self.spatial_tolerance = spatial_tolerance  # meters
        self.temporal_tolerance = temporal_tolerance  # seconds
        
    def align_datasets(self, lidar_points: List[LidarPoint], 
                      sonar_pings: List[SonarPing]) -> List[WaterSurfacePoint]:
        """Spatially and temporally align LIDAR and sonar data"""
        logger.info("Aligning LIDAR and sonar datasets")
        
        water_points = []
        
        if not lidar_points or not sonar_pings:
            logger.warning("Empty dataset provided")
            return water_points
        
        # Create spatial index for sonar data
        sonar_coords = np.array([[ping.x, ping.y] for ping in sonar_pings])
        sonar_tree = KDTree(sonar_coords)
        
        for lidar_point in lidar_points:
            # Find nearby sonar pings
            distances, indices = sonar_tree.query([lidar_point.x, lidar_point.y], 
                                                 k=5, distance_upper_bound=self.spatial_tolerance)
            
            # Filter by temporal proximity
            valid_matches = []
            for dist, idx in zip(distances, indices):
                if dist == np.inf or idx >= len(sonar_pings):
                    continue
                
                sonar_ping = sonar_pings[idx]
                time_diff = abs((lidar_point.timestamp - sonar_ping.timestamp).total_seconds())
                
                if time_diff <= self.temporal_tolerance:
                    valid_matches.append((sonar_ping, dist, time_diff))
            
            if valid_matches:
                # Use closest spatial match
                best_match = min(valid_matches, key=lambda x: x[1])
                sonar_ping = best_match[0]
                
                # Calculate water depth and confidence
                water_depth = abs(lidar_point.z - sonar_ping.depth)
                confidence = 1.0 / (1.0 + best_match[1] + best_match[2] * 0.1)
                
                water_point = WaterSurfacePoint(
                    x=lidar_point.x,
                    y=lidar_point.y,
                    surface_elevation=lidar_point.z,
                    bottom_depth=sonar_ping.depth,
                    water_depth=water_depth,
                    timestamp=lidar_point.timestamp,
                    confidence=confidence
                )
                
                water_points.append(water_point)
        
        logger.info(f"Created {len(water_points)} aligned water surface points")
        return water_points
    
    def detect_water_changes(self, water_points: List[WaterSurfacePoint], 
                           grid_resolution: float = 1.0) -> np.ndarray:
        """Detect changes in water surface using combined data"""
        logger.info("Detecting water surface changes")
        
        if not water_points:
            return np.array([])
        
        # Extract coordinates and depths
        coords = np.array([[p.x, p.y] for p in water_points])
        depths = np.array([p.water_depth for p in water_points])
        
        # Create regular grid
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        xi = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        yi = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate depths onto grid
        depth_grid = griddata(coords, depths, (xi_grid, yi_grid), method='linear')
        
        # Calculate depth gradients to detect changes
        dy, dx = np.gradient(depth_grid)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        return gradient_magnitude
    
    def export_results(self, water_points: List[WaterSurfacePoint], 
                      output_file: str):
        """Export fused data results to CSV"""
        logger.info(f"Exporting results to {output_file}")
        
        data = []
        for point in water_points:
            data.append({
                'x': point.x,
                'y': point.y,
                'surface_elevation': point.surface_elevation,
                'bottom_depth': point.bottom_depth,
                'water_depth': point.water_depth,
                'timestamp': point.timestamp.isoformat(),
                'confidence': point.confidence
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(data)} points to {output_file}")

class Visualizer:
    """Visualization tools for combined LIDAR-sonar data"""
    
    @staticmethod
    def plot_water_surface(water_points: List[WaterSurfacePoint], 
                          output_file: str = None):
        """Create 3D visualization of water surface detection"""
        if not water_points:
            logger.warning("No water points to visualize")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # Extract data
        x = [p.x for p in water_points]
        y = [p.y for p in water_points]
        surface_z = [p.surface_elevation for p in water_points]
        bottom_z = [p.bottom_depth for p in water_points]
        water_depth = [p.water_depth for p in water_points]
        
        # Plot 1: Water surface elevation
        ax1 = fig.add_subplot(221)
        scatter1 = ax1.scatter(x, y, c=surface_z, cmap='viridis', alpha=0.6)
        ax1.set_title('Water Surface Elevation (LIDAR)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        plt.colorbar(scatter1, ax=ax1, label='Elevation (m)')
        
        # Plot 2: Bottom depth
        ax2 = fig.add_subplot(222)
        scatter2 = ax2.scatter(x, y, c=bottom_z, cmap='plasma', alpha=0.6)
        ax2.set_title('Bottom Depth (Sonar)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        plt.colorbar(scatter2, ax=ax2, label='Depth (m)')
        
        # Plot 3: Water depth
        ax3 = fig.add_subplot(223)
        scatter3 = ax3.scatter(x, y, c=water_depth, cmap='coolwarm', alpha=0.6)
        ax3.set_title('Water Depth (Combined)')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        plt.colorbar(scatter3, ax=ax3, label='Water Depth (m)')
        
        # Plot 4: 3D view
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.scatter(x, y, surface_z, c='blue', alpha=0.3, label='Surface (LIDAR)')
        ax4.scatter(x, y, bottom_z, c='red', alpha=0.3, label='Bottom (Sonar)')
        ax4.set_title('Combined 3D View')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.legend()
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        plt.show()

def main():
    """Main processing pipeline"""
    logger.info("Starting LIDAR-Sonar Data Fusion Pipeline")
    
    # Configuration
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    
    # Initialize fusion system
    fusion = DataFusion(spatial_tolerance=2.0, temporal_tolerance=30.0)
    
    # Process all available datasets
    all_water_points = []
    
    # Process ILIDAR LAKE TEST data
    ilidar_dir = data_dir / "HYPACK iLIDAR DATA" / "ILIDAR LAKE TEST"
    wbms_dir = data_dir / "WBMS"
    
    # Find matching datasets by timestamp
    for hsx_file in ilidar_dir.glob("*.HSX"):
        logger.info(f"Processing LIDAR file: {hsx_file}")
        
        # Parse LIDAR data
        hsx_parser = HSXParser(str(hsx_file))
        hsx_parser.parse_header()
        lidar_points = hsx_parser.parse_lidar_data()
        
        # Find corresponding sonar data
        for wbms_session in wbms_dir.iterdir():
            if wbms_session.is_dir():
                s7k_files = list(wbms_session.glob("*.s7k"))
                if s7k_files:
                    s7k_file = s7k_files[0]
                    logger.info(f"Processing sonar file: {s7k_file}")
                    
                    # Parse sonar data
                    s7k_parser = S7KParser(str(s7k_file))
                    sonar_pings = s7k_parser.parse_sonar_data()
                    
                    # Fuse datasets
                    water_points = fusion.align_datasets(lidar_points, sonar_pings)
                    all_water_points.extend(water_points)
    
    if all_water_points:
        # Export results
        output_file = "/Users/tycrouch/Desktop/untitled folder 4/fused_water_data.csv"
        fusion.export_results(all_water_points, output_file)
        
        # Create visualizations
        viz_file = "/Users/tycrouch/Desktop/untitled folder 4/water_surface_analysis.png"
        Visualizer.plot_water_surface(all_water_points, viz_file)
        
        # Detect water changes
        changes = fusion.detect_water_changes(all_water_points)
        logger.info(f"Detected water surface changes with max gradient: {np.nanmax(changes):.3f}")
        
        logger.info("Data fusion pipeline completed successfully")
    else:
        logger.warning("No aligned water points found - check data formats and timestamps")

if __name__ == "__main__":
    main()

