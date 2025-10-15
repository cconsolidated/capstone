#!/usr/bin/env python3
"""
Real HYPACK Data Parser
Parses actual HSX/RAW files from your ILIDAR LAKE TEST
"""

import numpy as np
import datetime
import re
from pathlib import Path
from lidar_sonar_fusion import LidarPoint, SonarPing, DataFusion, Visualizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HYPACKRealParser:
    """Parser for real HYPACK ASCII RAW files"""
    
    def __init__(self, raw_file):
        self.raw_file = Path(raw_file)
        self.date = None
        self.base_time = None
        
    def parse_lidar_data(self):
        """Parse LiDAR points from HYPACK RAW file"""
        logger.info(f"Parsing real HYPACK data from {self.raw_file}")
        
        lidar_points = []
        
        with open(self.raw_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Parse date/time from TND record
                if line.startswith('TND'):
                    # TND 09:49:53 08/23/2025 300
                    parts = line.split()
                    time_str = parts[1]  # 09:49:53
                    date_str = parts[2]  # 08/23/2025
                    
                    # Parse to datetime
                    dt_str = f"{date_str} {time_str}"
                    self.base_time = datetime.datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
                    logger.info(f"Survey date/time: {self.base_time}")
                
                # Parse position records
                elif line.startswith('POS'):
                    # POS 0 35393.485 3196574.958 13937914.954
                    # device_id, timestamp, easting, northing
                    pass
                
                # Parse RAW records (LiDAR points)
                elif line.startswith('RAW'):
                    # RAW 0 35393.485 4 300105.24463 -950713.12731 -16.94759 144953.48522
                    # device, timestamp, count?, X, Y, Z, intensity?
                    parts = line.split()
                    
                    if len(parts) >= 7:
                        device_id = int(parts[1])
                        timestamp_sec = float(parts[2])  # Seconds since midnight
                        
                        # Calculate actual timestamp
                        if self.base_time:
                            # Timestamp is seconds since midnight
                            hours = int(timestamp_sec // 3600)
                            minutes = int((timestamp_sec % 3600) // 60)
                            seconds = timestamp_sec % 60
                            
                            point_time = self.base_time.replace(
                                hour=hours, minute=minutes, second=int(seconds),
                                microsecond=int((seconds % 1) * 1000000)
                            )
                        else:
                            point_time = datetime.datetime.now()
                        
                        # Extract coordinates (these might be in local coordinates)
                        x = float(parts[4])
                        y = float(parts[5])
                        z = float(parts[6])
                        
                        # Intensity if available
                        intensity = float(parts[7]) if len(parts) > 7 else 100.0
                        
                        point = LidarPoint(
                            x=x, y=y, z=z,
                            intensity=intensity,
                            timestamp=point_time
                        )
                        lidar_points.append(point)
        
        logger.info(f"✓ Parsed {len(lidar_points)} LiDAR points from real data")
        
        if len(lidar_points) > 0:
            logger.info(f"  X range: {min(p.x for p in lidar_points):.2f} to {max(p.x for p in lidar_points):.2f}")
            logger.info(f"  Y range: {min(p.y for p in lidar_points):.2f} to {max(p.y for p in lidar_points):.2f}")
            logger.info(f"  Z range: {min(p.z for p in lidar_points):.2f} to {max(p.z for p in lidar_points):.2f}")
        
        return lidar_points

def create_synthetic_matching_sonar(lidar_points, depth_offset=-5.0):
    """
    Create synthetic sonar data matching the LiDAR survey area
    (Until we can access real NORBIT data)
    """
    logger.info("Creating synthetic sonar data to match LiDAR coverage...")
    
    if not lidar_points:
        return []
    
    # Get bounds from LiDAR
    x_coords = [p.x for p in lidar_points]
    y_coords = [p.y for p in lidar_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Sample points across the same area
    num_sonar_points = len(lidar_points) // 2  # Fewer sonar points
    
    x = np.random.uniform(x_min, x_max, num_sonar_points)
    y = np.random.uniform(y_min, y_max, num_sonar_points)
    
    # Create depth based on position (simulated seafloor)
    depth = depth_offset + 0.5 * np.sin((x - x_min) / 1000) + 0.3 * np.cos((y - y_min) / 1000)
    depth += np.random.normal(0, 0.1, num_sonar_points)
    
    sonar_pings = []
    base_time = lidar_points[0].timestamp
    
    for i in range(num_sonar_points):
        ping = SonarPing(
            x=x[i], y=y[i], depth=depth[i],
            intensity=np.random.uniform(100, 200),
            timestamp=base_time + datetime.timedelta(seconds=i*0.1),
            beam_angle=np.random.uniform(-60, 60)
        )
        sonar_pings.append(ping)
    
    logger.info(f"✓ Created {len(sonar_pings)} synthetic sonar pings")
    logger.info(f"  Depth range: {min(depth):.2f} to {max(depth):.2f}")
    
    return sonar_pings

def process_real_lake_data():
    """Process the real ILIDAR LAKE TEST data"""
    
    logger.info("\n" + "="*80)
    logger.info("PROCESSING REAL HYPACK ILIDAR LAKE TEST DATA")
    logger.info("="*80 + "\n")
    
    # Path to real data
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    raw_file = data_dir / "HYPACK iLIDAR DATA" / "ILIDAR LAKE TEST" / "0000_0949.RAW"
    
    # Parse real LiDAR data
    parser = HYPACKRealParser(raw_file)
    lidar_points = parser.parse_lidar_data()
    
    if not lidar_points:
        logger.error("No LiDAR points parsed!")
        return
    
    # Create matching sonar data (synthetic until we get real NORBIT data)
    sonar_pings = create_synthetic_matching_sonar(lidar_points, depth_offset=-5.0)
    
    # Initialize fusion system
    logger.info("\n" + "="*80)
    logger.info("Running EA1-EA8 Workflow on Real Data")
    logger.info("="*80 + "\n")
    
    fusion = DataFusion(
        spatial_tolerance=10.0,  # Larger tolerance for real data
        temporal_tolerance=60.0,
        mlw_datum=0.5,
        sensor_offset_z=1.5
    )
    
    # EA2: Align coordinate systems
    lidar_aligned, sonar_aligned = fusion.apply_coordinate_transformation(
        lidar_points, sonar_pings
    )
    
    # EA3: Normalize to MLW
    lidar_norm, sonar_norm = fusion.normalize_to_mlw_datum(
        lidar_aligned, sonar_aligned, tidal_correction=0.2
    )
    
    # EA4: Merge point clouds
    water_points = fusion.align_datasets(lidar_norm, sonar_norm)
    
    if not water_points:
        logger.error("No merged points created!")
        return
    
    # EA5: Check alignment errors
    water_points_corrected = fusion.check_alignment_errors(water_points, tolerance=1.0)
    
    # EA6: Validate accuracy
    metrics = fusion.validate_accuracy(water_points_corrected)
    
    # EA7: Visualize
    logger.info("\n" + "="*80)
    logger.info("EA7: Creating Visualization")
    logger.info("="*80)
    
    output_viz = "/Users/tycrouch/Desktop/untitled folder 4/real_lake_data_visualization.png"
    Visualizer.plot_water_surface(water_points_corrected, output_viz)
    
    # EA8: Export
    logger.info("\n" + "="*80)
    logger.info("EA8: Exporting Results")
    logger.info("="*80)
    
    output_csv = "/Users/tycrouch/Desktop/untitled folder 4/real_lake_data_merged.csv"
    fusion.export_results(water_points_corrected, output_csv, include_metrics=True)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("REAL DATA PROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults:")
    logger.info(f"  • Merged points: {len(water_points_corrected)}")
    logger.info(f"  • Mean water depth: {metrics['depth_mean']:.2f}m ± {metrics['depth_std']:.2f}m")
    logger.info(f"  • Depth range: {metrics['depth_min']:.2f}m to {metrics['depth_max']:.2f}m")
    logger.info(f"  • Vertical alignment: {metrics['vertical_alignment_mean']:.3f}m")
    logger.info(f"  • Alignment errors corrected: {len(fusion.alignment_errors)}")
    
    logger.info(f"\nOutput Files:")
    logger.info(f"  • {output_viz}")
    logger.info(f"  • {output_csv}")
    logger.info(f"  • {output_csv.replace('.csv', '_metrics.csv')}")
    
    logger.info("\n" + "="*80 + "\n")
    
    return water_points_corrected, metrics

if __name__ == "__main__":
    process_real_lake_data()

