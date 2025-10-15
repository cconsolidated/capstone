#!/usr/bin/env python3
"""
EA1-EA8 Demonstration with Synthetic Data
Shows complete workflow from data collection through export
"""

import numpy as np
import datetime
from lidar_sonar_fusion import (
    LidarPoint, SonarPing, DataFusion, Visualizer
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_lidar(num_points=500):
    """Generate synthetic LiDAR data (surface)"""
    logger.info("EA1: Generating synthetic LiDAR data (above water)")
    
    points = []
    base_time = datetime.datetime.now()
    
    # Create a 50x50m survey area
    x = np.random.uniform(-25, 25, num_points)
    y = np.random.uniform(-25, 25, num_points)
    
    # Water surface at z=2m with small waves
    z = 2.0 + 0.2 * np.sin(x/5) + 0.1 * np.cos(y/5) + np.random.normal(0, 0.05, num_points)
    
    for i in range(num_points):
        point = LidarPoint(
            x=x[i],
            y=y[i],
            z=z[i],
            intensity=np.random.uniform(50, 250),
            timestamp=base_time + datetime.timedelta(seconds=i*0.1)
        )
        points.append(point)
    
    logger.info(f"EA1: Generated {len(points)} LiDAR points")
    return points

def generate_synthetic_sonar(num_pings=500):
    """Generate synthetic sonar data (underwater)"""
    logger.info("EA1: Generating synthetic sonar data (below water)")
    
    pings = []
    base_time = datetime.datetime.now()
    
    # Same survey area
    x = np.random.uniform(-25, 25, num_pings)
    y = np.random.uniform(-25, 25, num_pings)
    
    # Seafloor at depth=-5m with some variation
    depth = -5.0 - 0.5 * np.sin(x/8) - 0.3 * np.cos(y/8) + np.random.normal(0, 0.1, num_pings)
    
    for i in range(num_pings):
        ping = SonarPing(
            x=x[i],
            y=y[i],
            depth=depth[i],
            intensity=np.random.uniform(100, 200),
            timestamp=base_time + datetime.timedelta(seconds=i*0.1),
            beam_angle=np.random.uniform(-60, 60)
        )
        pings.append(ping)
    
    logger.info(f"EA1: Generated {len(pings)} sonar pings")
    return pings

def main():
    """Demonstrate complete EA1-EA8 workflow"""
    
    print("\n" + "="*80)
    print("ENGINEERING ACTIONS (EA1-EA8) DEMONSTRATION")
    print("Using Synthetic Data to Show Complete Workflow")
    print("="*80 + "\n")
    
    # EA1: Collect Data
    logger.info("="*80)
    logger.info("EA1: COLLECT SONAR AND LIDAR DATA")
    logger.info("="*80)
    lidar_points = generate_synthetic_lidar(num_points=500)
    sonar_pings = generate_synthetic_sonar(num_pings=500)
    
    # Initialize fusion system
    logger.info("\nInitializing DataFusion system with configurable parameters...")
    fusion = DataFusion(
        spatial_tolerance=2.0,      # meters (EA4 matching)
        temporal_tolerance=30.0,    # seconds (EA4 matching)
        mlw_datum=0.5,             # MLW offset (EA3)
        sensor_offset_x=0.0,       # Configurable X offset (EA2, R9)
        sensor_offset_y=0.0,       # Configurable Y offset (EA2, R9)
        sensor_offset_z=1.5        # Configurable Z offset (EA2, R9)
    )
    
    # EA2: Align Coordinate Systems
    logger.info("\n" + "="*80)
    logger.info("EA2: ALIGN DATA TO SAME COORDINATE SYSTEM")
    logger.info("="*80)
    lidar_aligned, sonar_aligned = fusion.apply_coordinate_transformation(
        lidar_points, sonar_pings
    )
    
    # EA3: Normalize to MLW Datum
    logger.info("\n" + "="*80)
    logger.info("EA3: NORMALIZE TO MEAN LOW WATER (MLW) DATUM")
    logger.info("="*80)
    tidal_correction = 0.2  # Example: 20cm tidal variation
    lidar_normalized, sonar_normalized = fusion.normalize_to_mlw_datum(
        lidar_aligned, sonar_aligned, tidal_correction
    )
    
    # EA4: Merge Point Clouds
    logger.info("\n" + "="*80)
    logger.info("EA4: MERGE POINT CLOUDS")
    logger.info("="*80)
    water_points = fusion.align_datasets(lidar_normalized, sonar_normalized)
    
    if not water_points:
        logger.error("No merged points created!")
        return
    
    # EA5: Check and Fix Alignment Errors
    logger.info("\n" + "="*80)
    logger.info("EA5: CHECK AND FIX ALIGNMENT ERRORS")
    logger.info("="*80)
    water_points_corrected = fusion.check_alignment_errors(
        water_points, tolerance=0.5
    )
    
    # EA6: Validate Accuracy
    logger.info("\n" + "="*80)
    logger.info("EA6: VALIDATE COMBINED DATA ACCURACY")
    logger.info("="*80)
    metrics = fusion.validate_accuracy(water_points_corrected)
    
    # Display validation metrics
    logger.info("\nValidation Metrics:")
    logger.info(f"  Number of points: {metrics['num_points']}")
    logger.info(f"  Mean water depth: {metrics['depth_mean']:.2f}m ± {metrics['depth_std']:.2f}m")
    logger.info(f"  Depth range: {metrics['depth_min']:.2f}m to {metrics['depth_max']:.2f}m")
    logger.info(f"  Mean confidence: {metrics['confidence_mean']:.3f}")
    logger.info(f"  Vertical alignment: {metrics['vertical_alignment_mean']:.3f}m ± {metrics['vertical_alignment_std']:.3f}m")
    
    # EA7: Visualize Combined 3D Model
    logger.info("\n" + "="*80)
    logger.info("EA7: VISUALIZE COMBINED 3D MODEL")
    logger.info("="*80)
    viz_file = "/Users/tycrouch/Desktop/untitled folder 4/demo_water_surface_analysis.png"
    Visualizer.plot_water_surface(water_points_corrected, viz_file)
    
    # EA8: Document and Export
    logger.info("\n" + "="*80)
    logger.info("EA8: DOCUMENT PROCESS AND EXPORT RESULTS")
    logger.info("="*80)
    output_file = "/Users/tycrouch/Desktop/untitled folder 4/demo_fused_water_data.csv"
    fusion.export_results(water_points_corrected, output_file, include_metrics=True)
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("ENGINEERING ACTIONS DEMONSTRATION COMPLETE!")
    logger.info("="*80)
    logger.info("\nAll Engineering Actions Executed:")
    logger.info("  ✓ EA1: Collected Sonar and LiDAR data")
    logger.info("  ✓ EA2: Aligned to same coordinate system")
    logger.info("  ✓ EA3: Normalized to MLW datum with tidal corrections")
    logger.info("  ✓ EA4: Merged point clouds (spatial + temporal matching)")
    logger.info("  ✓ EA5: Checked and fixed alignment errors")
    logger.info("  ✓ EA6: Validated accuracy with statistical metrics")
    logger.info("  ✓ EA7: Visualized combined 3D model")
    logger.info("  ✓ EA8: Documented and exported all results")
    
    logger.info("\nOutput Files Generated:")
    logger.info(f"  • {output_file}")
    logger.info(f"  • {output_file.replace('.csv', '_metrics.csv')}")
    if fusion.alignment_errors:
        logger.info(f"  • {output_file.replace('.csv', '_alignment_errors.csv')}")
    logger.info(f"  • {viz_file}")
    
    logger.info("\nKey Metrics:")
    logger.info(f"  • {metrics['num_points']} merged points")
    logger.info(f"  • {metrics['depth_mean']:.2f}m mean water depth")
    logger.info(f"  • {metrics['vertical_alignment_mean']:.3f}m vertical alignment")
    logger.info(f"  • {len(fusion.alignment_errors)} alignment errors detected/corrected")
    
    logger.info("\n" + "="*80)
    logger.info("SUCCESS: Complete EA1-EA8 workflow demonstrated!")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()

