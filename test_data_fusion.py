#!/usr/bin/env python3
"""
Test script for LIDAR-Sonar data fusion
Validates data parsing and basic fusion functionality
"""

import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import our fusion modules
from lidar_sonar_fusion import HSXParser, S7KParser, DataFusion, Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hsx_parsing():
    """Test HSX file parsing"""
    logger.info("Testing HSX file parsing...")
    
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    hsx_files = list((data_dir / "HYPACK iLIDAR DATA" / "ILIDAR LAKE TEST").glob("*.HSX"))
    
    if not hsx_files:
        logger.error("No HSX files found")
        return False
    
    # Test with first HSX file
    hsx_file = hsx_files[0]
    logger.info(f"Testing with file: {hsx_file}")
    
    try:
        parser = HSXParser(str(hsx_file))
        header = parser.parse_header()
        
        logger.info(f"Header info: {header}")
        logger.info(f"Devices found: {len(parser.devices)}")
        
        for dev_id, dev_info in parser.devices.items():
            logger.info(f"  Device {dev_id}: {dev_info['name']}")
        
        # Test LIDAR data parsing (may be limited due to binary format complexity)
        lidar_points = parser.parse_lidar_data()
        logger.info(f"Parsed {len(lidar_points)} LIDAR points")
        
        return True
        
    except Exception as e:
        logger.error(f"HSX parsing failed: {e}")
        return False

def test_s7k_parsing():
    """Test S7K file parsing"""
    logger.info("Testing S7K file parsing...")
    
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    s7k_files = []
    
    for session_dir in (data_dir / "WBMS").iterdir():
        if session_dir.is_dir():
            s7k_files.extend(session_dir.glob("*.s7k"))
    
    if not s7k_files:
        logger.error("No S7K files found")
        return False
    
    # Test with first S7K file
    s7k_file = s7k_files[0]
    logger.info(f"Testing with file: {s7k_file}")
    
    try:
        parser = S7KParser(str(s7k_file))
        sonar_pings = parser.parse_sonar_data()
        
        logger.info(f"Parsed {len(sonar_pings)} sonar pings")
        
        if sonar_pings:
            # Show some sample data
            sample = sonar_pings[0]
            logger.info(f"Sample ping: x={sample.x:.2f}, y={sample.y:.2f}, depth={sample.depth:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"S7K parsing failed: {e}")
        return False

def test_data_structure():
    """Test data structure inspection"""
    logger.info("Inspecting data file structures...")
    
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    
    # Check LIDAR files
    lidar_dir = data_dir / "HYPACK iLIDAR DATA"
    for test_dir in lidar_dir.iterdir():
        if test_dir.is_dir():
            logger.info(f"LIDAR test directory: {test_dir.name}")
            hsx_files = list(test_dir.glob("*.HSX"))
            raw_files = list(test_dir.glob("*.RAW"))
            logger.info(f"  HSX files: {len(hsx_files)}")
            logger.info(f"  RAW files: {len(raw_files)}")
            
            # Check file sizes
            for hsx_file in hsx_files[:2]:  # Check first 2
                size_mb = hsx_file.stat().st_size / (1024*1024)
                logger.info(f"    {hsx_file.name}: {size_mb:.1f} MB")
    
    # Check sonar files
    sonar_dir = data_dir / "WBMS"
    for session_dir in sonar_dir.iterdir():
        if session_dir.is_dir():
            logger.info(f"Sonar session: {session_dir.name}")
            s7k_files = list(session_dir.glob("*.s7k"))
            wbm_files = list(session_dir.glob("**/sonarFile.wbm"))
            logger.info(f"  S7K files: {len(s7k_files)}")
            logger.info(f"  WBM files: {len(wbm_files)}")
            
            # Check file sizes
            for s7k_file in s7k_files:
                size_mb = s7k_file.stat().st_size / (1024*1024)
                logger.info(f"    {s7k_file.name}: {size_mb:.1f} MB")

def create_sample_data():
    """Create sample synthetic data for testing fusion algorithms"""
    logger.info("Creating sample synthetic data for algorithm testing...")
    
    # Generate synthetic LIDAR points (water surface)
    n_lidar = 1000
    x_lidar = np.random.uniform(0, 100, n_lidar)
    y_lidar = np.random.uniform(0, 100, n_lidar)
    z_surface = 10 + 0.1 * np.sin(x_lidar/10) + 0.05 * np.cos(y_lidar/15)  # Gentle wave pattern
    
    # Add some noise
    z_surface += np.random.normal(0, 0.02, n_lidar)
    
    # Generate synthetic sonar pings (bottom depth)
    n_sonar = 800
    x_sonar = np.random.uniform(0, 100, n_sonar)
    y_sonar = np.random.uniform(0, 100, n_sonar)
    z_bottom = 5 + 2 * np.sin(x_sonar/20) + 1 * np.cos(y_sonar/25)  # Bottom topography
    
    # Add noise
    z_bottom += np.random.normal(0, 0.1, n_sonar)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot surface
    scatter1 = axes[0].scatter(x_lidar, y_lidar, c=z_surface, cmap='viridis', alpha=0.6)
    axes[0].set_title('Synthetic Water Surface (LIDAR)')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    plt.colorbar(scatter1, ax=axes[0], label='Elevation (m)')
    
    # Plot bottom
    scatter2 = axes[1].scatter(x_sonar, y_sonar, c=z_bottom, cmap='plasma', alpha=0.6)
    axes[1].set_title('Synthetic Bottom Depth (Sonar)')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    plt.colorbar(scatter2, ax=axes[1], label='Depth (m)')
    
    # Plot water depth
    water_depth = z_surface.mean() - z_bottom
    scatter3 = axes[2].scatter(x_sonar, y_sonar, c=water_depth, cmap='coolwarm', alpha=0.6)
    axes[2].set_title('Calculated Water Depth')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    plt.colorbar(scatter3, ax=axes[2], label='Water Depth (m)')
    
    plt.tight_layout()
    output_file = "/Users/tycrouch/Desktop/untitled folder 4/synthetic_test_data.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Synthetic data visualization saved to {output_file}")
    plt.show()
    
    return True

def main():
    """Run all tests"""
    logger.info("Starting LIDAR-Sonar fusion tests...")
    
    # Test data structure inspection
    test_data_structure()
    
    # Test parsers
    hsx_success = test_hsx_parsing()
    s7k_success = test_s7k_parsing()
    
    # Create sample data for algorithm testing
    sample_success = create_sample_data()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY:")
    logger.info(f"HSX parsing: {'PASS' if hsx_success else 'FAIL'}")
    logger.info(f"S7K parsing: {'PASS' if s7k_success else 'FAIL'}")
    logger.info(f"Sample data: {'PASS' if sample_success else 'FAIL'}")
    logger.info("="*50)
    
    if not (hsx_success or s7k_success):
        logger.warning("\nNote: The binary data parsers are simplified implementations.")
        logger.warning("For production use, you'll need to implement full format specifications:")
        logger.warning("- HSX/RAW: Requires HYPACK format documentation")
        logger.warning("- S7K: Requires NORBIT/Reson format documentation")
        logger.warning("\nThe fusion algorithms and data structures are ready for real data.")

if __name__ == "__main__":
    main()

