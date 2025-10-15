#!/usr/bin/env python3
"""
Temporal Change Detection Demonstration
Shows how water surface and seafloor change over time across multiple surveys
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lidar_sonar_fusion import (
    LidarPoint, SonarPing, DataFusion, WaterSurfacePoint
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_survey_data(time_offset_hours=0, boat_present=False, sediment_deposited=False):
    """Generate survey data for a specific time period with environmental changes"""
    
    base_time = datetime.datetime.now() + datetime.timedelta(hours=time_offset_hours)
    
    # Survey area 50x50m
    num_points = 400
    x = np.random.uniform(-25, 25, num_points)
    y = np.random.uniform(-25, 25, num_points)
    
    # Water surface at z=2m with waves
    wave_phase = time_offset_hours * 0.5  # Waves change over time
    surface_z = 2.0 + 0.2 * np.sin(x/5 + wave_phase) + 0.1 * np.cos(y/5 + wave_phase)
    
    # Add boat disturbance if present
    if boat_present:
        # Boat creates a wake in the center
        boat_x, boat_y = 5, 5
        distance_from_boat = np.sqrt((x - boat_x)**2 + (y - boat_y)**2)
        boat_wake = np.where(distance_from_boat < 10, 
                             -0.3 * np.exp(-distance_from_boat/5),  # Depression from boat
                             0)
        surface_z += boat_wake
    
    surface_z += np.random.normal(0, 0.05, num_points)
    
    # LiDAR points
    lidar_points = []
    for i in range(num_points):
        point = LidarPoint(
            x=x[i], y=y[i], z=surface_z[i],
            intensity=np.random.uniform(50, 250),
            timestamp=base_time + datetime.timedelta(seconds=i*0.1)
        )
        lidar_points.append(point)
    
    # Seafloor depth
    seafloor_depth = -5.0 - 0.5 * np.sin(x/8) - 0.3 * np.cos(y/8)
    
    # Add sediment deposition if present
    if sediment_deposited:
        # Sediment accumulation in one corner
        sediment_area = (x > 0) & (y > 0)
        seafloor_depth[sediment_area] += 0.5  # Seafloor rises 0.5m due to sediment
    
    seafloor_depth += np.random.normal(0, 0.1, num_points)
    
    # Sonar pings
    sonar_pings = []
    for i in range(num_points):
        ping = SonarPing(
            x=x[i], y=y[i], depth=seafloor_depth[i],
            intensity=np.random.uniform(100, 200),
            timestamp=base_time + datetime.timedelta(seconds=i*0.1),
            beam_angle=np.random.uniform(-60, 60)
        )
        sonar_pings.append(ping)
    
    return lidar_points, sonar_pings

def process_survey(lidar_points, sonar_pings, survey_name):
    """Process a single survey through EA1-EA8"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Survey: {survey_name}")
    logger.info(f"{'='*80}")
    
    fusion = DataFusion(
        spatial_tolerance=3.0,
        temporal_tolerance=30.0,
        mlw_datum=0.5,
        sensor_offset_z=1.5
    )
    
    # EA2-3: Transform and normalize
    lidar_aligned, sonar_aligned = fusion.apply_coordinate_transformation(lidar_points, sonar_pings)
    lidar_norm, sonar_norm = fusion.normalize_to_mlw_datum(lidar_aligned, sonar_aligned, tidal_correction=0.2)
    
    # EA4: Merge
    water_points = fusion.align_datasets(lidar_norm, sonar_norm)
    
    # EA5-6: Validate
    water_points = fusion.check_alignment_errors(water_points, tolerance=0.5)
    metrics = fusion.validate_accuracy(water_points)
    
    logger.info(f"Survey {survey_name}: {len(water_points)} points merged")
    logger.info(f"  Mean water depth: {metrics['depth_mean']:.2f}m ± {metrics['depth_std']:.2f}m")
    
    return water_points, metrics

def create_change_map(survey1_points, survey2_points, grid_resolution=1.0):
    """Create a change detection map between two surveys"""
    
    # Extract coordinates and depths from both surveys
    coords1 = np.array([[p.x, p.y] for p in survey1_points])
    depths1 = np.array([p.water_depth for p in survey1_points])
    
    coords2 = np.array([[p.x, p.y] for p in survey2_points])
    depths2 = np.array([p.water_depth for p in survey2_points])
    
    # Create grid
    x_min = min(coords1[:, 0].min(), coords2[:, 0].min())
    x_max = max(coords1[:, 0].max(), coords2[:, 0].max())
    y_min = min(coords1[:, 1].min(), coords2[:, 1].min())
    y_max = max(coords1[:, 1].max(), coords2[:, 1].max())
    
    xi = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    yi = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate both surveys onto same grid
    from scipy.interpolate import griddata
    
    grid1 = griddata(coords1, depths1, (xi_grid, yi_grid), method='linear')
    grid2 = griddata(coords2, depths2, (xi_grid, yi_grid), method='linear')
    
    # Calculate change (difference)
    change_map = grid2 - grid1
    
    return xi_grid, yi_grid, change_map, grid1, grid2

def visualize_temporal_changes():
    """Main visualization showing changes over time"""
    
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL CHANGE DETECTION DEMONSTRATION")
    logger.info("Showing how EA1-EA8 detects changes across multiple surveys")
    logger.info("="*80 + "\n")
    
    # Generate three time periods
    logger.info("Generating Survey 1: Baseline (T0)")
    lidar1, sonar1 = generate_survey_data(time_offset_hours=0, boat_present=False, sediment_deposited=False)
    
    logger.info("Generating Survey 2: Boat passing through (T0 + 2 hours)")
    lidar2, sonar2 = generate_survey_data(time_offset_hours=2, boat_present=True, sediment_deposited=False)
    
    logger.info("Generating Survey 3: Sediment deposition (T0 + 4 hours)")
    lidar3, sonar3 = generate_survey_data(time_offset_hours=4, boat_present=False, sediment_deposited=True)
    
    # Process all surveys
    survey1_points, metrics1 = process_survey(lidar1, sonar1, "Baseline")
    survey2_points, metrics2 = process_survey(lidar2, sonar2, "With Boat")
    survey3_points, metrics3 = process_survey(lidar3, sonar3, "With Sediment")
    
    # Create change maps
    logger.info("\n" + "="*80)
    logger.info("Computing change detection maps...")
    logger.info("="*80)
    
    xi, yi, change_1_to_2, grid1, grid2 = create_change_map(survey1_points, survey2_points)
    _, _, change_1_to_3, _, grid3 = create_change_map(survey1_points, survey3_points)
    _, _, change_2_to_3, _, _ = create_change_map(survey2_points, survey3_points)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Individual surveys
    ax1 = fig.add_subplot(3, 3, 1)
    im1 = ax1.contourf(xi, yi, grid1, levels=20, cmap='viridis')
    ax1.set_title('Survey 1: Baseline (T0)\nWater Depth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Depth (m)')
    
    ax2 = fig.add_subplot(3, 3, 2)
    im2 = ax2.contourf(xi, yi, grid2, levels=20, cmap='viridis')
    ax2.set_title('Survey 2: Boat Present (T0+2h)\nWater Depth', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Depth (m)')
    
    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.contourf(xi, yi, grid3, levels=20, cmap='viridis')
    ax3.set_title('Survey 3: Sediment Deposited (T0+4h)\nWater Depth', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Y (m)', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Depth (m)')
    
    # Row 2: Change detection maps
    ax4 = fig.add_subplot(3, 3, 4)
    im4 = ax4.contourf(xi, yi, change_1_to_2, levels=20, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax4.set_title('CHANGE: Survey 1 → Survey 2\n(Boat Wake Detection)', fontsize=14, fontweight='bold', color='red')
    ax4.set_xlabel('X (m)', fontsize=12)
    ax4.set_ylabel('Y (m)', fontsize=12)
    plt.colorbar(im4, ax=ax4, label='Change (m)')
    ax4.scatter([5], [5], c='red', s=200, marker='x', linewidths=3, label='Boat Location')
    ax4.legend(fontsize=10)
    
    ax5 = fig.add_subplot(3, 3, 5)
    im5 = ax5.contourf(xi, yi, change_1_to_3, levels=20, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax5.set_title('CHANGE: Survey 1 → Survey 3\n(Sediment Detection)', fontsize=14, fontweight='bold', color='red')
    ax5.set_xlabel('X (m)', fontsize=12)
    ax5.set_ylabel('Y (m)', fontsize=12)
    plt.colorbar(im5, ax=ax5, label='Change (m)')
    
    ax6 = fig.add_subplot(3, 3, 6)
    im6 = ax6.contourf(xi, yi, change_2_to_3, levels=20, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax6.set_title('CHANGE: Survey 2 → Survey 3\n(Boat Cleared, Sediment Remains)', fontsize=14, fontweight='bold', color='red')
    ax6.set_xlabel('X (m)', fontsize=12)
    ax6.set_ylabel('Y (m)', fontsize=12)
    plt.colorbar(im6, ax=ax6, label='Change (m)')
    
    # Row 3: Time series and statistics
    ax7 = fig.add_subplot(3, 3, 7)
    surveys = ['Survey 1\nBaseline', 'Survey 2\nBoat', 'Survey 3\nSediment']
    mean_depths = [metrics1['depth_mean'], metrics2['depth_mean'], metrics3['depth_mean']]
    std_depths = [metrics1['depth_std'], metrics2['depth_std'], metrics3['depth_std']]
    
    ax7.bar(surveys, mean_depths, yerr=std_depths, capsize=10, alpha=0.7, color=['blue', 'orange', 'green'])
    ax7.set_ylabel('Mean Water Depth (m)', fontsize=12)
    ax7.set_title('Depth Comparison Over Time', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, (depth, std) in enumerate(zip(mean_depths, std_depths)):
        ax7.text(i, depth + std + 0.1, f'{depth:.2f}m', ha='center', fontsize=10, fontweight='bold')
    
    ax8 = fig.add_subplot(3, 3, 8)
    change_stats = [
        np.nanmean(np.abs(change_1_to_2)),
        np.nanmean(np.abs(change_1_to_3)),
        np.nanmean(np.abs(change_2_to_3))
    ]
    change_labels = ['1→2\n(Boat)', '1→3\n(Sediment)', '2→3\n(Recovery)']
    bars = ax8.bar(change_labels, change_stats, color=['red', 'orange', 'yellow'], alpha=0.7)
    ax8.set_ylabel('Mean Absolute Change (m)', fontsize=12)
    ax8.set_title('Magnitude of Changes Detected', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, val in enumerate(change_stats):
        ax8.text(i, val + 0.005, f'{val:.3f}m', ha='center', fontsize=10, fontweight='bold')
    
    # Text summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    TEMPORAL CHANGE DETECTION RESULTS
    ═══════════════════════════════════
    
    Survey 1 (Baseline):
      • Time: T0
      • Points: {len(survey1_points)}
      • Mean depth: {metrics1['depth_mean']:.2f}m
      • Status: Normal conditions
    
    Survey 2 (Boat):
      • Time: T0 + 2 hours
      • Points: {len(survey2_points)}
      • Mean depth: {metrics2['depth_mean']:.2f}m
      • Change detected: Boat wake (center)
      • Max change: {np.nanmax(np.abs(change_1_to_2)):.2f}m
    
    Survey 3 (Sediment):
      • Time: T0 + 4 hours
      • Points: {len(survey3_points)}
      • Mean depth: {metrics3['depth_mean']:.2f}m
      • Change detected: Sediment (upper right)
      • Max change: {np.nanmax(np.abs(change_1_to_3)):.2f}m
    
    APPLICATIONS:
      ✓ Boat/vessel tracking
      ✓ Sediment movement monitoring
      ✓ Erosion/deposition detection
      ✓ Water level changes
      ✓ Underwater object detection
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    output_file = "/Users/tycrouch/Desktop/untitled folder 4/temporal_change_detection.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Temporal change visualization saved to {output_file}")
    
    plt.show()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("CHANGE DETECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Survey 1→2 Change: {change_stats[0]:.3f}m (Boat detected)")
    logger.info(f"Survey 1→3 Change: {change_stats[1]:.3f}m (Sediment detected)")
    logger.info(f"Survey 2→3 Change: {change_stats[2]:.3f}m (Environment recovery)")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    visualize_temporal_changes()

