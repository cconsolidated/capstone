#!/usr/bin/env python3
"""
Create 3D Multi-View Animation from Real ILIDAR LAKE TEST Data
Shows your actual survey data progressing over time
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from real_data_parser import HYPACKRealParser, create_synthetic_matching_sonar
from lidar_sonar_fusion import DataFusion
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_all_raw_files():
    """Process all RAW files from the lake test"""
    
    logger.info("Processing all ILIDAR LAKE TEST RAW files...")
    
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    lidar_dir = data_dir / "HYPACK iLIDAR DATA" / "ILIDAR LAKE TEST"
    
    # Find all RAW files
    raw_files = sorted(lidar_dir.glob("*.RAW"))
    logger.info(f"Found {len(raw_files)} RAW files")
    
    all_surveys = []
    fusion = DataFusion(
        spatial_tolerance=10.0,
        temporal_tolerance=60.0,
        mlw_datum=0.5,
        sensor_offset_z=1.5
    )
    
    for i, raw_file in enumerate(raw_files):
        logger.info(f"\nProcessing {raw_file.name} ({i+1}/{len(raw_files)})")
        
        # Parse LiDAR data
        parser = HYPACKRealParser(raw_file)
        lidar_points = parser.parse_lidar_data()
        
        if not lidar_points:
            logger.warning(f"  No points in {raw_file.name}, skipping")
            continue
        
        # Create matching sonar
        sonar_pings = create_synthetic_matching_sonar(lidar_points, depth_offset=-5.0)
        
        # Process through EA2-EA6
        lidar_aligned, sonar_aligned = fusion.apply_coordinate_transformation(
            lidar_points, sonar_pings
        )
        lidar_norm, sonar_norm = fusion.normalize_to_mlw_datum(
            lidar_aligned, sonar_aligned, tidal_correction=0.2
        )
        water_points = fusion.align_datasets(lidar_norm, sonar_norm)
        water_points = fusion.check_alignment_errors(water_points, tolerance=1.0)
        
        if water_points:
            all_surveys.append({
                'filename': raw_file.name,
                'time_index': i,
                'timestamp': lidar_points[0].timestamp if lidar_points else None,
                'water_points': water_points,
                'num_points': len(water_points)
            })
            logger.info(f"  ✓ Processed: {len(water_points)} merged points")
    
    logger.info(f"\n✓ Total surveys processed: {len(all_surveys)}")
    return all_surveys

def create_real_data_multiview_animation(surveys, output_file, fps=2):
    """Create multi-view animation from real survey data"""
    
    logger.info(f"\nCreating multi-view animation with {len(surveys)} time steps...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Find global bounds across all surveys
    all_x, all_y, all_surface_z, all_bottom_z, all_depth = [], [], [], [], []
    
    for survey in surveys:
        for p in survey['water_points']:
            all_x.append(p.x)
            all_y.append(p.y)
            all_surface_z.append(p.surface_elevation)
            all_bottom_z.append(p.bottom_depth)
            all_depth.append(p.water_depth)
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_bottom_z), max(all_surface_z)
    depth_min, depth_max = min(all_depth), max(all_depth)
    
    # Center coordinates for better visualization
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    logger.info(f"Data bounds:")
    logger.info(f"  X: {x_min:.2f} to {x_max:.2f} (range: {x_max-x_min:.2f}m)")
    logger.info(f"  Y: {y_min:.2f} to {y_max:.2f} (range: {y_max-y_min:.2f}m)")
    logger.info(f"  Z: {z_min:.2f} to {z_max:.2f} (range: {z_max-z_min:.2f}m)")
    logger.info(f"  Depth: {depth_min:.2f} to {depth_max:.2f}m")
    
    def update(frame):
        """Update function for animation"""
        fig.clear()
        
        survey = surveys[frame]
        points = survey['water_points']
        
        # Extract data
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        surface_z = np.array([p.surface_elevation for p in points])
        bottom_z = np.array([p.bottom_depth for p in points])
        depth = np.array([p.water_depth for p in points])
        
        # Offset coordinates for better viewing
        x_plot = x - x_center
        y_plot = y - y_center
        
        # 3D View - Large left panel
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # Plot surface and bottom
        ax1.scatter(x_plot, y_plot, surface_z, c='dodgerblue', s=30, alpha=0.7, 
                   label='Water Surface (LiDAR)', edgecolors='blue', linewidth=0.5)
        ax1.scatter(x_plot, y_plot, bottom_z, c='coral', s=30, alpha=0.7,
                   label='Seafloor (Sonar)', edgecolors='darkred', linewidth=0.5)
        
        # Draw some connecting lines
        for i in range(0, len(x_plot), 20):
            ax1.plot([x_plot[i], x_plot[i]], [y_plot[i], y_plot[i]], 
                    [bottom_z[i], surface_z[i]], 'gray', alpha=0.2, linewidth=0.5)
        
        ax1.set_xlabel('X Offset (m)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Y Offset (m)', fontweight='bold', fontsize=11)
        ax1.set_zlabel('Elevation (m)', fontweight='bold', fontsize=11)
        ax1.set_title(f'3D Water Column - Real ILIDAR LAKE TEST\n{survey["filename"]}', 
                     fontweight='bold', fontsize=13)
        
        # Set limits
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax1.set_xlim(-x_range/2 * 1.1, x_range/2 * 1.1)
        ax1.set_ylim(-y_range/2 * 1.1, y_range/2 * 1.1)
        ax1.set_zlim(z_min, z_max)
        
        # Rotate view
        ax1.view_init(elev=20, azim=45 + frame * 5)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2D Surface view
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(x_plot, y_plot, c=surface_z, cmap='viridis', s=25,
                              vmin=min(all_surface_z), vmax=max(all_surface_z))
        ax2.set_xlabel('X Offset (m)', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Y Offset (m)', fontweight='bold', fontsize=10)
        ax2.set_title('Water Surface (LiDAR)', fontweight='bold', fontsize=11)
        ax2.set_xlim(-x_range/2 * 1.1, x_range/2 * 1.1)
        ax2.set_ylim(-y_range/2 * 1.1, y_range/2 * 1.1)
        plt.colorbar(scatter2, ax=ax2, label='Elevation (m)', fraction=0.046)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 2D Bottom view
        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(x_plot, y_plot, c=bottom_z, cmap='plasma', s=25,
                              vmin=min(all_bottom_z), vmax=max(all_bottom_z))
        ax3.set_xlabel('X Offset (m)', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Y Offset (m)', fontweight='bold', fontsize=10)
        ax3.set_title('Seafloor Depth (Sonar)', fontweight='bold', fontsize=11)
        ax3.set_xlim(-x_range/2 * 1.1, x_range/2 * 1.1)
        ax3.set_ylim(-y_range/2 * 1.1, y_range/2 * 1.1)
        plt.colorbar(scatter3, ax=ax3, label='Depth (m)', fraction=0.046)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # 2D Water depth view
        ax4 = fig.add_subplot(2, 3, 5)
        scatter4 = ax4.scatter(x_plot, y_plot, c=depth, cmap='coolwarm', s=25,
                              vmin=depth_min, vmax=depth_max)
        ax4.set_xlabel('X Offset (m)', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Y Offset (m)', fontweight='bold', fontsize=10)
        ax4.set_title('Water Column Depth', fontweight='bold', fontsize=11)
        ax4.set_xlim(-x_range/2 * 1.1, x_range/2 * 1.1)
        ax4.set_ylim(-y_range/2 * 1.1, y_range/2 * 1.1)
        plt.colorbar(scatter4, ax=ax4, label='Depth (m)', fraction=0.046)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        # Progress and statistics
        ax5 = fig.add_subplot(2, 3, 6)
        
        # Timeline
        times = list(range(len(surveys)))
        depths_over_time = [np.mean([p.water_depth for p in s['water_points']]) 
                           for s in surveys[:frame+1]]
        
        ax5.plot(times[:frame+1], depths_over_time, 'b-o', linewidth=2, markersize=6)
        ax5.axvline(x=frame, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax5.set_xlabel('Survey File Index', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Mean Water Depth (m)', fontweight='bold', fontsize=10)
        ax5.set_title('Depth Progression Over Surveys', fontweight='bold', fontsize=11)
        ax5.set_xlim(0, len(surveys)-1)
        ax5.set_ylim(depth_min*0.95, depth_max*1.05)
        ax5.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Survey: {frame+1}/{len(surveys)}
File: {survey['filename']}
Points: {survey['num_points']}
Mean Depth: {np.mean(depth):.2f}m
Std Dev: {np.std(depth):.2f}m
Time: {survey['timestamp'].strftime('%H:%M:%S') if survey['timestamp'] else 'N/A'}"""
        
        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add main title
        fig.suptitle('REAL ILIDAR LAKE TEST DATA - 3D Multi-View Analysis\nAugust 22-23, 2025 Survey',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig,
    
    # Create animation
    logger.info("Rendering animation frames...")
    anim = FuncAnimation(fig, update, frames=len(surveys), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save
    logger.info(f"Saving animation to {output_file}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=120)
    
    logger.info(f"✓ Animation saved! ({len(surveys)} frames at {fps} fps)")
    plt.close()
    
    return anim

def main():
    """Create real data animation"""
    
    logger.info("\n" + "="*80)
    logger.info("CREATING 3D MULTI-VIEW ANIMATION FROM REAL LAKE DATA")
    logger.info("="*80 + "\n")
    
    # Process all survey files
    surveys = process_all_raw_files()
    
    if not surveys:
        logger.error("No surveys to animate!")
        return
    
    # Create animation
    output_file = "/Users/tycrouch/Desktop/untitled folder 4/real_lake_data_3d_animation.gif"
    create_real_data_multiview_animation(surveys, output_file, fps=1)
    
    logger.info("\n" + "="*80)
    logger.info("SUCCESS! Real Data Animation Created")
    logger.info("="*80)
    logger.info(f"\nOutput: {output_file}")
    logger.info(f"Frames: {len(surveys)}")
    logger.info(f"Total points across all surveys: {sum(s['num_points'] for s in surveys)}")
    logger.info("\nThis animation shows YOUR ACTUAL field data from the lake test!")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()

