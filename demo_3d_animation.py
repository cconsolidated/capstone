#!/usr/bin/env python3
"""
3D Animated Visualization - Above and Below Water Over Time
Shows temporal changes in both water surface (LiDAR) and seafloor (Sonar) as a video
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from lidar_sonar_fusion import (
    LidarPoint, SonarPing, DataFusion
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_time_series_data(num_timesteps=10, points_per_survey=300):
    """Generate a time series of surveys showing evolution"""
    
    logger.info(f"Generating {num_timesteps} time steps for animation...")
    
    all_surveys = []
    
    for t in range(num_timesteps):
        base_time = datetime.datetime.now() + datetime.timedelta(hours=t)
        
        # Survey area
        x = np.random.uniform(-25, 25, points_per_survey)
        y = np.random.uniform(-25, 25, points_per_survey)
        
        # Water surface changes over time (waves, boat)
        wave_phase = t * 0.3
        surface_z = 2.0 + 0.2 * np.sin(x/5 + wave_phase) + 0.1 * np.cos(y/5 + wave_phase)
        
        # Boat moves across the area
        boat_x = -20 + (t / num_timesteps) * 40  # Moves from -20 to +20
        boat_y = -15 + (t / num_timesteps) * 30  # Moves from -15 to +15
        
        if 2 <= t <= 7:  # Boat present during middle of sequence
            distance_from_boat = np.sqrt((x - boat_x)**2 + (y - boat_y)**2)
            boat_wake = np.where(distance_from_boat < 8, 
                                 -0.4 * np.exp(-distance_from_boat/4),
                                 0)
            surface_z += boat_wake
        
        surface_z += np.random.normal(0, 0.03, points_per_survey)
        
        # Seafloor changes over time (sediment deposition)
        seafloor_depth = -5.0 - 0.5 * np.sin(x/8) - 0.3 * np.cos(y/8)
        
        # Sediment accumulates progressively in one area
        sediment_progress = min(t / num_timesteps, 1.0)
        sediment_area = (x > 5) & (y > 5)
        seafloor_depth[sediment_area] += 0.8 * sediment_progress  # Gradual rise
        
        seafloor_depth += np.random.normal(0, 0.05, points_per_survey)
        
        # Create point objects
        lidar_points = [
            LidarPoint(x[i], y[i], surface_z[i], 
                      np.random.uniform(100, 250),
                      base_time + datetime.timedelta(seconds=i*0.1))
            for i in range(points_per_survey)
        ]
        
        sonar_pings = [
            SonarPing(x[i], y[i], seafloor_depth[i],
                     np.random.uniform(100, 200),
                     base_time + datetime.timedelta(seconds=i*0.1),
                     np.random.uniform(-60, 60))
            for i in range(points_per_survey)
        ]
        
        all_surveys.append({
            'time': t,
            'lidar': lidar_points,
            'sonar': sonar_pings,
            'boat_pos': (boat_x, boat_y) if 2 <= t <= 7 else None
        })
    
    logger.info(f"✓ Generated {num_timesteps} time steps")
    return all_surveys

def process_time_series(all_surveys):
    """Process all surveys through EA1-EA8"""
    
    logger.info("Processing all time steps through EA1-EA8...")
    
    fusion = DataFusion(
        spatial_tolerance=3.0,
        temporal_tolerance=30.0,
        mlw_datum=0.5,
        sensor_offset_z=1.5
    )
    
    processed_surveys = []
    
    for i, survey in enumerate(all_surveys):
        # EA2-4: Transform, normalize, merge
        lidar_aligned, sonar_aligned = fusion.apply_coordinate_transformation(
            survey['lidar'], survey['sonar']
        )
        lidar_norm, sonar_norm = fusion.normalize_to_mlw_datum(
            lidar_aligned, sonar_aligned, tidal_correction=0.2
        )
        water_points = fusion.align_datasets(lidar_norm, sonar_norm)
        
        # EA5-6: Validate
        water_points = fusion.check_alignment_errors(water_points, tolerance=0.5)
        
        processed_surveys.append({
            'time': survey['time'],
            'water_points': water_points,
            'boat_pos': survey['boat_pos']
        })
        
        logger.info(f"  Processed timestep {i+1}/{len(all_surveys)}: {len(water_points)} points")
    
    logger.info("✓ All time steps processed")
    return processed_surveys

def create_3d_animation(processed_surveys, output_file, fps=2):
    """Create animated 3D visualization"""
    
    logger.info("Creating 3D animation...")
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find global bounds for consistent scaling
    all_x, all_y = [], []
    all_surface_z, all_bottom_z = [], []
    
    for survey in processed_surveys:
        for p in survey['water_points']:
            all_x.append(p.x)
            all_y.append(p.y)
            all_surface_z.append(p.surface_elevation)
            all_bottom_z.append(p.bottom_depth)
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_bottom_z), max(all_surface_z)
    
    def update(frame):
        """Update function for animation"""
        ax.clear()
        
        survey = processed_surveys[frame]
        points = survey['water_points']
        
        # Extract coordinates
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        surface_z = np.array([p.surface_elevation for p in points])
        bottom_z = np.array([p.bottom_depth for p in points])
        
        # Plot water surface (LIDAR - above water)
        ax.scatter(x, y, surface_z, 
                  c='dodgerblue', marker='o', s=30, alpha=0.6, 
                  label='Water Surface (LiDAR)', edgecolors='blue', linewidth=0.5)
        
        # Plot seafloor (SONAR - below water)
        ax.scatter(x, y, bottom_z, 
                  c='coral', marker='o', s=30, alpha=0.6,
                  label='Seafloor (Sonar)', edgecolors='darkred', linewidth=0.5)
        
        # Draw vertical lines connecting surface to bottom (water column)
        for i in range(0, len(x), 10):  # Draw every 10th line to avoid clutter
            ax.plot([x[i], x[i]], [y[i], y[i]], [bottom_z[i], surface_z[i]], 
                   'gray', alpha=0.2, linewidth=0.5)
        
        # Add boat marker if present
        if survey['boat_pos']:
            boat_x, boat_y = survey['boat_pos']
            # Draw boat at water surface
            ax.scatter([boat_x], [boat_y], [2.0], 
                      c='red', marker='^', s=500, 
                      label='Boat', edgecolors='darkred', linewidth=2)
            # Draw wake indicator
            circle = plt.Circle((boat_x, boat_y), 8, color='red', fill=False, linewidth=2)
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Elevation (meters)', fontsize=12, fontweight='bold')
        
        # Add time indicator
        time_hours = survey['time']
        ax.set_title(f'3D Water Column Visualization - Time: T0 + {time_hours}h\n' +
                    f'Blue = Water Surface (LiDAR) | Red = Seafloor (Sonar)',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set consistent limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Rotate view angle over time for dramatic effect
        angle = 45 + frame * 3  # Rotate 3 degrees per frame
        ax.view_init(elev=20, azim=angle)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add text annotation showing statistics
        mean_depth = np.mean([p.water_depth for p in points])
        num_points = len(points)
        
        info_text = f'Points: {num_points}\nMean Depth: {mean_depth:.2f}m'
        ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 family='monospace')
        
        return ax,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(processed_surveys), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save as GIF
    logger.info(f"Saving animation to {output_file}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=100)
    
    logger.info(f"✓ Animation saved! ({len(processed_surveys)} frames at {fps} fps)")
    
    plt.close()
    
    return anim

def create_side_by_side_animation(processed_surveys, output_file, fps=2):
    """Create animation with multiple views"""
    
    logger.info("Creating multi-view animation...")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Find global bounds
    all_x, all_y, all_surface_z, all_bottom_z, all_depth = [], [], [], [], []
    
    for survey in processed_surveys:
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
    
    def update(frame):
        """Update function for multi-view animation"""
        fig.clear()
        
        survey = processed_surveys[frame]
        points = survey['water_points']
        
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        surface_z = np.array([p.surface_elevation for p in points])
        bottom_z = np.array([p.bottom_depth for p in points])
        depth = np.array([p.water_depth for p in points])
        
        # 3D View 1: Full water column
        ax1 = fig.add_subplot(2, 3, (1, 4), projection='3d')
        ax1.scatter(x, y, surface_z, c='dodgerblue', s=20, alpha=0.6, label='Surface')
        ax1.scatter(x, y, bottom_z, c='coral', s=20, alpha=0.6, label='Bottom')
        
        if survey['boat_pos']:
            boat_x, boat_y = survey['boat_pos']
            ax1.scatter([boat_x], [boat_y], [2.0], c='red', marker='^', s=400, label='Boat')
        
        ax1.set_xlabel('X (m)', fontweight='bold')
        ax1.set_ylabel('Y (m)', fontweight='bold')
        ax1.set_zlabel('Z (m)', fontweight='bold')
        ax1.set_title(f'3D Water Column View\nTime: T0+{survey["time"]}h', fontweight='bold')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax1.view_init(elev=20, azim=45 + frame * 3)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2D View 1: Surface elevation
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(x, y, c=surface_z, cmap='viridis', s=30, vmin=min(all_surface_z), vmax=max(all_surface_z))
        ax2.set_xlabel('X (m)', fontweight='bold')
        ax2.set_ylabel('Y (m)', fontweight='bold')
        ax2.set_title('Water Surface (LiDAR)', fontweight='bold')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        plt.colorbar(scatter2, ax=ax2, label='Elevation (m)')
        ax2.grid(True, alpha=0.3)
        
        # 2D View 2: Bottom depth
        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(x, y, c=bottom_z, cmap='plasma', s=30, vmin=min(all_bottom_z), vmax=max(all_bottom_z))
        ax3.set_xlabel('X (m)', fontweight='bold')
        ax3.set_ylabel('Y (m)', fontweight='bold')
        ax3.set_title('Seafloor Depth (Sonar)', fontweight='bold')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        plt.colorbar(scatter3, ax=ax3, label='Depth (m)')
        ax3.grid(True, alpha=0.3)
        
        # 2D View 3: Water depth
        ax4 = fig.add_subplot(2, 3, 5)
        scatter4 = ax4.scatter(x, y, c=depth, cmap='coolwarm', s=30, vmin=depth_min, vmax=depth_max)
        ax4.set_xlabel('X (m)', fontweight='bold')
        ax4.set_ylabel('Y (m)', fontweight='bold')
        ax4.set_title('Water Column Depth', fontweight='bold')
        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        plt.colorbar(scatter4, ax=ax4, label='Depth (m)')
        ax4.grid(True, alpha=0.3)
        
        # Timeline plot
        ax5 = fig.add_subplot(2, 3, 6)
        times = [s['time'] for s in processed_surveys[:frame+1]]
        depths = [np.mean([p.water_depth for p in s['water_points']]) for s in processed_surveys[:frame+1]]
        ax5.plot(times, depths, 'b-o', linewidth=2, markersize=8)
        ax5.set_xlabel('Time (hours)', fontweight='bold')
        ax5.set_ylabel('Mean Water Depth (m)', fontweight='bold')
        ax5.set_title('Depth Over Time', fontweight='bold')
        ax5.set_xlim(0, len(processed_surveys)-1)
        ax5.set_ylim(depth_min*0.95, depth_max*1.05)
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=frame, color='red', linestyle='--', alpha=0.5, label='Current')
        ax5.legend()
        
        plt.tight_layout()
        
        return fig,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(processed_surveys), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save
    logger.info(f"Saving multi-view animation to {output_file}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=120)
    
    logger.info(f"✓ Multi-view animation saved!")
    
    plt.close()
    
    return anim

def main():
    """Create 3D animated visualizations"""
    
    logger.info("\n" + "="*80)
    logger.info("3D ANIMATED WATER COLUMN VISUALIZATION")
    logger.info("Showing Above and Below Water Changes Over Time")
    logger.info("="*80 + "\n")
    
    # Generate time series data
    all_surveys = generate_time_series_data(num_timesteps=15, points_per_survey=250)
    
    # Process through EA1-EA8
    processed_surveys = process_time_series(all_surveys)
    
    # Create animations
    logger.info("\n" + "="*80)
    logger.info("Creating Animations...")
    logger.info("="*80 + "\n")
    
    # Simple 3D rotating view
    output_file1 = "/Users/tycrouch/Desktop/untitled folder 4/3d_water_column_animation.gif"
    create_3d_animation(processed_surveys, output_file1, fps=2)
    
    # Multi-view animation
    output_file2 = "/Users/tycrouch/Desktop/untitled folder 4/3d_multiview_animation.gif"
    create_side_by_side_animation(processed_surveys, output_file2, fps=2)
    
    logger.info("\n" + "="*80)
    logger.info("SUCCESS! Animations Created:")
    logger.info("="*80)
    logger.info(f"  ✓ {output_file1}")
    logger.info(f"  ✓ {output_file2}")
    logger.info("\nThese animations show:")
    logger.info("  • Water surface (blue) - LiDAR above water")
    logger.info("  • Seafloor (red) - Sonar below water")
    logger.info("  • Boat movement over time")
    logger.info("  • Sediment deposition progression")
    logger.info("  • Complete water column changes")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    main()

