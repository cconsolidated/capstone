#!/usr/bin/env python3
"""
Water Change Detection System
Detects boats, chemical spills, and other anomalies using combined LIDAR-Sonar data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# import cv2  # Optional - only needed for advanced morphological operations
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our base classes
from lidar_sonar_fusion import WaterSurfacePoint, DataFusion, Visualizer

logger = logging.getLogger(__name__)

@dataclass
class WaterAnomaly:
    """Detected water anomaly (boat, chemical spill, etc.)"""
    anomaly_type: str  # 'vessel', 'chemical_spill', 'debris', 'unknown'
    center_x: float
    center_y: float
    area: float  # square meters
    confidence: float
    timestamp: datetime
    properties: Dict  # Additional properties specific to anomaly type
    affected_points: List[WaterSurfacePoint]

@dataclass
class ChangeEvent:
    """Temporal change event in water"""
    event_type: str  # 'new_object', 'object_moved', 'object_removed', 'surface_change'
    location: Tuple[float, float]
    magnitude: float
    timestamp: datetime
    duration: Optional[timedelta]
    description: str

class WaterChangeDetector:
    """Advanced change detection for water surfaces"""
    
    def __init__(self, grid_resolution: float = 0.5, temporal_window: int = 10):
        self.grid_resolution = grid_resolution
        self.temporal_window = temporal_window
        self.baseline_established = False
        self.baseline_surface = None
        self.baseline_depth = None
        self.historical_data = []
        
    def establish_baseline(self, water_points: List[WaterSurfacePoint]):
        """Establish baseline water conditions for change detection"""
        logger.info("Establishing baseline water conditions...")
        
        if not water_points:
            logger.warning("No water points provided for baseline")
            return
        
        # Create regular grid
        coords = np.array([[p.x, p.y] for p in water_points])
        surface_elevations = np.array([p.surface_elevation for p in water_points])
        water_depths = np.array([p.water_depth for p in water_points])
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Create baseline grids
        self.x_grid = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        self.y_grid = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        xi, yi = np.meshgrid(self.x_grid, self.y_grid)
        
        # Interpolate baseline surfaces
        from scipy.interpolate import griddata
        self.baseline_surface = griddata(coords, surface_elevations, (xi, yi), method='linear')
        self.baseline_depth = griddata(coords, water_depths, (xi, yi), method='linear')
        
        # Fill NaN values with mean
        self.baseline_surface = np.nan_to_num(self.baseline_surface, nan=np.nanmean(self.baseline_surface))
        self.baseline_depth = np.nan_to_num(self.baseline_depth, nan=np.nanmean(self.baseline_depth))
        
        # Apply smoothing to reduce noise
        self.baseline_surface = gaussian_filter(self.baseline_surface, sigma=1.0)
        self.baseline_depth = gaussian_filter(self.baseline_depth, sigma=1.0)
        
        self.baseline_established = True
        logger.info(f"Baseline established with grid size: {self.baseline_surface.shape}")
    
    def detect_surface_anomalies(self, water_points: List[WaterSurfacePoint]) -> List[WaterAnomaly]:
        """Detect anomalies in water surface (boats, debris, etc.)"""
        if not self.baseline_established:
            logger.warning("Baseline not established - cannot detect anomalies")
            return []
        
        logger.info("Detecting surface anomalies...")
        anomalies = []
        
        # Create current surface grid
        coords = np.array([[p.x, p.y] for p in water_points])
        surface_elevations = np.array([p.surface_elevation for p in water_points])
        
        xi, yi = np.meshgrid(self.x_grid, self.y_grid)
        
        from scipy.interpolate import griddata
        current_surface = griddata(coords, surface_elevations, (xi, yi), method='linear')
        current_surface = np.nan_to_num(current_surface, nan=np.nanmean(current_surface))
        current_surface = gaussian_filter(current_surface, sigma=1.0)
        
        # Calculate surface difference
        surface_diff = current_surface - self.baseline_surface
        
        # Detect significant elevation changes (potential vessels/objects)
        elevation_threshold = np.std(surface_diff) * 2.5
        elevated_areas = surface_diff > elevation_threshold
        
        # Use morphological operations to clean up detections
        if np.any(elevated_areas):
            elevated_areas = self._clean_binary_image(elevated_areas)
            
            # Find connected components (potential objects)
            from scipy.ndimage import label
            labeled_objects, num_objects = label(elevated_areas)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = labeled_objects == obj_id
                obj_area = np.sum(obj_mask) * (self.grid_resolution ** 2)
                
                # Filter by minimum size (e.g., must be at least 2 m²)
                if obj_area >= 2.0:
                    # Calculate object properties
                    obj_coords = np.where(obj_mask)
                    center_y = self.y_grid[int(np.mean(obj_coords[0]))]
                    center_x = self.x_grid[int(np.mean(obj_coords[1]))]
                    
                    elevation_change = np.mean(surface_diff[obj_mask])
                    max_elevation = np.max(surface_diff[obj_mask])
                    
                    # Classify anomaly type based on characteristics
                    anomaly_type = self._classify_surface_anomaly(obj_area, elevation_change, max_elevation)
                    
                    # Calculate confidence
                    confidence = min(1.0, (elevation_change / elevation_threshold) * 0.7 + 
                                   (obj_area / 100.0) * 0.3)
                    
                    # Find affected water points
                    affected_points = []
                    for point in water_points:
                        grid_x = int((point.x - self.x_grid[0]) / self.grid_resolution)
                        grid_y = int((point.y - self.y_grid[0]) / self.grid_resolution)
                        
                        if (0 <= grid_x < obj_mask.shape[1] and 
                            0 <= grid_y < obj_mask.shape[0] and 
                            obj_mask[grid_y, grid_x]):
                            affected_points.append(point)
                    
                    anomaly = WaterAnomaly(
                        anomaly_type=anomaly_type,
                        center_x=center_x,
                        center_y=center_y,
                        area=obj_area,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        properties={
                            'elevation_change': elevation_change,
                            'max_elevation': max_elevation,
                            'surface_roughness': np.std(surface_diff[obj_mask])
                        },
                        affected_points=affected_points
                    )
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} surface anomalies")
        return anomalies
    
    def detect_subsurface_anomalies(self, water_points: List[WaterSurfacePoint]) -> List[WaterAnomaly]:
        """Detect subsurface anomalies (chemical plumes, underwater objects)"""
        if not self.baseline_established:
            logger.warning("Baseline not established - cannot detect subsurface anomalies")
            return []
        
        logger.info("Detecting subsurface anomalies...")
        anomalies = []
        
        # Create current depth grid
        coords = np.array([[p.x, p.y] for p in water_points])
        water_depths = np.array([p.water_depth for p in water_points])
        intensities = np.array([p.confidence for p in water_points])  # Use confidence as proxy for return intensity
        
        xi, yi = np.meshgrid(self.x_grid, self.y_grid)
        
        from scipy.interpolate import griddata
        current_depth = griddata(coords, water_depths, (xi, yi), method='linear')
        current_intensity = griddata(coords, intensities, (xi, yi), method='linear')
        
        current_depth = np.nan_to_num(current_depth, nan=np.nanmean(current_depth))
        current_intensity = np.nan_to_num(current_intensity, nan=np.nanmean(current_intensity))
        
        # Detect depth anomalies
        depth_diff = current_depth - self.baseline_depth
        depth_threshold = np.std(depth_diff) * 2.0
        
        # Look for areas with unusual depth or intensity patterns
        shallow_anomalies = depth_diff < -depth_threshold  # Shallower than expected
        intensity_anomalies = current_intensity < (np.mean(current_intensity) - 2 * np.std(current_intensity))
        
        # Combine anomaly indicators
        subsurface_anomalies = shallow_anomalies | intensity_anomalies
        
        if np.any(subsurface_anomalies):
            subsurface_anomalies = self._clean_binary_image(subsurface_anomalies)
            
            from scipy.ndimage import label
            labeled_objects, num_objects = label(subsurface_anomalies)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = labeled_objects == obj_id
                obj_area = np.sum(obj_mask) * (self.grid_resolution ** 2)
                
                if obj_area >= 5.0:  # Minimum 5 m² for subsurface anomalies
                    obj_coords = np.where(obj_mask)
                    center_y = self.y_grid[int(np.mean(obj_coords[0]))]
                    center_x = self.x_grid[int(np.mean(obj_coords[1]))]
                    
                    depth_change = np.mean(depth_diff[obj_mask])
                    intensity_change = np.mean(current_intensity[obj_mask])
                    
                    # Classify subsurface anomaly
                    anomaly_type = self._classify_subsurface_anomaly(obj_area, depth_change, intensity_change)
                    
                    confidence = min(1.0, abs(depth_change / depth_threshold) * 0.6 + 
                                   obj_area / 50.0 * 0.4)
                    
                    # Find affected points
                    affected_points = []
                    for point in water_points:
                        grid_x = int((point.x - self.x_grid[0]) / self.grid_resolution)
                        grid_y = int((point.y - self.y_grid[0]) / self.grid_resolution)
                        
                        if (0 <= grid_x < obj_mask.shape[1] and 
                            0 <= grid_y < obj_mask.shape[0] and 
                            obj_mask[grid_y, grid_x]):
                            affected_points.append(point)
                    
                    anomaly = WaterAnomaly(
                        anomaly_type=anomaly_type,
                        center_x=center_x,
                        center_y=center_y,
                        area=obj_area,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        properties={
                            'depth_change': depth_change,
                            'intensity_change': intensity_change,
                            'depth_variance': np.var(current_depth[obj_mask])
                        },
                        affected_points=affected_points
                    )
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} subsurface anomalies")
        return anomalies
    
    def detect_chemical_spills(self, water_points: List[WaterSurfacePoint]) -> List[WaterAnomaly]:
        """Detect potential chemical spills using surface tension and reflection changes"""
        logger.info("Detecting potential chemical spills...")
        anomalies = []
        
        if len(water_points) < 50:
            logger.warning("Insufficient points for chemical spill detection")
            return anomalies
        
        # Extract spatial and intensity data
        coords = np.array([[p.x, p.y] for p in water_points])
        intensities = np.array([p.confidence for p in water_points])
        surface_elevations = np.array([p.surface_elevation for p in water_points])
        
        # Look for unusual surface texture patterns that might indicate chemical films
        # Chemical spills often create distinctive surface patterns
        
        # Create intensity grid
        xi, yi = np.meshgrid(self.x_grid, self.y_grid)
        
        from scipy.interpolate import griddata
        intensity_grid = griddata(coords, intensities, (xi, yi), method='linear')
        intensity_grid = np.nan_to_num(intensity_grid, nan=np.nanmean(intensities))
        
        # Apply edge detection to find unusual patterns
        edges_x = sobel(intensity_grid, axis=1)
        edges_y = sobel(intensity_grid, axis=0)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        
        # Look for areas with very low surface roughness (oil creates smooth surfaces)
        smoothness = gaussian_filter(edge_magnitude, sigma=2.0)
        smooth_threshold = np.mean(smoothness) - 1.5 * np.std(smoothness)
        smooth_areas = smoothness < smooth_threshold
        
        # Also look for rainbow-like intensity patterns (oil interference)
        intensity_variance = gaussian_filter(np.var(np.stack([
            np.roll(intensity_grid, 1, axis=0),
            intensity_grid,
            np.roll(intensity_grid, -1, axis=0)
        ]), axis=0), sigma=1.0)
        
        high_variance_areas = intensity_variance > (np.mean(intensity_variance) + 2 * np.std(intensity_variance))
        
        # Combine indicators for potential chemical spills
        chemical_indicators = smooth_areas & high_variance_areas
        
        if np.any(chemical_indicators):
            chemical_indicators = self._clean_binary_image(chemical_indicators)
            
            from scipy.ndimage import label
            labeled_spills, num_spills = label(chemical_indicators)
            
            for spill_id in range(1, num_spills + 1):
                spill_mask = labeled_spills == spill_id
                spill_area = np.sum(spill_mask) * (self.grid_resolution ** 2)
                
                if spill_area >= 10.0:  # Minimum 10 m² for chemical spill
                    spill_coords = np.where(spill_mask)
                    center_y = self.y_grid[int(np.mean(spill_coords[0]))]
                    center_x = self.x_grid[int(np.mean(spill_coords[1]))]
                    
                    # Calculate spill characteristics
                    smoothness_value = np.mean(smoothness[spill_mask])
                    variance_value = np.mean(intensity_variance[spill_mask])
                    
                    # Higher confidence for larger, smoother areas with high variance
                    confidence = min(1.0, 
                                   (spill_area / 100.0) * 0.4 +
                                   (1.0 - smoothness_value / np.max(smoothness)) * 0.3 +
                                   (variance_value / np.max(intensity_variance)) * 0.3)
                    
                    # Find affected points
                    affected_points = []
                    for point in water_points:
                        grid_x = int((point.x - self.x_grid[0]) / self.grid_resolution)
                        grid_y = int((point.y - self.y_grid[0]) / self.grid_resolution)
                        
                        if (0 <= grid_x < spill_mask.shape[1] and 
                            0 <= grid_y < spill_mask.shape[0] and 
                            spill_mask[grid_y, grid_x]):
                            affected_points.append(point)
                    
                    anomaly = WaterAnomaly(
                        anomaly_type='chemical_spill',
                        center_x=center_x,
                        center_y=center_y,
                        area=spill_area,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        properties={
                            'surface_smoothness': smoothness_value,
                            'intensity_variance': variance_value,
                            'estimated_thickness': self._estimate_spill_thickness(affected_points)
                        },
                        affected_points=affected_points
                    )
                    
                    anomalies.append(anomaly)
        
        logger.info(f"Detected {len(anomalies)} potential chemical spills")
        return anomalies
    
    def _clean_binary_image(self, binary_image: np.ndarray) -> np.ndarray:
        """Clean binary image using morphological operations"""
        try:
            import cv2
            # Convert to uint8 for OpenCV
            img = (binary_image * 255).astype(np.uint8)
            
            # Remove small noise
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            # Fill small gaps
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            
            return img > 0
        except ImportError:
            # Fallback: use scipy morphological operations
            from scipy.ndimage import binary_opening, binary_closing
            
            # Remove small noise (opening)
            cleaned = binary_opening(binary_image, structure=np.ones((3, 3)))
            
            # Fill small gaps (closing)
            cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
            
            return cleaned
    
    def _classify_surface_anomaly(self, area: float, elevation_change: float, max_elevation: float) -> str:
        """Classify surface anomaly based on characteristics"""
        if area > 50 and elevation_change > 1.0:
            return 'vessel'
        elif area > 20 and elevation_change > 0.5:
            return 'debris'
        elif area > 100 and elevation_change < 0.3:
            return 'surface_disturbance'
        else:
            return 'unknown'
    
    def _classify_subsurface_anomaly(self, area: float, depth_change: float, intensity_change: float) -> str:
        """Classify subsurface anomaly based on characteristics"""
        if depth_change < -0.5 and area > 20:
            return 'underwater_object'
        elif intensity_change < 0.3 and area > 30:
            return 'water_column_disturbance'
        elif area > 50:
            return 'bottom_anomaly'
        else:
            return 'unknown'
    
    def _estimate_spill_thickness(self, points: List[WaterSurfacePoint]) -> float:
        """Estimate chemical spill thickness based on surface characteristics"""
        if not points:
            return 0.0
        
        # Simplified estimation based on surface elevation variance
        elevations = [p.surface_elevation for p in points]
        thickness = np.std(elevations) * 1000  # Convert to millimeters
        return max(0.1, min(10.0, thickness))  # Clamp between 0.1-10mm

class AlertSystem:
    """Real-time alert system for water anomalies"""
    
    def __init__(self, alert_thresholds: Dict = None):
        self.alert_thresholds = alert_thresholds or {
            'vessel_min_size': 10.0,  # m²
            'chemical_spill_min_size': 5.0,  # m²
            'min_confidence': 0.6,
            'critical_confidence': 0.8
        }
        self.active_alerts = []
        
    def process_anomalies(self, anomalies: List[WaterAnomaly]) -> List[Dict]:
        """Process anomalies and generate alerts"""
        alerts = []
        
        for anomaly in anomalies:
            if anomaly.confidence >= self.alert_thresholds['min_confidence']:
                alert_level = self._determine_alert_level(anomaly)
                
                alert = {
                    'timestamp': anomaly.timestamp,
                    'alert_level': alert_level,
                    'anomaly_type': anomaly.anomaly_type,
                    'location': (anomaly.center_x, anomaly.center_y),
                    'area': anomaly.area,
                    'confidence': anomaly.confidence,
                    'description': self._generate_alert_description(anomaly),
                    'recommended_action': self._get_recommended_action(anomaly)
                }
                
                alerts.append(alert)
        
        self.active_alerts = alerts
        return alerts
    
    def _determine_alert_level(self, anomaly: WaterAnomaly) -> str:
        """Determine alert level based on anomaly characteristics"""
        if anomaly.confidence >= self.alert_thresholds['critical_confidence']:
            return 'CRITICAL'
        elif anomaly.anomaly_type == 'vessel' and anomaly.area > 50:
            return 'HIGH'
        elif anomaly.anomaly_type == 'chemical_spill':
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _generate_alert_description(self, anomaly: WaterAnomaly) -> str:
        """Generate human-readable alert description"""
        descriptions = {
            'vessel': f"Vessel detected: {anomaly.area:.1f} m² object at ({anomaly.center_x:.1f}, {anomaly.center_y:.1f})",
            'chemical_spill': f"Potential chemical spill: {anomaly.area:.1f} m² area at ({anomaly.center_x:.1f}, {anomaly.center_y:.1f})",
            'debris': f"Debris detected: {anomaly.area:.1f} m² object at ({anomaly.center_x:.1f}, {anomaly.center_y:.1f})",
            'underwater_object': f"Underwater object: {anomaly.area:.1f} m² anomaly at ({anomaly.center_x:.1f}, {anomaly.center_y:.1f})",
            'water_column_disturbance': f"Water column disturbance: {anomaly.area:.1f} m² area at ({anomaly.center_x:.1f}, {anomaly.center_y:.1f})"
        }
        
        return descriptions.get(anomaly.anomaly_type, f"Unknown anomaly at ({anomaly.center_x:.1f}, {anomaly.center_y:.1f})")
    
    def _get_recommended_action(self, anomaly: WaterAnomaly) -> str:
        """Get recommended action for anomaly type"""
        actions = {
            'vessel': "Monitor vessel movement, verify authorized presence",
            'chemical_spill': "IMMEDIATE RESPONSE: Deploy containment measures, notify environmental authorities",
            'debris': "Schedule debris removal, check navigation safety",
            'underwater_object': "Investigate with ROV, update navigation charts if needed",
            'water_column_disturbance': "Monitor water quality, check for pollution sources"
        }
        
        return actions.get(anomaly.anomaly_type, "Further investigation required")

class WaterChangeVisualizer:
    """Specialized visualization for water changes and anomalies"""
    
    @staticmethod
    def plot_anomaly_detection(water_points: List[WaterSurfacePoint], 
                              anomalies: List[WaterAnomaly],
                              baseline_detector: WaterChangeDetector,
                              output_file: str = None):
        """Create comprehensive anomaly detection visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        x = [p.x for p in water_points]
        y = [p.y for p in water_points]
        surface_z = [p.surface_elevation for p in water_points]
        depth = [p.water_depth for p in water_points]
        
        # Plot 1: Water surface with anomalies
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(x, y, c=surface_z, cmap='viridis', alpha=0.6, s=20)
        
        # Overlay anomalies
        for anomaly in anomalies:
            if anomaly.anomaly_type in ['vessel', 'debris', 'surface_disturbance']:
                circle = plt.Circle((anomaly.center_x, anomaly.center_y), 
                                  np.sqrt(anomaly.area/np.pi), 
                                  color='red', fill=False, linewidth=2)
                ax1.add_patch(circle)
                ax1.annotate(anomaly.anomaly_type, 
                           (anomaly.center_x, anomaly.center_y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red', weight='bold')
        
        ax1.set_title('Surface Anomalies Detection')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        plt.colorbar(scatter1, ax=ax1, label='Surface Elevation (m)')
        
        # Plot 2: Water depth with subsurface anomalies
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(x, y, c=depth, cmap='plasma', alpha=0.6, s=20)
        
        for anomaly in anomalies:
            if anomaly.anomaly_type in ['underwater_object', 'water_column_disturbance', 'bottom_anomaly']:
                circle = plt.Circle((anomaly.center_x, anomaly.center_y), 
                                  np.sqrt(anomaly.area/np.pi), 
                                  color='blue', fill=False, linewidth=2)
                ax2.add_patch(circle)
                ax2.annotate(anomaly.anomaly_type, 
                           (anomaly.center_x, anomaly.center_y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='blue', weight='bold')
        
        ax2.set_title('Subsurface Anomalies Detection')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        plt.colorbar(scatter2, ax=ax2, label='Water Depth (m)')
        
        # Plot 3: Chemical spill detection
        ax3 = axes[0, 2]
        confidence = [p.confidence for p in water_points]
        scatter3 = ax3.scatter(x, y, c=confidence, cmap='coolwarm', alpha=0.6, s=20)
        
        for anomaly in anomalies:
            if anomaly.anomaly_type == 'chemical_spill':
                circle = plt.Circle((anomaly.center_x, anomaly.center_y), 
                                  np.sqrt(anomaly.area/np.pi), 
                                  color='orange', fill=False, linewidth=3)
                ax3.add_patch(circle)
                ax3.annotate(f'SPILL\n{anomaly.area:.0f}m²', 
                           (anomaly.center_x, anomaly.center_y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, color='orange', weight='bold')
        
        ax3.set_title('Chemical Spill Detection')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        plt.colorbar(scatter3, ax=ax3, label='Confidence/Intensity')
        
        # Plot 4: Baseline vs Current (if available)
        ax4 = axes[1, 0]
        if baseline_detector.baseline_established:
            im4 = ax4.imshow(baseline_detector.baseline_surface, 
                           extent=[baseline_detector.x_grid[0], baseline_detector.x_grid[-1],
                                  baseline_detector.y_grid[0], baseline_detector.y_grid[-1]],
                           origin='lower', cmap='viridis', alpha=0.8)
            ax4.set_title('Baseline Water Surface')
            plt.colorbar(im4, ax=ax4, label='Elevation (m)')
        else:
            ax4.text(0.5, 0.5, 'No Baseline\nEstablished', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
            ax4.set_title('Baseline Water Surface')
        
        # Plot 5: Change magnitude
        ax5 = axes[1, 1]
        if len(anomalies) > 0:
            # Create change magnitude map
            change_map = np.zeros_like(baseline_detector.baseline_surface)
            for anomaly in anomalies:
                # Add anomaly influence to change map
                grid_x = int((anomaly.center_x - baseline_detector.x_grid[0]) / baseline_detector.grid_resolution)
                grid_y = int((anomaly.center_y - baseline_detector.y_grid[0]) / baseline_detector.grid_resolution)
                
                if (0 <= grid_x < change_map.shape[1] and 0 <= grid_y < change_map.shape[0]):
                    radius = int(np.sqrt(anomaly.area/np.pi) / baseline_detector.grid_resolution)
                    y_indices, x_indices = np.ogrid[:change_map.shape[0], :change_map.shape[1]]
                    mask = (x_indices - grid_x)**2 + (y_indices - grid_y)**2 <= radius**2
                    change_map[mask] = anomaly.confidence
            
            im5 = ax5.imshow(change_map, 
                           extent=[baseline_detector.x_grid[0], baseline_detector.x_grid[-1],
                                  baseline_detector.y_grid[0], baseline_detector.y_grid[-1]],
                           origin='lower', cmap='Reds', alpha=0.8)
            plt.colorbar(im5, ax=ax5, label='Change Magnitude')
        else:
            ax5.text(0.5, 0.5, 'No Changes\nDetected', 
                    transform=ax5.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
        
        ax5.set_title('Water Change Magnitude')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        
        # Plot 6: Alert summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create alert summary text
        alert_text = "DETECTION SUMMARY\n" + "="*20 + "\n\n"
        
        if anomalies:
            type_counts = {}
            for anomaly in anomalies:
                type_counts[anomaly.anomaly_type] = type_counts.get(anomaly.anomaly_type, 0) + 1
            
            for anomaly_type, count in type_counts.items():
                alert_text += f"{anomaly_type.replace('_', ' ').title()}: {count}\n"
            
            alert_text += f"\nTotal Anomalies: {len(anomalies)}\n"
            alert_text += f"High Confidence: {sum(1 for a in anomalies if a.confidence > 0.8)}\n"
            
            # Add specific alerts
            alert_text += "\nCRITICAL ALERTS:\n" + "-"*15 + "\n"
            critical_alerts = [a for a in anomalies if a.confidence > 0.8]
            
            if critical_alerts:
                for alert in critical_alerts[:3]:  # Show top 3
                    alert_text += f"• {alert.anomaly_type.replace('_', ' ').title()}\n"
                    alert_text += f"  Location: ({alert.center_x:.1f}, {alert.center_y:.1f})\n"
                    alert_text += f"  Area: {alert.area:.1f} m²\n"
                    alert_text += f"  Confidence: {alert.confidence:.2f}\n\n"
            else:
                alert_text += "No critical alerts\n"
        else:
            alert_text += "No anomalies detected\n\n"
            alert_text += "Water conditions appear normal"
        
        ax6.text(0.05, 0.95, alert_text, transform=ax6.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly detection visualization saved to {output_file}")
        
        plt.show()

def demo_change_detection():
    """Demonstrate the water change detection system with synthetic data"""
    logger.info("Running water change detection demo...")
    
    # Create synthetic baseline water data
    np.random.seed(42)
    n_points = 2000
    
    # Normal water surface
    x_base = np.random.uniform(0, 200, n_points)
    y_base = np.random.uniform(0, 200, n_points)
    z_surface_base = 10 + 0.2 * np.sin(x_base/20) + 0.1 * np.cos(y_base/30)
    z_surface_base += np.random.normal(0, 0.05, n_points)
    
    # Water depth
    z_bottom = 5 + 1.5 * np.sin(x_base/40) + 0.8 * np.cos(y_base/35)
    water_depth_base = z_surface_base - z_bottom
    
    # Create baseline water points
    baseline_points = []
    for i in range(n_points):
        point = WaterSurfacePoint(
            x=x_base[i], y=y_base[i],
            surface_elevation=z_surface_base[i],
            bottom_depth=z_bottom[i],
            water_depth=water_depth_base[i],
            timestamp=datetime.now(),
            confidence=np.random.uniform(0.7, 1.0)
        )
        baseline_points.append(point)
    
    # Initialize detector and establish baseline
    detector = WaterChangeDetector(grid_resolution=1.0)
    detector.establish_baseline(baseline_points)
    
    # Create current water data with anomalies
    x_current = np.random.uniform(0, 200, n_points)
    y_current = np.random.uniform(0, 200, n_points)
    z_surface_current = 10 + 0.2 * np.sin(x_current/20) + 0.1 * np.cos(y_current/30)
    z_surface_current += np.random.normal(0, 0.05, n_points)
    
    # Add synthetic anomalies
    # 1. Boat (elevated surface area)
    boat_mask = ((x_current - 50)**2 + (y_current - 75)**2) < 15**2
    z_surface_current[boat_mask] += 2.0
    
    # 2. Chemical spill (smooth surface with different reflection)
    spill_mask = ((x_current - 120)**2 + (y_current - 150)**2) < 25**2
    
    # 3. Underwater object (shallow area)
    object_mask = ((x_current - 170)**2 + (y_current - 50)**2) < 10**2
    
    # Create current water points
    current_points = []
    for i in range(n_points):
        confidence = np.random.uniform(0.7, 1.0)
        
        # Modify confidence for chemical spill (creates smooth, iridescent surface)
        if spill_mask[i]:
            confidence = np.random.uniform(0.3, 0.6)  # Lower reflection from oil
        
        # Modify depth for underwater object
        bottom_depth = z_bottom[i] if not object_mask[i] else z_bottom[i] + 1.5
        
        point = WaterSurfacePoint(
            x=x_current[i], y=y_current[i],
            surface_elevation=z_surface_current[i],
            bottom_depth=bottom_depth,
            water_depth=z_surface_current[i] - bottom_depth,
            timestamp=datetime.now(),
            confidence=confidence
        )
        current_points.append(point)
    
    # Detect anomalies
    surface_anomalies = detector.detect_surface_anomalies(current_points)
    subsurface_anomalies = detector.detect_subsurface_anomalies(current_points)
    chemical_spills = detector.detect_chemical_spills(current_points)
    
    all_anomalies = surface_anomalies + subsurface_anomalies + chemical_spills
    
    # Generate alerts
    alert_system = AlertSystem()
    alerts = alert_system.process_anomalies(all_anomalies)
    
    # Create visualization
    output_file = "/Users/tycrouch/Desktop/untitled folder 4/water_change_detection_demo.png"
    WaterChangeVisualizer.plot_anomaly_detection(
        current_points, all_anomalies, detector, output_file
    )
    
    # Print alert summary
    print("\n" + "="*60)
    print("WATER CHANGE DETECTION RESULTS")
    print("="*60)
    print(f"Total anomalies detected: {len(all_anomalies)}")
    print(f"Surface anomalies: {len(surface_anomalies)}")
    print(f"Subsurface anomalies: {len(subsurface_anomalies)}")
    print(f"Chemical spills: {len(chemical_spills)}")
    print(f"Alerts generated: {len(alerts)}")
    
    print(f"\nALERT DETAILS:")
    for alert in alerts:
        print(f"  {alert['alert_level']}: {alert['description']}")
        print(f"    Action: {alert['recommended_action']}")
    
    return all_anomalies, alerts

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    anomalies, alerts = demo_change_detection()
