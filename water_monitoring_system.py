#!/usr/bin/env python3
"""
Integrated Water Monitoring System
Real-time detection of boats, chemical spills, and other water anomalies
using combined LIDAR and Sonar data
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import signal
import threading
import time

# Import our modules
from lidar_sonar_fusion import HSXParser, S7KParser, DataFusion, WaterSurfacePoint
from water_change_detection import WaterChangeDetector, AlertSystem, WaterChangeVisualizer
from advanced_parsers import AdvancedHSXParser, AdvancedS7KParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WaterMonitoringSystem:
    """Main water monitoring system class"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config = self._load_config(config_file)
        self.running = False
        self.data_fusion = DataFusion(
            spatial_tolerance=self.config['fusion']['spatial_tolerance'],
            temporal_tolerance=self.config['fusion']['temporal_tolerance']
        )
        self.change_detector = WaterChangeDetector(
            grid_resolution=self.config['fusion']['grid_resolution']
        )
        self.alert_system = AlertSystem(self.config.get('alert_thresholds', {}))
        self.baseline_established = False
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_file} not found, using defaults")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data': {
                'input_directory': 'Data for OARS - Copy',
                'lidar_directory': 'HYPACK iLIDAR DATA',
                'sonar_directory': 'WBMS',
                'output_directory': 'processed_data'
            },
            'fusion': {
                'spatial_tolerance': 2.0,
                'temporal_tolerance': 30.0,
                'grid_resolution': 1.0,
                'min_confidence': 0.3
            },
            'water_detection': {
                'surface_smoothing': 0.5,
                'depth_change_threshold': 0.1,
                'outlier_std_dev': 2.0
            },
            'alert_thresholds': {
                'vessel_min_size': 10.0,
                'chemical_spill_min_size': 5.0,
                'min_confidence': 0.6,
                'critical_confidence': 0.8
            }
        }
    
    def establish_baseline(self, data_directory: str = None) -> bool:
        """Establish baseline water conditions from historical data"""
        logger.info("Establishing baseline water conditions...")
        
        if data_directory is None:
            data_directory = self.config['data']['input_directory']
        
        data_path = Path(data_directory)
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_directory}")
            return False
        
        # Find baseline data files (use first available dataset)
        baseline_points = self._process_data_files(data_path, baseline_mode=True)
        
        if not baseline_points:
            logger.error("No data found for baseline establishment")
            return False
        
        self.change_detector.establish_baseline(baseline_points)
        self.baseline_established = True
        logger.info(f"Baseline established with {len(baseline_points)} points")
        return True
    
    def _process_data_files(self, data_path: Path, baseline_mode: bool = False) -> List[WaterSurfacePoint]:
        """Process LIDAR and sonar data files"""
        water_points = []
        
        # Find LIDAR files
        lidar_dir = data_path / self.config['data']['lidar_directory']
        sonar_dir = data_path / self.config['data']['sonar_directory']
        
        if not lidar_dir.exists() or not sonar_dir.exists():
            logger.error("LIDAR or sonar directories not found")
            return water_points
        
        # Process each LIDAR test directory
        for test_dir in lidar_dir.iterdir():
            if test_dir.is_dir():
                hsx_files = list(test_dir.glob("*.HSX"))
                
                for hsx_file in hsx_files:
                    if baseline_mode:
                        logger.info(f"Processing baseline data: {hsx_file.name}")
                    else:
                        logger.info(f"Processing current data: {hsx_file.name}")
                    
                    # Parse LIDAR data
                    lidar_points = self._parse_lidar_file(hsx_file)
                    
                    # Find corresponding sonar data
                    sonar_pings = self._find_and_parse_sonar_data(sonar_dir, hsx_file)
                    
                    # Fuse the data
                    if lidar_points and sonar_pings:
                        fused_points = self.data_fusion.align_datasets(lidar_points, sonar_pings)
                        water_points.extend(fused_points)
                        
                        if baseline_mode:
                            # Only use first file for baseline
                            break
                
                if baseline_mode and water_points:
                    break
        
        return water_points
    
    def _parse_lidar_file(self, hsx_file: Path) -> List:
        """Parse LIDAR data from HSX file"""
        try:
            parser = AdvancedHSXParser(str(hsx_file))
            parser.parse_complete_header()
            
            # For now, return simplified parsing results
            # In production, would implement full binary parsing
            logger.info(f"LIDAR devices found: {len(parser.devices)}")
            
            # Return empty list for now - would contain actual LIDAR points
            return []
            
        except Exception as e:
            logger.error(f"Error parsing LIDAR file {hsx_file}: {e}")
            return []
    
    def _find_and_parse_sonar_data(self, sonar_dir: Path, hsx_file: Path) -> List:
        """Find and parse corresponding sonar data"""
        try:
            # Find sonar files from same time period
            for session_dir in sonar_dir.iterdir():
                if session_dir.is_dir():
                    s7k_files = list(session_dir.glob("*.s7k"))
                    
                    if s7k_files:
                        s7k_file = s7k_files[0]
                        parser = AdvancedS7KParser(str(s7k_file))
                        analysis = parser.analyze_s7k_structure()
                        
                        logger.info(f"Sonar records found: {analysis['total_records']}")
                        
                        # Return empty list for now - would contain actual sonar pings
                        return []
            
            return []
            
        except Exception as e:
            logger.error(f"Error parsing sonar data: {e}")
            return []
    
    def monitor_real_time(self, data_directory: str = None, interval: int = 30):
        """Start real-time monitoring for water changes"""
        logger.info("Starting real-time water monitoring...")
        
        if not self.baseline_established:
            logger.info("Baseline not established, establishing now...")
            if not self.establish_baseline(data_directory):
                logger.error("Failed to establish baseline - cannot start monitoring")
                return
        
        self.running = True
        
        try:
            while self.running:
                logger.info("Processing current water data...")
                
                # Process current data
                data_path = Path(data_directory or self.config['data']['input_directory'])
                current_points = self._process_data_files(data_path, baseline_mode=False)
                
                if current_points:
                    # Detect anomalies
                    anomalies = self._detect_all_anomalies(current_points)
                    
                    # Generate alerts
                    if anomalies:
                        alerts = self.alert_system.process_anomalies(anomalies)
                        self._handle_alerts(alerts, anomalies, current_points)
                    else:
                        logger.info("No anomalies detected - water conditions normal")
                
                # Wait for next monitoring cycle
                if self.running:
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
        finally:
            self.running = False
    
    def _detect_all_anomalies(self, water_points: List[WaterSurfacePoint]) -> List:
        """Detect all types of anomalies"""
        all_anomalies = []
        
        # Surface anomalies (boats, debris)
        surface_anomalies = self.change_detector.detect_surface_anomalies(water_points)
        all_anomalies.extend(surface_anomalies)
        
        # Subsurface anomalies (underwater objects)
        subsurface_anomalies = self.change_detector.detect_subsurface_anomalies(water_points)
        all_anomalies.extend(subsurface_anomalies)
        
        # Chemical spills
        chemical_spills = self.change_detector.detect_chemical_spills(water_points)
        all_anomalies.extend(chemical_spills)
        
        logger.info(f"Detected {len(all_anomalies)} total anomalies")
        return all_anomalies
    
    def _handle_alerts(self, alerts: List[Dict], anomalies: List, water_points: List[WaterSurfacePoint]):
        """Handle generated alerts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"{alert['alert_level']} ALERT: {alert['description']}")
            logger.info(f"Recommended action: {alert['recommended_action']}")
        
        # Create output directory
        output_dir = Path(self.config['data']['output_directory'])
        output_dir.mkdir(exist_ok=True)
        
        # Save detection visualization
        viz_file = output_dir / f"detection_{timestamp}.png"
        WaterChangeVisualizer.plot_anomaly_detection(
            water_points, anomalies, self.change_detector, str(viz_file)
        )
        
        # Save alert report
        self._save_alert_report(alerts, anomalies, output_dir / f"alert_report_{timestamp}.txt")
        
        # Send notifications for critical alerts
        critical_alerts = [a for a in alerts if a['alert_level'] == 'CRITICAL']
        if critical_alerts:
            self._send_critical_notifications(critical_alerts)
    
    def _save_alert_report(self, alerts: List[Dict], anomalies: List, report_file: Path):
        """Save detailed alert report"""
        with open(report_file, 'w') as f:
            f.write("WATER MONITORING ALERT REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Total Anomalies: {len(anomalies)}\n")
            f.write(f"Total Alerts: {len(alerts)}\n")
            
            alert_levels = {}
            for alert in alerts:
                level = alert['alert_level']
                alert_levels[level] = alert_levels.get(level, 0) + 1
            
            for level, count in alert_levels.items():
                f.write(f"{level} Alerts: {count}\n")
            
            f.write(f"\nDETAILED ALERTS:\n")
            f.write("-" * 30 + "\n")
            
            for i, alert in enumerate(alerts, 1):
                f.write(f"\nAlert {i}: {alert['alert_level']}\n")
                f.write(f"Type: {alert['anomaly_type']}\n")
                f.write(f"Location: ({alert['location'][0]:.1f}, {alert['location'][1]:.1f})\n")
                f.write(f"Area: {alert['area']:.1f} m²\n")
                f.write(f"Confidence: {alert['confidence']:.2f}\n")
                f.write(f"Description: {alert['description']}\n")
                f.write(f"Recommended Action: {alert['recommended_action']}\n")
        
        logger.info(f"Alert report saved to {report_file}")
    
    def _send_critical_notifications(self, critical_alerts: List[Dict]):
        """Send notifications for critical alerts"""
        # In a real system, this would send emails, SMS, or API calls
        logger.critical(f"CRITICAL ALERT: {len(critical_alerts)} critical water anomalies detected!")
        
        for alert in critical_alerts:
            logger.critical(f"  {alert['description']}")
            
        # Could integrate with:
        # - Email notifications
        # - SMS alerts
        # - Webhook notifications
        # - Emergency response systems
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        logger.info("Stopping water monitoring system...")
        self.running = False
    
    def analyze_historical_data(self, data_directory: str = None) -> Dict:
        """Analyze historical data for trends and patterns"""
        logger.info("Analyzing historical water data...")
        
        data_path = Path(data_directory or self.config['data']['input_directory'])
        
        # Process all available data
        all_points = self._process_data_files(data_path, baseline_mode=False)
        
        if not all_points:
            logger.warning("No historical data found")
            return {}
        
        # Establish baseline from first portion of data
        baseline_points = all_points[:len(all_points)//3]
        self.change_detector.establish_baseline(baseline_points)
        
        # Analyze remaining data for anomalies
        analysis_points = all_points[len(all_points)//3:]
        anomalies = self._detect_all_anomalies(analysis_points)
        
        # Generate analysis report
        analysis = {
            'total_points': len(all_points),
            'baseline_points': len(baseline_points),
            'analysis_points': len(analysis_points),
            'anomalies_detected': len(anomalies),
            'anomaly_types': {},
            'recommendations': []
        }
        
        # Count anomaly types
        for anomaly in anomalies:
            anomaly_type = anomaly.anomaly_type
            analysis['anomaly_types'][anomaly_type] = analysis['anomaly_types'].get(anomaly_type, 0) + 1
        
        # Generate recommendations
        if analysis['anomalies_detected'] > 0:
            analysis['recommendations'].append("Anomalies detected in historical data")
            analysis['recommendations'].append("Recommend establishing updated baseline")
            analysis['recommendations'].append("Consider environmental factors affecting detections")
        else:
            analysis['recommendations'].append("No significant anomalies in historical data")
            analysis['recommendations'].append("Water conditions appear stable")
        
        logger.info(f"Historical analysis complete: {analysis['anomalies_detected']} anomalies found")
        return analysis

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info("Received interrupt signal, shutting down...")
    sys.exit(0)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Water Monitoring System")
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--data-dir', help='Data directory path')
    parser.add_argument('--mode', choices=['monitor', 'analyze', 'baseline', 'demo'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create monitoring system
    monitoring_system = WaterMonitoringSystem(args.config)
    
    try:
        if args.mode == 'demo':
            # Run demonstration with synthetic data
            logger.info("Running demonstration mode...")
            from water_change_detection import demo_change_detection
            demo_change_detection()
            
        elif args.mode == 'baseline':
            # Establish baseline only
            logger.info("Establishing baseline mode...")
            success = monitoring_system.establish_baseline(args.data_dir)
            if success:
                logger.info("Baseline establishment completed successfully")
            else:
                logger.error("Baseline establishment failed")
                
        elif args.mode == 'analyze':
            # Analyze historical data
            logger.info("Historical analysis mode...")
            analysis = monitoring_system.analyze_historical_data(args.data_dir)
            
            print("\nHISTORICAL DATA ANALYSIS")
            print("=" * 40)
            print(f"Total data points: {analysis.get('total_points', 0)}")
            print(f"Anomalies detected: {analysis.get('anomalies_detected', 0)}")
            
            if analysis.get('anomaly_types'):
                print("\nAnomaly types found:")
                for anomaly_type, count in analysis['anomaly_types'].items():
                    print(f"  {anomaly_type}: {count}")
            
            print("\nRecommendations:")
            for rec in analysis.get('recommendations', []):
                print(f"  • {rec}")
                
        elif args.mode == 'monitor':
            # Start real-time monitoring
            logger.info("Real-time monitoring mode...")
            monitoring_system.monitor_real_time(args.data_dir, args.interval)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        monitoring_system.stop_monitoring()
        logger.info("Water monitoring system shutdown complete")

if __name__ == "__main__":
    main()
