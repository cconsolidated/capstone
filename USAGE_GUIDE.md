# Water Monitoring System - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo (Recommended First Step)
```bash
python3 water_monitoring_system.py --mode demo
```

This will:
- Create synthetic water data with boats and underwater objects
- Demonstrate anomaly detection algorithms
- Generate visualization showing detected anomalies
- Display alert summaries

## Detection Capabilities

### 🚢 Vessels/Boats
- **Detection Method**: Surface elevation changes above baseline
- **Minimum Size**: 10 m² (configurable)
- **Indicators**: Elevated areas, distinct boundaries
- **Alert Level**: CRITICAL for large vessels (>50 m²)

### 🛢️ Chemical Spills
- **Detection Method**: Surface texture and reflection analysis
- **Minimum Size**: 5 m² (configurable)
- **Indicators**: Smooth surfaces with high intensity variance (oil creates rainbow patterns)
- **Alert Level**: HIGH (immediate response required)

### 🪨 Underwater Objects/Debris
- **Detection Method**: Depth anomalies and return intensity changes
- **Minimum Size**: 5 m² (configurable)
- **Indicators**: Shallower depths than expected, unusual bottom signatures
- **Alert Level**: MEDIUM to HIGH depending on size

### 💧 Water Column Disturbances
- **Detection Method**: Intensity pattern analysis
- **Indicators**: Unusual acoustic returns, turbidity changes
- **Applications**: Pollution monitoring, sediment plumes

## Operation Modes

### Demo Mode
```bash
python3 water_monitoring_system.py --mode demo
```
- Uses synthetic data to demonstrate capabilities
- Perfect for testing and training
- Shows all detection types

### Real-Time Monitoring
```bash
python3 water_monitoring_system.py --mode monitor --data-dir "Data for OARS - Copy"
```
- Continuous monitoring of water conditions
- Establishes baseline automatically
- Generates alerts and reports
- Creates timestamped visualizations

### Historical Analysis
```bash
python3 water_monitoring_system.py --mode analyze --data-dir "Data for OARS - Copy"
```
- Analyzes all available historical data
- Identifies trends and patterns
- Generates comprehensive reports

### Baseline Establishment
```bash
python3 water_monitoring_system.py --mode baseline --data-dir "Data for OARS - Copy"
```
- Establishes clean water baseline conditions
- Required before real-time monitoring
- Uses earliest available data

## Configuration

Edit `config.yaml` to customize detection parameters:

```yaml
fusion:
  spatial_tolerance: 2.0      # Max distance for data alignment (meters)
  temporal_tolerance: 30.0    # Max time difference (seconds)
  grid_resolution: 1.0        # Processing grid resolution (meters)

alert_thresholds:
  vessel_min_size: 10.0       # Minimum vessel size to alert (m²)
  chemical_spill_min_size: 5.0 # Minimum spill size (m²)
  min_confidence: 0.6         # Minimum confidence for alerts
  critical_confidence: 0.8    # Threshold for critical alerts
```

## Understanding the Output

### Visualization Plots

1. **Surface Anomalies**: Shows LIDAR-detected objects above water
   - Red circles indicate vessels/debris
   - Color represents surface elevation

2. **Subsurface Anomalies**: Shows sonar-detected underwater objects
   - Blue circles indicate underwater anomalies
   - Color represents water depth

3. **Chemical Spill Detection**: Shows potential contamination
   - Orange circles indicate suspicious areas
   - Based on surface texture analysis

4. **Baseline vs Current**: Comparison with normal conditions
5. **Change Magnitude**: Areas of significant change
6. **Alert Summary**: Text summary of all detections

### Alert Levels

- **CRITICAL**: Immediate attention required
  - Large vessels
  - High-confidence detections
  - Chemical spills

- **HIGH**: Important but not emergency
  - Medium-sized objects
  - Pollution indicators

- **MEDIUM**: Monitor and investigate
  - Small anomalies
  - Low-confidence detections

## Real-World Usage Scenarios

### Harbor Security
```bash
# Monitor for unauthorized vessels
python3 water_monitoring_system.py --mode monitor --interval 10
```
- Detects boats entering restricted areas
- Identifies suspicious underwater activity
- Monitors for potential threats

### Environmental Protection
```bash
# Monitor for pollution and spills
python3 water_monitoring_system.py --mode monitor --interval 60
```
- Early detection of chemical spills
- Monitors industrial discharge
- Tracks environmental changes

### Navigation Safety
```bash
# Monitor for navigation hazards
python3 water_monitoring_system.py --mode monitor
```
- Detects debris and obstacles
- Identifies shallow areas
- Updates hazard maps

### Scientific Research
```bash
# Analyze ecosystem changes
python3 water_monitoring_system.py --mode analyze
```
- Long-term environmental monitoring
- Habitat change detection
- Research data analysis

## Data Requirements

### LIDAR Data (HSX/RAW format)
- **Source**: HYPACK with Velodyne VLP-16
- **Contains**: 3D point clouds above water surface
- **Key Info**: Surface elevation, reflection intensity
- **Frequency**: High-resolution spatial data

### Sonar Data (S7K format)
- **Source**: NORBIT WBMS multibeam
- **Contains**: Bathymetric data below water surface
- **Key Info**: Bottom depth, beam quality
- **Frequency**: Acoustic returns from water column and bottom

### Coordinate Alignment
- Both datasets must share common coordinate system
- GPS/INS synchronization essential
- Sensor mounting offsets configured in HSX headers

## Troubleshooting

### No Anomalies Detected
1. Check if baseline is established
2. Verify data quality and coverage
3. Adjust detection thresholds in config
4. Ensure spatial/temporal alignment

### False Positives
1. Increase confidence thresholds
2. Improve baseline with more data
3. Filter by minimum object size
4. Check environmental conditions

### Missing Data
1. Verify file paths and formats
2. Check HSX/S7K file integrity
3. Ensure device synchronization
4. Review coordinate systems

### Performance Issues
1. Reduce grid resolution for faster processing
2. Limit spatial coverage area
3. Increase monitoring intervals
4. Process data in batches

## Advanced Features

### Custom Detection Algorithms
- Modify `water_change_detection.py` for specialized detection
- Add new anomaly types in classification functions
- Implement custom alert logic

### Integration Options
- REST API endpoints for external systems
- Database logging for historical analysis
- Email/SMS notification integration
- GIS system compatibility

### Machine Learning Enhancement
- Train models on your specific water conditions
- Implement adaptive thresholds
- Add predictive anomaly detection
- Use deep learning for pattern recognition

## Best Practices

### Data Collection
1. Maintain consistent survey patterns
2. Ensure proper sensor calibration
3. Record environmental conditions
4. Document known objects/changes

### Baseline Management
1. Update baseline periodically
2. Use clean water conditions
3. Account for seasonal variations
4. Document baseline parameters

### Alert Management
1. Establish response procedures
2. Train operators on system outputs
3. Maintain contact lists for notifications
4. Document all incidents and responses

### System Maintenance
1. Regular software updates
2. Sensor calibration checks
3. Data backup procedures
4. Performance monitoring

## Support and Development

### Getting Help
- Review log files (`water_monitoring.log`)
- Check configuration parameters
- Verify data formats and paths
- Test with demo mode first

### Contributing
- Report issues and bugs
- Suggest new detection algorithms
- Contribute to documentation
- Share real-world use cases

### Future Enhancements
- Real-time streaming data processing
- Mobile app for field operations
- Cloud-based processing
- AI-powered detection improvements
