# Water Monitoring System 🌊

**Real-time detection of boats, chemical spills, and water anomalies using combined LIDAR and Sonar data**

This advanced system combines HYPACK LIDAR (HSX format) and NORBIT WBMS sonar (S7K format) data to detect changes in water conditions from above and below the surface. Perfect for harbor security, environmental monitoring, and navigation safety.

## 🎯 Key Detection Capabilities

- **🚢 Vessels & Boats**: Detect unauthorized or suspicious vessels
- **🛢️ Chemical Spills**: Early warning for oil spills and contamination  
- **🪨 Underwater Objects**: Find debris, obstacles, and navigation hazards
- **💧 Water Column Disturbances**: Monitor pollution and environmental changes

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demonstration (recommended first step)
python3 water_monitoring_system.py --mode demo

# Start real-time monitoring
python3 water_monitoring_system.py --mode monitor --data-dir "Data for OARS - Copy"
```

## Overview

The system processes:
- **LIDAR data**: Velodyne VLP-16 point clouds from above water surface
- **Sonar data**: NORBIT WBMS multibeam bathymetry from below water surface
- **AI Detection**: Machine learning algorithms for anomaly identification
- **Real-time Alerts**: Immediate notifications for critical situations

## Features

### Engineering Actions (EA1-EA8) - Complete Implementation:

- ✅ **EA1**: HSX/RAW file parsing for HYPACK LIDAR data + S7K parsing for NORBIT WBMS sonar
- ✅ **EA2**: GNSS/RTK coordinate system alignment with configurable sensor offsets  
- ✅ **EA3**: Mean Low Water (MLW) datum normalization with tidal corrections
- ✅ **EA4**: Spatial and temporal point cloud merging
- ✅ **EA5**: Alignment error detection and correction using median filtering
- ✅ **EA6**: Accuracy validation with RMSE, MAE, and statistical metrics
- ✅ **EA7**: 3D visualization of merged sonar-LiDAR water surface
- ✅ **EA8**: Comprehensive process documentation and data export

**See [ENGINEERING_ACTIONS_IMPLEMENTATION.md](ENGINEERING_ACTIONS_IMPLEMENTATION.md) for complete technical details.**

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data structure matches:
```
Data for OARS - Copy/
├── HYPACK iLIDAR DATA/
│   ├── ILIDAR LAKE TEST/
│   │   ├── *.HSX (header files)
│   │   └── *.RAW (binary LIDAR data)
│   └── NORBIT TEST/
└── WBMS/
    └── */
        ├── *.s7k (sonar data)
        └── SessionInfo.ini
```

## Usage

### Basic Processing
```bash
python lidar_sonar_fusion.py
```

### Testing
```bash
python test_data_fusion.py
```

### Configuration
Edit `config.yaml` to adjust processing parameters:
- Spatial/temporal tolerance for data alignment
- Water detection thresholds
- Output formats and visualization settings

## Data Formats

### LIDAR Data (HSX/RAW)
- **HSX**: ASCII header with survey parameters, device configurations, and geodetic information
- **RAW**: Binary point cloud data from Velodyne VLP-16
- **Key fields**: X, Y, Z coordinates, intensity, timestamp

### Sonar Data (S7K)
- **S7K**: NORBIT WBMS proprietary format
- **Record types**: Navigation, bathymetry, backscatter
- **Key fields**: Beam depth, position, quality metrics

## Processing Pipeline

1. **Data Parsing**
   - Extract LIDAR points from HSX/RAW files
   - Extract sonar pings from S7K files
   - Parse timestamps and coordinates

2. **Data Alignment**
   - Spatial matching within tolerance (default: 2m)
   - Temporal matching within tolerance (default: 30s)
   - Quality filtering and confidence scoring

3. **Water Detection**
   - Surface elevation from LIDAR returns
   - Bottom depth from sonar bathymetry
   - Water depth calculation and validation

4. **Change Detection**
   - Grid-based interpolation
   - Gradient analysis for change detection
   - Statistical outlier removal

5. **Output & Visualization**
   - CSV export with aligned measurements
   - 3D visualization of water surface
   - Change detection maps

## Output Files

- `fused_water_data.csv`: Combined LIDAR-sonar measurements
- `water_surface_analysis.png`: Visualization plots
- `synthetic_test_data.png`: Algorithm validation plots

## Data Quality Factors

### Spatial Alignment
- GPS/INS positioning accuracy
- Device mounting offsets
- Survey line overlap

### Temporal Synchronization  
- Clock synchronization between systems
- Data acquisition timing
- Processing latency

### Environmental Conditions
- Water surface conditions (waves, turbulence)
- Weather effects on LIDAR
- Water clarity for bottom detection

## Limitations

### Current Implementation
- Simplified binary format parsers (need full format specs)
- Basic spatial interpolation methods
- Limited error handling for corrupted data

### Recommended Improvements
- Implement full HSX/RAW format specification
- Add advanced filtering (Kalman, particle filters)
- Include tide and weather corrections
- Real-time processing capabilities

## Technical Notes

### Coordinate Systems
- Assumes consistent coordinate reference system
- Check geodetic parameters in HSX headers
- May require coordinate transformations

### Performance
- Memory usage scales with point density
- Processing time depends on spatial/temporal tolerances
- Consider data decimation for large datasets

### Validation
- Compare with ground truth measurements
- Cross-validate with manual survey data
- Assess accuracy against known control points

## Support

For questions about:
- **Data formats**: Consult HYPACK and NORBIT documentation
- **Processing algorithms**: See comments in source code
- **Configuration**: Review `config.yaml` parameters

## Future Development

- [ ] Real-time data streaming
- [ ] Machine learning for water detection
- [ ] Integration with GIS systems
- [ ] Mobile app for field validation
- [ ] Cloud processing capabilities

