# Engineering Actions (EA1-EA8) - Implementation Guide

This document details how each Engineering Action is implemented in the LIDAR-Sonar Data Fusion System.

---

## EA1 – Collect Sonar and LiDAR Data

**Description:** Gather Sonar (.S7K) and LiDAR (.HSX/.RAW) data from the same area to ensure both cover the same ground and overlap correctly.

### Implementation:
- **Classes:** `HSXParser`, `S7KParser`
- **Methods:**
  - `HSXParser.parse_header()` - Parses HYPACK HSX header information
  - `HSXParser.parse_lidar_data()` - Extracts LiDAR point cloud from RAW files
  - `S7KParser.parse_sonar_data()` - Extracts sonar bathymetric data from S7K files

### Code Location:
```python
# Lines 54-183: HSXParser class
# Lines 184-253: S7KParser class
# Lines 674-690: main() function EA1 execution
```

### Data Formats:
- **Input:** HSX (ASCII header), RAW (binary LiDAR), S7K (binary sonar)
- **Output:** `List[LidarPoint]`, `List[SonarPing]`

### Requirements Satisfied:
- R1: Identifies SONAR and LiDAR systems (NORBIT WBMS, Velodyne VLP-16)
- R2: Documents how each system collects and stores data
- R3: Documents data formats (HSX/RAW for LiDAR, S7K for sonar)

---

## EA2 – Align Data to the Same Coordinate System

**Description:** Use GNSS/RTK data to make sure both Sonar and LiDAR points are tied to the same global position and elevation reference.

### Implementation:
- **Method:** `DataFusion.apply_coordinate_transformation()`
- **Parameters:**
  - `sensor_offset_x` - LiDAR X offset from vessel reference (meters)
  - `sensor_offset_y` - LiDAR Y offset from vessel reference (meters)
  - `sensor_offset_z` - LiDAR Z offset from vessel reference (meters)

### Code Location:
```python
# Lines 281-302: apply_coordinate_transformation() method
# Lines 654-661: main() function initialization with offsets
# Lines 696-699: EA2 execution in workflow
```

### Process:
1. Apply configurable sensor offsets to LiDAR points
2. Transform coordinates to common reference frame
3. Log transformation parameters for documentation

### Requirements Satisfied:
- R9: Configurable sensor offsets for different USV hull configurations
- Part of R5: Candidate method for aligning datasets

---

## EA3 – Normalize to a Common Vertical Datum

**Description:** Adjust both datasets to the same vertical level (Mean Low Water) so that depths and elevations line up correctly.

### Implementation:
- **Method:** `DataFusion.normalize_to_mlw_datum()`
- **Parameters:**
  - `mlw_datum` - Mean Low Water datum offset (meters)
  - `tidal_correction` - Real-time tidal correction (meters)

### Code Location:
```python
# Lines 304-336: normalize_to_mlw_datum() method
# Lines 701-705: EA3 execution with tidal correction
```

### Process:
1. Subtract MLW datum and tidal correction from LiDAR elevations
2. Add MLW datum and tidal correction to sonar depths
3. Ensure consistent vertical reference across datasets

### Requirements Satisfied:
- R6: Normalization procedure with MLW datum and tidal corrections
- R7: Ensures vertical alignment between LiDAR and MBES data

---

## EA4 – Merge Point Clouds

**Description:** Combine the Sonar (underwater) and LiDAR (above-water) point clouds into one complete 3D dataset.

### Implementation:
- **Method:** `DataFusion.align_datasets()`
- **Data Structure:** `WaterSurfacePoint` combines both datasets

### Code Location:
```python
# Lines 338-392: align_datasets() method (EA4)
# Lines 44-52: WaterSurfacePoint dataclass definition
# Lines 707-708: EA4 execution in workflow
```

### Process:
1. Build spatial KDTree index for sonar data
2. For each LiDAR point, find nearby sonar pings (within spatial tolerance)
3. Filter matches by temporal proximity (within time tolerance)
4. Calculate water depth, surface elevation, and confidence
5. Create unified `WaterSurfacePoint` objects

### Output:
```python
WaterSurfacePoint:
    x, y: Spatial coordinates
    surface_elevation: From LiDAR (above water)
    bottom_depth: From sonar (below water)
    water_depth: Calculated difference
    confidence: Match quality metric
    timestamp: Time of observation
```

### Requirements Satisfied:
- R4: Demonstrates datasets can be merged
- R5: Implements spatial-temporal alignment method
- R11: Creates pilot dataset demonstrating integration

---

## EA5 – Check and Fix Alignment Errors

**Description:** Compare overlapping areas to find gaps or mismatches, then correct them to reduce vertical and horizontal bias.

### Implementation:
- **Method:** `DataFusion.check_alignment_errors()`
- **Algorithm:** Median filtering of overlapping points

### Code Location:
```python
# Lines 394-453: check_alignment_errors() method
# Lines 714-717: EA5 execution in workflow
```

### Process:
1. Build spatial index for all merged points
2. For each point, find 10 nearest neighbors
3. Calculate median depth of neighbors
4. Detect errors exceeding tolerance threshold
5. Correct outliers using median value
6. Record all errors for analysis

### Error Metrics Tracked:
- Vertical error magnitude
- Original vs corrected depth
- Spatial location of errors

### Requirements Satisfied:
- Part of R5: Evaluates alignment accuracy
- R7: Verifies vertical alignment within tolerance

---

## EA6 – Validate Combined Data Accuracy

**Description:** Calculate how well the merged dataset matches real-world measurements (using RMSE or Plate Check tests).

### Implementation:
- **Method:** `DataFusion.validate_accuracy()`
- **Metrics:** RMSE, MAE, Bias, Statistical analysis

### Code Location:
```python
# Lines 455-513: validate_accuracy() method
# Lines 719-720: EA6 execution in workflow
# Lines 757-760: Display validation results
```

### Validation Metrics:

#### Internal Consistency:
- `num_points` - Total merged points
- `depth_mean` - Mean water depth
- `depth_std` - Standard deviation
- `depth_min/max` - Depth range
- `confidence_mean/std` - Match quality statistics
- `vertical_alignment_mean/std` - LiDAR-sonar agreement

#### Ground Truth Comparison (if available):
- `rmse` - Root Mean Square Error
- `mae` - Mean Absolute Error
- `bias` - Systematic error
- `num_matched_points` - Points compared to ground truth

### Requirements Satisfied:
- R5: Reports quantitative results on accuracy
- R7: Verifies vertical alignment
- R12: Documents feasibility analysis

---

## EA7 – Visualize Combined 3D Model

**Description:** Display the merged Sonar–LiDAR data together in a single 3D view for easier interpretation and analysis.

### Implementation:
- **Class:** `Visualizer`
- **Method:** `Visualizer.plot_water_surface()`

### Code Location:
```python
# Lines 579-641: Visualizer class and plot_water_surface() method
# Lines 729-731: EA7 execution in workflow
```

### Visualization Components:

#### Plot 1: Water Surface Elevation (LiDAR)
- 2D scatter plot colored by elevation
- Shows above-water topography

#### Plot 2: Bottom Depth (Sonar)
- 2D scatter plot colored by depth
- Shows underwater bathymetry

#### Plot 3: Water Depth (Combined)
- 2D scatter plot colored by water column height
- Shows integrated measurement

#### Plot 4: 3D View
- 3D scatter showing both surface and bottom
- Blue points: LiDAR surface
- Red points: Sonar bottom
- Demonstrates complete water column profile

### Output:
- High-resolution PNG (300 DPI)
- 15x10 inch figure with 4 subplots

### Requirements Satisfied:
- Part of R11: Visualizes pilot dataset
- Demonstrates integrated workflow

---

## EA8 – Document Process

**Description:** Write clear steps showing how to align, normalize, and merge the two datasets so the process can be repeated consistently.

### Implementation:
- **Method:** `DataFusion.export_results()`
- **Documentation:** Comprehensive logging throughout entire pipeline

### Code Location:
```python
# Lines 1-47: File header with complete EA documentation
# Lines 544-577: export_results() method with metrics export
# Lines 643-763: main() function with step-by-step logging
# Lines 733-755: Final summary and EA checklist
```

### Documentation Components:

#### 1. Process Logging:
Every EA step logs:
- Action being performed
- Input parameters
- Number of points processed
- Results and metrics

#### 2. Exported Files:

**fused_water_data.csv:**
```csv
x, y, surface_elevation, bottom_depth, water_depth, timestamp, confidence
```

**fused_water_data_metrics.csv:**
```csv
num_points, depth_mean, depth_std, depth_min, depth_max,
confidence_mean, confidence_std, vertical_alignment_mean,
vertical_alignment_std, rmse, mae, bias
```

**fused_water_data_alignment_errors.csv:**
```csv
x, y, vertical_error, original_depth, corrected_depth
```

#### 3. Workflow Summary:
Final output includes:
- ✓ Checklist of completed EAs
- Validation metrics summary
- File locations
- Processing statistics

### Requirements Satisfied:
- R8: Supports CSV export format
- R11: Demonstrates complete workflow
- R12: Documents all methods and results

---

## Complete Workflow Execution

### Order of Operations:
```python
# Initialize system with configurable parameters
fusion = DataFusion(
    spatial_tolerance=2.0,      # meters
    temporal_tolerance=30.0,    # seconds  
    mlw_datum=0.5,             # MLW offset
    sensor_offset_x=0.0,       # X offset
    sensor_offset_y=0.0,       # Y offset
    sensor_offset_z=1.5        # Z offset (e.g., LiDAR height on USV)
)

# EA1: Collect data
lidar_points = HSXParser(hsx_file).parse_lidar_data()
sonar_pings = S7KParser(s7k_file).parse_sonar_data()

# EA2: Align coordinate systems
lidar_aligned, sonar_aligned = fusion.apply_coordinate_transformation(
    lidar_points, sonar_pings
)

# EA3: Normalize to MLW datum
lidar_norm, sonar_norm = fusion.normalize_to_mlw_datum(
    lidar_aligned, sonar_aligned, tidal_correction=0.2
)

# EA4: Merge point clouds
water_points = fusion.align_datasets(lidar_norm, sonar_norm)

# EA5: Check and fix errors
water_points_corrected = fusion.check_alignment_errors(
    water_points, tolerance=0.5
)

# EA6: Validate accuracy
metrics = fusion.validate_accuracy(water_points_corrected)

# EA7: Visualize 3D model
Visualizer.plot_water_surface(water_points_corrected, 'output.png')

# EA8: Document and export
fusion.export_results(water_points_corrected, 'results.csv', 
                     include_metrics=True)
```

---

## Requirements Traceability

| Engineering Action | Requirements Satisfied |
|-------------------|------------------------|
| EA1 | R1, R2, R3, R10, R12 |
| EA2 | R5, R9 |
| EA3 | R6, R7 |
| EA4 | R4, R5, R11 |
| EA5 | R5, R7 |
| EA6 | R5, R7, R12 |
| EA7 | R11 |
| EA8 | R8, R11, R12 |

All 12 requirements (R1-R12) are satisfied through the implementation of EA1-EA8.

---

## Configuration Parameters

### Adjustable for Different USV Configurations (R9):

```python
# Sensor offsets (meters) - adjust for different hull configurations
sensor_offset_x: float = 0.0    # Forward/aft offset
sensor_offset_y: float = 0.0    # Port/starboard offset  
sensor_offset_z: float = 1.5    # Vertical offset (height above waterline)

# Datum parameters
mlw_datum: float = 0.5          # Mean Low Water offset from reference
tidal_correction: float = 0.2   # Real-time tidal adjustment

# Processing parameters
spatial_tolerance: float = 2.0   # Maximum spatial distance for matching (m)
temporal_tolerance: float = 30.0 # Maximum time difference for matching (s)
alignment_tolerance: float = 0.5 # Error threshold for EA5 correction (m)
```

---

## Testing and Validation

To verify all EA implementations:

```bash
# Run complete pipeline
python lidar_sonar_fusion.py

# Check output files
ls -lh fused_water_data*.csv
ls -lh water_surface_analysis.png

# Verify EA execution in logs
cat water_monitoring.log | grep "EA[1-8]"
```

Expected console output will show:
- ✓ EA1: Collected Sonar (.S7K) and LiDAR (.HSX/.RAW) data
- ✓ EA2: Aligned data to same coordinate system with GNSS/RTK offsets  
- ✓ EA3: Normalized to Mean Low Water (MLW) datum
- ✓ EA4: Merged point clouds into unified dataset
- ✓ EA5: Checked and corrected alignment errors
- ✓ EA6: Validated accuracy with RMSE and statistical metrics
- ✓ EA7: Visualized combined 3D model
- ✓ EA8: Documented and exported all results

---

## Summary

All Engineering Actions (EA1-EA8) are fully implemented in `lidar_sonar_fusion.py`:

1. ✅ **EA1** - Data collection parsers functional
2. ✅ **EA2** - Coordinate transformation with configurable offsets
3. ✅ **EA3** - MLW datum normalization with tidal corrections
4. ✅ **EA4** - Point cloud merging with spatial-temporal alignment
5. ✅ **EA5** - Alignment error detection and median filtering correction
6. ✅ **EA6** - Comprehensive accuracy validation (RMSE, MAE, bias, statistics)
7. ✅ **EA7** - Multi-panel 3D visualization
8. ✅ **EA8** - Complete documentation, logging, and export functionality

The system provides a complete, documented, and validated workflow for merging LIDAR and Sonar datasets for water surface detection and bathymetric analysis.

