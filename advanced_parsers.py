#!/usr/bin/env python3
"""
Advanced parsers for HYPACK and NORBIT binary formats
These implementations provide more robust binary data parsing
"""

import struct
import numpy as np
import datetime
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class HSXRecord:
    """Represents a parsed HSX record"""
    record_type: str
    timestamp: datetime.datetime
    device_id: int
    data: Dict

class AdvancedHSXParser:
    """Advanced HSX/RAW parser with better binary format handling"""
    
    def __init__(self, hsx_file: str):
        self.hsx_file = Path(hsx_file)
        self.raw_file = self.hsx_file.with_suffix('.RAW')
        self.header_info = {}
        self.devices = {}
        self.geodetic_info = {}
        
    def parse_complete_header(self) -> Dict:
        """Parse complete HSX header with all sections"""
        logger.info(f"Parsing complete HSX header from {self.hsx_file}")
        
        with open(self.hsx_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            record_type = parts[0]
            
            # Basic info
            if record_type == 'TND':
                self.header_info['time'] = parts[1] if len(parts) > 1 else ''
                self.header_info['date'] = parts[2] if len(parts) > 2 else ''
                
            elif record_type == 'INF':
                # Project information
                self.header_info['project_info'] = ' '.join(parts[1:])
                
            elif record_type == 'HSP':
                # Hypack survey parameters
                if len(parts) >= 5:
                    self.header_info['survey_params'] = {
                        'start_angle': float(parts[1]) if parts[1] != '' else 0.0,
                        'end_angle': float(parts[2]) if parts[2] != '' else 0.0,
                        'port_range': float(parts[3]) if parts[3] != '' else 0.0,
                        'starboard_range': float(parts[4]) if parts[4] != '' else 0.0
                    }
                    
            elif record_type == 'DEV':
                # Device definition
                if len(parts) >= 4:
                    device_id = int(parts[1])
                    device_type = int(parts[2])
                    device_name = ' '.join(parts[3:]).strip('"')
                    self.devices[device_id] = {
                        'type': device_type,
                        'name': device_name,
                        'offsets': {},
                        'parameters': {}
                    }
                    
            elif record_type == 'OF2':
                # Device offsets
                if len(parts) >= 9:
                    device_id = int(parts[1])
                    offset_type = int(parts[2])
                    if device_id in self.devices:
                        self.devices[device_id]['offsets'] = {
                            'type': offset_type,
                            'x': float(parts[3]),
                            'y': float(parts[4]),
                            'z': float(parts[5]),
                            'roll': float(parts[6]),
                            'pitch': float(parts[7]),
                            'yaw': float(parts[8]),
                            'delay': float(parts[9]) if len(parts) > 9 else 0.0
                        }
                        
            elif record_type == 'MBI':
                # Multibeam/LIDAR parameters
                if len(parts) >= 8:
                    device_id = int(parts[1])
                    if device_id in self.devices:
                        self.devices[device_id]['parameters'] = {
                            'beams': int(parts[2]),
                            'samples': int(parts[3]),
                            'frequency': int(parts[4]) if parts[4] != '0' else 0,
                            'pulse_length': int(parts[5]),
                            'beam_spacing': float(parts[6]),
                            'sound_velocity': float(parts[7])
                        }
                        
            elif record_type == 'INI':
                # Geodetic initialization
                if len(parts) >= 2:
                    key = parts[1]
                    value = '='.join(parts[1:]).split('=', 1)[1] if '=' in line else ''
                    self.geodetic_info[key] = value
        
        return self.header_info
    
    def parse_raw_data_structure(self) -> List[HSXRecord]:
        """Analyze RAW file structure without full parsing"""
        logger.info(f"Analyzing RAW file structure: {self.raw_file}")
        
        if not self.raw_file.exists():
            logger.error(f"RAW file not found: {self.raw_file}")
            return []
        
        records = []
        file_size = self.raw_file.stat().st_size
        logger.info(f"RAW file size: {file_size / (1024*1024):.1f} MB")
        
        try:
            with open(self.raw_file, 'rb') as f:
                position = 0
                record_count = 0
                
                while position < file_size and record_count < 100:  # Limit for analysis
                    # Try to read potential record header
                    header_data = f.read(32)  # Read larger header
                    if len(header_data) < 32:
                        break
                    
                    # Look for patterns in the header
                    # Different record types have different structures
                    potential_patterns = self._identify_record_patterns(header_data, position)
                    
                    if potential_patterns:
                        for pattern in potential_patterns:
                            record = HSXRecord(
                                record_type=pattern['type'],
                                timestamp=pattern['timestamp'],
                                device_id=pattern['device_id'],
                                data={'size': pattern['size'], 'position': position}
                            )
                            records.append(record)
                            
                            # Skip to next potential record
                            f.seek(position + pattern['size'])
                            position += pattern['size']
                            record_count += 1
                            break
                    else:
                        # Move forward and try again
                        f.seek(position + 1)
                        position += 1
        
        except Exception as e:
            logger.error(f"Error analyzing RAW structure: {e}")
        
        logger.info(f"Found {len(records)} potential records in structure analysis")
        return records
    
    def _identify_record_patterns(self, data: bytes, position: int) -> List[Dict]:
        """Identify potential record patterns in binary data"""
        patterns = []
        
        # Common HYPACK record patterns
        if len(data) >= 20:
            # Pattern 1: Standard navigation/positioning record
            try:
                # Many HYPACK records start with sync patterns
                sync1 = struct.unpack('<H', data[0:2])[0]
                sync2 = struct.unpack('<H', data[2:4])[0]
                
                # Look for common sync patterns
                if sync1 in [0x2400, 0x2401, 0x2402]:  # Common HYPACK sync patterns
                    device_id = struct.unpack('<H', data[4:6])[0]
                    timestamp = struct.unpack('<I', data[8:12])[0]
                    record_size = struct.unpack('<H', data[12:14])[0]
                    
                    patterns.append({
                        'type': f'HYPACK_{sync1:04X}',
                        'device_id': device_id,
                        'timestamp': datetime.datetime.fromtimestamp(timestamp),
                        'size': record_size
                    })
                    
            except (struct.error, OSError, ValueError):
                pass
            
            # Pattern 2: LIDAR data records
            try:
                # Velodyne packets often have specific patterns
                if data[0:2] == b'\xFF\xEE':  # Velodyne sync
                    patterns.append({
                        'type': 'VELODYNE_DATA',
                        'device_id': 2,  # Assuming LIDAR is device 2
                        'timestamp': datetime.datetime.now(),  # Would need proper extraction
                        'size': 1206  # Standard Velodyne packet size
                    })
                    
            except (struct.error, OSError, ValueError):
                pass
        
        return patterns

class AdvancedS7KParser:
    """Advanced S7K parser with comprehensive record handling"""
    
    def __init__(self, s7k_file: str):
        self.s7k_file = Path(s7k_file)
        self.s7k_version = None
        self.record_types = {}
        
    def analyze_s7k_structure(self) -> Dict:
        """Analyze S7K file structure and record types"""
        logger.info(f"Analyzing S7K structure: {self.s7k_file}")
        
        file_size = self.s7k_file.stat().st_size
        logger.info(f"S7K file size: {file_size / (1024*1024):.1f} MB")
        
        analysis = {
            'file_size': file_size,
            'record_types': {},
            'total_records': 0,
            'time_range': {'start': None, 'end': None}
        }
        
        try:
            with open(self.s7k_file, 'rb') as f:
                while f.tell() < file_size:
                    # Read potential S7K record header
                    header_pos = f.tell()
                    header = f.read(64)
                    
                    if len(header) < 64:
                        break
                    
                    # Check for S7K sync pattern
                    sync_pattern = struct.unpack('<H', header[0:2])[0]
                    
                    if sync_pattern == 0x0000:  # S7K sync pattern
                        try:
                            # Parse S7K header
                            record_type = struct.unpack('<I', header[8:12])[0]
                            timestamp = struct.unpack('<Q', header[16:24])[0]
                            record_size = struct.unpack('<I', header[32:36])[0]
                            
                            # Convert timestamp (S7K uses different epoch)
                            dt = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=timestamp/1000000.0)
                            
                            # Update analysis
                            if record_type not in analysis['record_types']:
                                analysis['record_types'][record_type] = {
                                    'count': 0,
                                    'name': self._get_s7k_record_name(record_type),
                                    'size_range': [record_size, record_size]
                                }
                            
                            analysis['record_types'][record_type]['count'] += 1
                            size_range = analysis['record_types'][record_type]['size_range']
                            size_range[0] = min(size_range[0], record_size)
                            size_range[1] = max(size_range[1], record_size)
                            
                            # Update time range
                            if analysis['time_range']['start'] is None or dt < analysis['time_range']['start']:
                                analysis['time_range']['start'] = dt
                            if analysis['time_range']['end'] is None or dt > analysis['time_range']['end']:
                                analysis['time_range']['end'] = dt
                            
                            analysis['total_records'] += 1
                            
                            # Skip to next record
                            f.seek(header_pos + record_size)
                            
                        except (struct.error, OSError, ValueError) as e:
                            # Not a valid S7K record, move forward
                            f.seek(header_pos + 1)
                            
                    else:
                        # Not S7K sync, move forward
                        f.seek(header_pos + 1)
                    
                    # Limit analysis for very large files
                    if analysis['total_records'] > 10000:
                        logger.info("Limiting analysis to first 10000 records")
                        break
                        
        except Exception as e:
            logger.error(f"Error analyzing S7K structure: {e}")
        
        # Log analysis results
        logger.info(f"S7K Analysis Results:")
        logger.info(f"  Total records: {analysis['total_records']}")
        logger.info(f"  Record types found: {len(analysis['record_types'])}")
        
        for record_type, info in analysis['record_types'].items():
            logger.info(f"    {record_type} ({info['name']}): {info['count']} records, "
                       f"size {info['size_range'][0]}-{info['size_range'][1]} bytes")
        
        if analysis['time_range']['start']:
            logger.info(f"  Time range: {analysis['time_range']['start']} to {analysis['time_range']['end']}")
        
        return analysis
    
    def _get_s7k_record_name(self, record_type: int) -> str:
        """Get human-readable name for S7K record type"""
        s7k_record_names = {
            1000: "Reference Point",
            1001: "Sensor Position",
            1002: "Sensor Position Relative",
            1003: "Sensor Position Time Series",
            1012: "Roll Pitch Heading",
            1013: "Roll Pitch Heading Time Series",
            1015: "Navigation",
            1016: "Attitude Time Series",
            7000: "Sonar Settings",
            7001: "Configuration",
            7002: "Calibration",
            7004: "Beam Geometry",
            7006: "Bathymetric Data",
            7007: "Side Scan Data",
            7008: "Generic Water Column",
            7010: "TVG",
            7011: "Image",
            7012: "Ping Motion",
            7027: "Raw Detection Data",
            7028: "Snippet",
            7030: "Sonar Installation Parameters",
            7041: "Beam Formed Data",
            7042: "Calibrated Beam Data",
            7048: "Backscatter Imagery",
            7057: "Generic Sensor",
            7058: "Specific Sensor",
            7059: "Calibrated Side Scan",
            7200: "File Header"
        }
        
        return s7k_record_names.get(record_type, f"Unknown_{record_type}")

def create_data_extraction_report():
    """Create a comprehensive report of available data"""
    logger.info("Creating data extraction report...")
    
    data_dir = Path("/Users/tycrouch/Desktop/untitled folder 4/Data for OARS - Copy")
    report = {
        'lidar_files': [],
        'sonar_files': [],
        'processing_recommendations': []
    }
    
    # Analyze LIDAR files
    lidar_dir = data_dir / "HYPACK iLIDAR DATA"
    for test_dir in lidar_dir.iterdir():
        if test_dir.is_dir():
            for hsx_file in test_dir.glob("*.HSX"):
                logger.info(f"Analyzing LIDAR file: {hsx_file.name}")
                
                parser = AdvancedHSXParser(str(hsx_file))
                header = parser.parse_complete_header()
                structure = parser.parse_raw_data_structure()
                
                file_info = {
                    'file': str(hsx_file),
                    'test_directory': test_dir.name,
                    'header_info': header,
                    'devices': parser.devices,
                    'geodetic_info': parser.geodetic_info,
                    'record_analysis': len(structure),
                    'file_size_mb': hsx_file.stat().st_size / (1024*1024)
                }
                
                report['lidar_files'].append(file_info)
    
    # Analyze sonar files
    sonar_dir = data_dir / "WBMS"
    for session_dir in sonar_dir.iterdir():
        if session_dir.is_dir():
            for s7k_file in session_dir.glob("*.s7k"):
                logger.info(f"Analyzing sonar file: {s7k_file.name}")
                
                parser = AdvancedS7KParser(str(s7k_file))
                analysis = parser.analyze_s7k_structure()
                
                file_info = {
                    'file': str(s7k_file),
                    'session': session_dir.name,
                    'analysis': analysis,
                    'file_size_mb': s7k_file.stat().st_size / (1024*1024)
                }
                
                report['sonar_files'].append(file_info)
    
    # Generate processing recommendations
    if report['lidar_files'] and report['sonar_files']:
        report['processing_recommendations'].extend([
            "Both LIDAR and sonar data available for fusion",
            "Recommend temporal alignment based on timestamps",
            "Check coordinate system consistency between datasets",
            "Consider data quality filtering based on file analysis"
        ])
    
    return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = create_data_extraction_report()
    
    print("\n" + "="*60)
    print("DATA EXTRACTION REPORT")
    print("="*60)
    
    print(f"\nLIDAR Files Found: {len(report['lidar_files'])}")
    for file_info in report['lidar_files']:
        print(f"  {Path(file_info['file']).name} ({file_info['file_size_mb']:.1f} MB)")
        print(f"    Devices: {len(file_info['devices'])}")
        for dev_id, dev_info in file_info['devices'].items():
            print(f"      {dev_id}: {dev_info['name']}")
    
    print(f"\nSonar Files Found: {len(report['sonar_files'])}")
    for file_info in report['sonar_files']:
        print(f"  {Path(file_info['file']).name} ({file_info['file_size_mb']:.1f} MB)")
        print(f"    Records: {file_info['analysis']['total_records']}")
        print(f"    Record types: {len(file_info['analysis']['record_types'])}")
    
    print(f"\nProcessing Recommendations:")
    for rec in report['processing_recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*60)

