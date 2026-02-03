#!/usr/bin/env python3
"""
Data Quality Assessment Module for Athlete Monitoring System

This module provides comprehensive sensor data quality assessment including:
- Sensor data reliability scoring
- Outlier detection and classification
- Data completeness analysis
- Physiological plausibility checks
- Real-time data quality monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import logging
from datetime import datetime, timedelta
import json

class SensorDataQualityAssessor:
    """
    Comprehensive sensor data quality assessment for athlete monitoring.
    
    Provides real-time and batch quality assessment of sensor data including
    accelerometer, gyroscope, magnetometer, and heart rate data.
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 enable_logging: bool = True,
                 quality_threshold: float = 0.7):
        """
        Initialize the data quality assessor.
        
        Args:
            window_size: Size of sliding window for quality assessment
            enable_logging: Whether to enable detailed logging
            quality_threshold: Minimum quality score threshold
        """
        self.window_size = window_size
        self.enable_logging = enable_logging
        self.quality_threshold = quality_threshold
        
        # Data buffers for quality assessment
        self.acc_buffer = deque(maxlen=window_size)
        self.gyro_buffer = deque(maxlen=window_size)
        self.hr_buffer = deque(maxlen=window_size)
        self.mag_buffer = deque(maxlen=window_size)
        
        # Quality metrics history
        self.quality_history = deque(maxlen=100)
        self.quality_scores = {
            'acceleration': deque(maxlen=50),
            'gyroscope': deque(maxlen=50),
            'heart_rate': deque(maxlen=50),
            'magnetometer': deque(maxlen=50),
            'overall': deque(maxlen=50)
        }
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Physiological limits
        self.physiological_limits = {
            'heart_rate': {'min': 30, 'max': 220},
            'acceleration': {'min': -50, 'max': 50},  # m/s²
            'gyroscope': {'min': -2000, 'max': 2000},  # deg/s
            'magnetometer': {'min': -100, 'max': 100}  # μT
        }
        
        # Quality assessment parameters
        self.quality_params = {
            'outlier_threshold': 3.0,  # Standard deviations
            'missing_data_threshold': 0.1,  # 10% missing data
            'noise_threshold': 0.5,  # Noise level threshold
            'consistency_window': 10  # Window for consistency checks
        }
    
    @staticmethod
    def _to_float(val, default: float = np.nan) -> float:
        """Convert value to float; return default if None, non-numeric, or invalid. Safe for np.isnan()."""
        if val is None:
            return default
        try:
            f = float(val)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default
    
    def assess_sensor_data_quality(self, 
                                 acc_data: Dict[str, float],
                                 gyro_data: Dict[str, float],
                                 hr_data: float,
                                 mag_data: Dict[str, float],
                                 timestamp: datetime = None) -> Dict[str, Any]:
        """
        Assess quality of incoming sensor data.
        
        Args:
            acc_data: Dictionary with 'x', 'y', 'z' acceleration values
            gyro_data: Dictionary with 'x', 'y', 'z' gyroscope values
            hr_data: Heart rate value
            mag_data: Dictionary with 'x', 'y', 'z' magnetometer values
            timestamp: Timestamp of the data
            
        Returns:
            Dictionary containing quality assessment results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Coerce inputs to float so np.isnan() and arithmetic work (handles string/None from pipe-delimited data)
        acc_data = {k: self._to_float(acc_data.get(k)) for k in ('x', 'y', 'z')}
        gyro_data = {k: self._to_float(gyro_data.get(k)) for k in ('x', 'y', 'z')}
        hr_data = self._to_float(hr_data)
        mag_data = {k: self._to_float(mag_data.get(k)) for k in ('x', 'y', 'z')}
        
        # Add data to buffers
        self._add_to_buffers(acc_data, gyro_data, hr_data, mag_data)
        
        # Assess individual sensor quality
        acc_quality = self._assess_acceleration_quality(acc_data)
        gyro_quality = self._assess_gyroscope_quality(gyro_data)
        hr_quality = self._assess_heart_rate_quality(hr_data)
        mag_quality = self._assess_magnetometer_quality(mag_data)
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(
            acc_quality, gyro_quality, hr_quality, mag_quality
        )
        
        # Store quality scores
        self.quality_scores['acceleration'].append(acc_quality['score'])
        self.quality_scores['gyroscope'].append(gyro_quality['score'])
        self.quality_scores['heart_rate'].append(hr_quality['score'])
        self.quality_scores['magnetometer'].append(mag_quality['score'])
        self.quality_scores['overall'].append(overall_quality)
        
        # Create comprehensive quality report
        quality_report = {
            'timestamp': timestamp.isoformat(),
            'overall_quality_score': overall_quality,
            'quality_status': self._get_quality_status(overall_quality),
            'sensor_quality': {
                'acceleration': acc_quality,
                'gyroscope': gyro_quality,
                'heart_rate': hr_quality,
                'magnetometer': mag_quality
            },
            'data_completeness': self._assess_data_completeness(acc_data, gyro_data, hr_data, mag_data),
            'physiological_plausibility': self._assess_physiological_plausibility(acc_data, gyro_data, hr_data, mag_data),
            'recommendations': self._generate_quality_recommendations(overall_quality, acc_quality, gyro_quality, hr_quality, mag_quality)
        }
        
        # Log quality assessment if enabled
        if self.enable_logging and self.logger:
            self._log_quality_assessment(quality_report)
        
        return quality_report
    
    def _add_to_buffers(self, acc_data, gyro_data, hr_data, mag_data):
        """Add sensor data to assessment buffers."""
        # Acceleration magnitude
        acc_mag = np.sqrt(acc_data['x']**2 + acc_data['y']**2 + acc_data['z']**2)
        self.acc_buffer.append(acc_mag)
        
        # Gyroscope magnitude
        gyro_mag = np.sqrt(gyro_data['x']**2 + gyro_data['y']**2 + gyro_data['z']**2)
        self.gyro_buffer.append(gyro_mag)
        
        # Heart rate
        self.hr_buffer.append(hr_data)
        
        # Magnetometer magnitude
        mag_mag = np.sqrt(mag_data['x']**2 + mag_data['y']**2 + mag_data['z']**2)
        self.mag_buffer.append(mag_mag)
    
    def _assess_acceleration_quality(self, acc_data: Dict[str, float]) -> Dict[str, Any]:
        """Assess acceleration data quality."""
        quality_score = 1.0
        issues = []
        warnings = []
        
        # Check for missing values
        if any(np.isnan([acc_data['x'], acc_data['y'], acc_data['z']])):
            quality_score *= 0.0
            issues.append("Missing acceleration values")
        
        # Check physiological limits
        acc_mag = np.sqrt(acc_data['x']**2 + acc_data['y']**2 + acc_data['z']**2)
        if acc_mag > self.physiological_limits['acceleration']['max']:
            quality_score *= 0.3
            issues.append(f"Acceleration magnitude too high: {acc_mag:.2f} m/s²")
        elif acc_mag < 0.1:  # Too low (sensor might be dead)
            quality_score *= 0.7
            warnings.append(f"Acceleration magnitude very low: {acc_mag:.2f} m/s²")
        
        # Check for outliers using buffer data
        if len(self.acc_buffer) >= 5:
            recent_data = list(self.acc_buffer)[-5:]
            mean_acc = np.mean(recent_data)
            std_acc = np.std(recent_data)
            
            if std_acc > 0:
                z_score = abs(acc_mag - mean_acc) / std_acc
                if z_score > self.quality_params['outlier_threshold']:
                    quality_score *= 0.8
                    warnings.append(f"Acceleration outlier detected (z-score: {z_score:.2f})")
        
        # Check for noise (high frequency variations)
        if len(self.acc_buffer) >= 3:
            recent_data = list(self.acc_buffer)[-3:]
            noise_level = np.std(np.diff(recent_data))
            if noise_level > self.quality_params['noise_threshold']:
                quality_score *= 0.9
                warnings.append(f"High acceleration noise detected: {noise_level:.3f}")
        
        return {
            'score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'magnitude': float(acc_mag),
            'components': {
                'x': float(acc_data['x']),
                'y': float(acc_data['y']),
                'z': float(acc_data['z'])
            }
        }
    
    def _assess_gyroscope_quality(self, gyro_data: Dict[str, float]) -> Dict[str, Any]:
        """Assess gyroscope data quality."""
        quality_score = 1.0
        issues = []
        warnings = []
        
        # Check for missing values
        if any(np.isnan([gyro_data['x'], gyro_data['y'], gyro_data['z']])):
            quality_score *= 0.0
            issues.append("Missing gyroscope values")
        
        # Check physiological limits
        gyro_mag = np.sqrt(gyro_data['x']**2 + gyro_data['y']**2 + gyro_data['z']**2)
        if gyro_mag > self.physiological_limits['gyroscope']['max']:
            quality_score *= 0.3
            issues.append(f"Gyroscope magnitude too high: {gyro_mag:.2f} deg/s")
        
        # Check for outliers
        if len(self.gyro_buffer) >= 5:
            recent_data = list(self.gyro_buffer)[-5:]
            mean_gyro = np.mean(recent_data)
            std_gyro = np.std(recent_data)
            
            if std_gyro > 0:
                z_score = abs(gyro_mag - mean_gyro) / std_gyro
                if z_score > self.quality_params['outlier_threshold']:
                    quality_score *= 0.8
                    warnings.append(f"Gyroscope outlier detected (z-score: {z_score:.2f})")
        
        return {
            'score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'magnitude': float(gyro_mag),
            'components': {
                'x': float(gyro_data['x']),
                'y': float(gyro_data['y']),
                'z': float(gyro_data['z'])
            }
        }
    
    def _assess_heart_rate_quality(self, hr_data: float) -> Dict[str, Any]:
        """Assess heart rate data quality."""
        quality_score = 1.0
        issues = []
        warnings = []
        
        # Check for missing values
        if np.isnan(hr_data) or hr_data is None:
            quality_score *= 0.0
            issues.append("Missing heart rate value")
            return {
                'score': quality_score,
                'issues': issues,
                'warnings': warnings,
                'value': None
            }
        
        # Check physiological limits
        if hr_data < self.physiological_limits['heart_rate']['min']:
            quality_score *= 0.2
            issues.append(f"Heart rate too low: {hr_data:.1f} BPM")
        elif hr_data > self.physiological_limits['heart_rate']['max']:
            quality_score *= 0.2
            issues.append(f"Heart rate too high: {hr_data:.1f} BPM")
        
        # Check for outliers using buffer data
        if len(self.hr_buffer) >= 5:
            recent_data = list(self.hr_buffer)[-5:]
            mean_hr = np.mean(recent_data)
            std_hr = np.std(recent_data)
            
            if std_hr > 0:
                z_score = abs(hr_data - mean_hr) / std_hr
                if z_score > self.quality_params['outlier_threshold']:
                    quality_score *= 0.8
                    warnings.append(f"Heart rate outlier detected (z-score: {z_score:.2f})")
        
        # Check for sudden changes (physiologically implausible)
        if len(self.hr_buffer) >= 2:
            prev_hr = self.hr_buffer[-2]
            hr_change = abs(hr_data - prev_hr)
            if hr_change > 30:  # More than 30 BPM change in one reading
                quality_score *= 0.7
                warnings.append(f"Sudden heart rate change: {hr_change:.1f} BPM")
        
        return {
            'score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'value': float(hr_data)
        }
    
    def _assess_magnetometer_quality(self, mag_data: Dict[str, float]) -> Dict[str, Any]:
        """Assess magnetometer data quality."""
        quality_score = 1.0
        issues = []
        warnings = []
        
        # Check for missing values
        if any(np.isnan([mag_data['x'], mag_data['y'], mag_data['z']])):
            quality_score *= 0.0
            issues.append("Missing magnetometer values")
        
        # Check for reasonable magnetometer values
        mag_mag = np.sqrt(mag_data['x']**2 + mag_data['y']**2 + mag_data['z']**2)
        if mag_mag > self.physiological_limits['magnetometer']['max']:
            quality_score *= 0.5
            warnings.append(f"Magnetometer magnitude high: {mag_mag:.2f} μT")
        
        return {
            'score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'magnitude': float(mag_mag),
            'components': {
                'x': float(mag_data['x']),
                'y': float(mag_data['y']),
                'z': float(mag_data['z'])
            }
        }
    
    def _calculate_overall_quality(self, acc_quality, gyro_quality, hr_quality, mag_quality) -> float:
        """Calculate overall data quality score."""
        # Weighted average with heart rate having higher importance
        weights = {'acceleration': 0.25, 'gyroscope': 0.25, 'heart_rate': 0.35, 'magnetometer': 0.15}
        
        overall_score = (
            weights['acceleration'] * acc_quality['score'] +
            weights['gyroscope'] * gyro_quality['score'] +
            weights['heart_rate'] * hr_quality['score'] +
            weights['magnetometer'] * mag_quality['score']
        )
        
        return round(overall_score, 3)
    
    def _get_quality_status(self, quality_score: float) -> str:
        """Get quality status based on score."""
        if quality_score >= 0.9:
            return "Excellent"
        elif quality_score >= 0.8:
            return "Good"
        elif quality_score >= 0.7:
            return "Fair"
        elif quality_score >= 0.5:
            return "Poor"
        else:
            return "Critical"
    
    def _assess_data_completeness(self, acc_data, gyro_data, hr_data, mag_data) -> Dict[str, Any]:
        """Assess data completeness."""
        total_fields = 10  # 3 acc + 3 gyro + 1 hr + 3 mag
        missing_fields = 0
        
        # Check acceleration
        if any(np.isnan([acc_data['x'], acc_data['y'], acc_data['z']])):
            missing_fields += 3
        
        # Check gyroscope
        if any(np.isnan([gyro_data['x'], gyro_data['y'], gyro_data['z']])):
            missing_fields += 3
        
        # Check heart rate
        if np.isnan(hr_data) or hr_data is None:
            missing_fields += 1
        
        # Check magnetometer
        if any(np.isnan([mag_data['x'], mag_data['y'], mag_data['z']])):
            missing_fields += 3
        
        completeness_score = (total_fields - missing_fields) / total_fields
        
        return {
            'score': completeness_score,
            'missing_fields': missing_fields,
            'total_fields': total_fields,
            'completeness_percentage': completeness_score * 100
        }
    
    def _assess_physiological_plausibility(self, acc_data, gyro_data, hr_data, mag_data) -> Dict[str, Any]:
        """Assess physiological plausibility of the data."""
        plausibility_score = 1.0
        issues = []
        
        # Check if all sensors are within reasonable ranges
        acc_mag = np.sqrt(acc_data['x']**2 + acc_data['y']**2 + acc_data['z']**2)
        if acc_mag > 20:  # Very high acceleration
            plausibility_score *= 0.5
            issues.append("Extremely high acceleration detected")
        
        gyro_mag = np.sqrt(gyro_data['x']**2 + gyro_data['y']**2 + gyro_data['z']**2)
        if gyro_mag > 1000:  # Very high rotation
            plausibility_score *= 0.5
            issues.append("Extremely high rotation detected")
        
        if hr_data and (hr_data < 20 or hr_data > 250):
            plausibility_score *= 0.3
            issues.append("Physiologically implausible heart rate")
        
        return {
            'score': plausibility_score,
            'issues': issues,
            'status': 'Plausible' if plausibility_score > 0.8 else 'Questionable'
        }
    
    def _generate_quality_recommendations(self, overall_quality, acc_quality, gyro_quality, hr_quality, mag_quality) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if overall_quality < self.quality_threshold:
            recommendations.append("Overall data quality is below threshold - check sensor connections")
        
        if acc_quality['score'] < 0.8:
            recommendations.append("Acceleration data quality issues detected - verify accelerometer placement")
        
        if gyro_quality['score'] < 0.8:
            recommendations.append("Gyroscope data quality issues detected - check for sensor interference")
        
        if hr_quality['score'] < 0.8:
            recommendations.append("Heart rate data quality issues detected - verify HR sensor contact")
        
        if mag_quality['score'] < 0.8:
            recommendations.append("Magnetometer data quality issues detected - check for magnetic interference")
        
        if not recommendations:
            recommendations.append("Data quality is good - continue monitoring")
        
        return recommendations
    
    def _log_quality_assessment(self, quality_report: Dict[str, Any]):
        """Log quality assessment results."""
        if self.logger:
            self.logger.info(f"Data Quality Assessment: {quality_report['overall_quality_score']:.3f} ({quality_report['quality_status']})")
            
            for sensor, quality in quality_report['sensor_quality'].items():
                if quality['issues']:
                    self.logger.warning(f"{sensor.title()} issues: {', '.join(quality['issues'])}")
                if quality['warnings']:
                    self.logger.info(f"{sensor.title()} warnings: {', '.join(quality['warnings'])}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of recent quality assessments."""
        if not self.quality_scores['overall']:
            return {'message': 'No quality data available'}
        
        recent_scores = list(self.quality_scores['overall'])[-10:]  # Last 10 assessments
        
        return {
            'average_quality': float(np.mean(recent_scores)),
            'quality_trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable',
            'sensor_averages': {
                sensor: float(np.mean(list(scores)[-10:])) if scores else 0.0
                for sensor, scores in self.quality_scores.items()
            },
            'total_assessments': len(self.quality_scores['overall'])
        }
    
    def reset_quality_assessment(self):
        """Reset all quality assessment buffers and history."""
        self.acc_buffer.clear()
        self.gyro_buffer.clear()
        self.hr_buffer.clear()
        self.mag_buffer.clear()
        self.quality_history.clear()
        
        for scores in self.quality_scores.values():
            scores.clear()
        
        if self.enable_logging and self.logger:
            self.logger.info("Quality assessment buffers reset")


# Convenience function for integration
def create_data_quality_assessor(window_size: int = 30, 
                                enable_logging: bool = True,
                                quality_threshold: float = 0.7) -> SensorDataQualityAssessor:
    """
    Create a data quality assessor with default settings.
    
    Args:
        window_size: Size of sliding window for quality assessment
        enable_logging: Whether to enable detailed logging
        quality_threshold: Minimum quality score threshold
        
    Returns:
        Configured SensorDataQualityAssessor instance
    """
    return SensorDataQualityAssessor(
        window_size=window_size,
        enable_logging=enable_logging,
        quality_threshold=quality_threshold
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create assessor
    assessor = create_data_quality_assessor()
    
    # Test with sample data
    test_data = {
        'acc': {'x': 0.1, 'y': 0.2, 'z': 9.8},
        'gyro': {'x': 0.5, 'y': -0.3, 'z': 0.1},
        'hr': 75.0,
        'mag': {'x': 25.0, 'y': -15.0, 'z': 40.0}
    }
    
    # Assess quality
    quality_report = assessor.assess_sensor_data_quality(
        test_data['acc'], test_data['gyro'], test_data['hr'], test_data['mag']
    )
    
    print("Quality Assessment Results:")
    print(f"Overall Quality: {quality_report['overall_quality_score']:.3f} ({quality_report['quality_status']})")
    print(f"Recommendations: {', '.join(quality_report['recommendations'])}")
    
    # Get summary
    summary = assessor.get_quality_summary()
    print(f"\nQuality Summary: {summary}")
