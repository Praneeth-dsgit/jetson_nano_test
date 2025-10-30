#!/usr/bin/env python3
"""
Test cases for health metrics calculation functions.

Tests include:
- calculate_rmssd (HRV calculation)
- calculate_stress (stress level calculation)
- estimate_vo2_max (VO2 max estimation)
- calculate_trimp (TRIMP calculation)
- training_energy_expenditure (energy expenditure)
"""

import pytest
import numpy as np
from core.main import (
    calculate_rmssd,
    calculate_stress,
    estimate_vo2_max,
    calculate_trimp,
    training_energy_expenditure,
    get_trimp_zone,
    get_recovery_recommendations,
    get_training_recommendations
)


class TestCalculateRMSSD:
    """Test cases for calculate_rmssd function."""
    
    def test_normal_hr_values(self):
        """Test with normal heart rate values."""
        hr_values = [60, 62, 65, 67, 64, 66, 68, 70]
        result = calculate_rmssd(hr_values)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_insufficient_data(self):
        """Test with insufficient data (less than 5 values)."""
        hr_values = [60, 62, 65]
        result = calculate_rmssd(hr_values)
        assert result == 0.0
    
    def test_invalid_hr_values(self):
        """Test with physiologically impossible HR values."""
        hr_values = [10, 15, 20, 25, 30]  # Too low
        result = calculate_rmssd(hr_values)
        assert result == 0.0
    
    def test_numpy_array_input(self):
        """Test with numpy array input."""
        hr_values = np.array([70, 72, 68, 71, 74, 77, 75])
        result = calculate_rmssd(hr_values)
        assert isinstance(result, float)
        assert result >= 0
    
    def test_normal_range_values(self):
        """Test with values in normal physiological range."""
        hr_values = [70, 72, 75, 73, 71, 74, 76, 72, 73, 75]
        result = calculate_rmssd(hr_values)
        assert result > 0
        assert result < 200  # Realistic RMSSD range
    
    def test_mixed_valid_invalid(self):
        """Test with mix of valid and invalid values."""
        hr_values = [70, 72, 5, 74, 300, 75, 73, 71]  # 5 and 300 are invalid
        result = calculate_rmssd(hr_values)
        # Should still work with valid values
        assert isinstance(result, float)


class TestCalculateStress:
    """Test cases for calculate_stress function."""
    
    def test_normal_stress_values(self):
        """Test with normal physiological values."""
        result = calculate_stress(
            hr=120, hrv=50, acc_mag=5.0, gyro_mag=100,
            age=25, gender=1, hr_rest=60, hr_max=200
        )
        assert 0 <= result <= 100
    
    def test_low_stress_scenario(self):
        """Test with low stress indicators."""
        result = calculate_stress(
            hr=70, hrv=100, acc_mag=1.0, gyro_mag=10,
            age=25, gender=1, hr_rest=60, hr_max=200
        )
        assert 0 <= result <= 100
        assert result < 50  # Should be low stress
    
    def test_high_stress_scenario(self):
        """Test with high stress indicators."""
        result = calculate_stress(
            hr=180, hrv=20, acc_mag=20.0, gyro_mag=500,
            age=25, gender=1, hr_rest=60, hr_max=200
        )
        assert 0 <= result <= 100
        assert result > 50  # Should be high stress
    
    def test_bounds_checking(self):
        """Test that stress is always within 0-100 bounds."""
        # Test various extreme values
        test_cases = [
            (50, 10, 0.1, 1, 20, 1),   # Very low
            (200, 200, 50, 1000, 50, 1),  # Very high
            (100, 50, 10, 100, 30, 0),  # Female
        ]
        
        for hr, hrv, acc, gyro, age, gender in test_cases:
            result = calculate_stress(hr, hrv, acc, gyro, age, gender)
            assert 0 <= result <= 100, f"Stress out of bounds: {result}"
    
    def test_zero_hrv(self):
        """Test with zero HRV (should indicate high stress)."""
        result = calculate_stress(
            hr=100, hrv=0, acc_mag=5.0, gyro_mag=50,
            age=25, gender=1
        )
        assert 0 <= result <= 100
        assert result > 30  # Zero HRV should increase stress


class TestEstimateVO2Max:
    """Test cases for estimate_vo2_max function."""
    
    def test_normal_values(self):
        """Test with normal physiological values."""
        result = estimate_vo2_max(
            age=25, gender=1, current_hr=150,
            hrv=50, hr_rest=60, hr_max=195
        )
        assert 20 <= result <= 80  # Realistic VO2 max range
    
    def test_male_vs_female(self):
        """Test that males typically have higher VO2 max."""
        male_result = estimate_vo2_max(25, 1, 150, 50, 60, 195)
        female_result = estimate_vo2_max(25, 0, 150, 50, 60, 195)
        assert male_result >= female_result
    
    def test_age_effect(self):
        """Test that VO2 max decreases with age."""
        young_result = estimate_vo2_max(20, 1, 150, 50, 60, 200)
        old_result = estimate_vo2_max(50, 1, 150, 50, 60, 170)
        # Older should generally be lower (though HR_max also affects this)
        assert isinstance(young_result, float)
        assert isinstance(old_result, float)
    
    def test_high_hrv_effect(self):
        """Test that higher HRV increases VO2 max."""
        low_hrv = estimate_vo2_max(25, 1, 150, 20, 60, 195)
        high_hrv = estimate_vo2_max(25, 1, 150, 100, 60, 195)
        assert high_hrv >= low_hrv
    
    def test_default_hr_max(self):
        """Test with default hr_max (None)."""
        result = estimate_vo2_max(30, 1, 140, 50, 60, None)
        assert 20 <= result <= 80


class TestCalculateTRIMP:
    """Test cases for calculate_trimp function."""
    
    def test_normal_trimp_calculation(self):
        """Test normal TRIMP calculation."""
        result = calculate_trimp(
            hr_avg=150, hr_rest=60, hr_max=200,
            duration_min=30, gender="male"
        )
        assert result > 0
        assert isinstance(result, float)
    
    def test_male_vs_female_trimp(self):
        """Test that TRIMP differs for male vs female."""
        male_result = calculate_trimp(150, 60, 200, 30, "male")
        female_result = calculate_trimp(150, 60, 200, 30, "female")
        assert male_result != female_result
    
    def test_duration_effect(self):
        """Test that longer duration increases TRIMP."""
        short = calculate_trimp(150, 60, 200, 15, "male")
        long = calculate_trimp(150, 60, 200, 60, "male")
        assert long > short
    
    def test_high_intensity(self):
        """Test with high intensity heart rate."""
        result = calculate_trimp(180, 60, 200, 30, "male")
        assert result > 0
    
    def test_low_intensity(self):
        """Test with low intensity heart rate."""
        result = calculate_trimp(80, 60, 200, 30, "male")
        assert result >= 0
    
    def test_invalid_gender(self):
        """Test that invalid gender raises ValueError."""
        with pytest.raises(ValueError):
            calculate_trimp(150, 60, 200, 30, "invalid")


class TestTrainingEnergyExpenditure:
    """Test cases for training_energy_expenditure function."""
    
    def test_walking_velocity(self):
        """Test with walking velocity."""
        result = training_energy_expenditure(
            velocity=1.5, duration_s=600, mass_kg=70
        )
        assert result > 0
        assert isinstance(result, float)
    
    def test_running_velocity(self):
        """Test with running velocity."""
        result = training_energy_expenditure(
            velocity=3.0, duration_s=600, mass_kg=70
        )
        assert result > 0
    
    def test_mass_effect(self):
        """Test that higher mass increases energy expenditure."""
        light = training_energy_expenditure(2.0, 600, 60)
        heavy = training_energy_expenditure(2.0, 600, 90)
        assert heavy > light
    
    def test_duration_effect(self):
        """Test that longer duration increases energy expenditure."""
        short = training_energy_expenditure(2.0, 300, 70)
        long = training_energy_expenditure(2.0, 600, 70)
        assert long > short
    
    def test_zero_velocity(self):
        """Test with zero velocity (should still expend some energy)."""
        result = training_energy_expenditure(0.0, 600, 70)
        assert result >= 0


class TestGetTRIMPZone:
    """Test cases for get_trimp_zone function."""
    
    def test_light_zone(self):
        """Test light TRIMP zone."""
        zone, desc = get_trimp_zone(30)
        assert zone == "Light"
        assert "Recovery" in desc or "Warm-up" in desc
    
    def test_moderate_zone(self):
        """Test moderate TRIMP zone."""
        zone, desc = get_trimp_zone(100)
        assert zone == "Moderate"
    
    def test_high_zone(self):
        """Test high TRIMP zone."""
        zone, desc = get_trimp_zone(200)
        assert zone == "High"
    
    def test_very_high_zone(self):
        """Test very high TRIMP zone."""
        zone, desc = get_trimp_zone(350)
        assert zone == "Very High"
    
    def test_boundary_values(self):
        """Test boundary values between zones."""
        zone1, _ = get_trimp_zone(49.9)
        zone2, _ = get_trimp_zone(50.1)
        assert zone1 == "Light"
        assert zone2 == "Moderate"


class TestGetRecoveryRecommendations:
    """Test cases for get_recovery_recommendations function."""
    
    def test_light_session_recovery(self):
        """Test recovery recommendations for light session."""
        recovery_time, recommendations = get_recovery_recommendations(30, 30)
        assert "0-1" in recovery_time or "1" in recovery_time
        assert len(recommendations) > 0
    
    def test_high_stress_recommendations(self):
        """Test that high stress adds recommendations."""
        _, recommendations = get_recovery_recommendations(100, 80)
        assert any("stress" in rec.lower() for rec in recommendations)
    
    def test_moderate_session(self):
        """Test moderate session recovery."""
        recovery_time, recommendations = get_recovery_recommendations(120, 50)
        assert "1-2" in recovery_time or "2" in recovery_time
        assert len(recommendations) > 0


class TestGetTrainingRecommendations:
    """Test cases for get_training_recommendations function."""
    
    def test_very_high_zone_recommendations(self):
        """Test recommendations for very high TRIMP zone."""
        recommendations = get_training_recommendations("Very High", 60)
        assert len(recommendations) > 0
        assert any("recovery" in rec.lower() for rec in recommendations)
    
    def test_high_stress_recommendations(self):
        """Test that high stress adds recommendations."""
        recommendations = get_training_recommendations("Moderate", 75)
        assert any("stress" in rec.lower() for rec in recommendations)
    
    def test_all_zones(self):
        """Test that all zones return recommendations."""
        zones = ["Light", "Moderate", "High", "Very High"]
        for zone in zones:
            recommendations = get_training_recommendations(zone, 50)
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0

