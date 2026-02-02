"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

# Try to import ConfigDict for Pydantic v2, fallback to Config for v1
try:
    from pydantic import ConfigDict
    PYDANTIC_V2 = True
except ImportError:
    PYDANTIC_V2 = False


class AthleteProfile(BaseModel):
    """Athlete profile information."""
    name: str
    age: int
    weight: float
    height: float
    gender: str


class HealthMetrics(BaseModel):
    """Health metrics data."""
    heart_rate: Optional[float] = None
    hrv_rmssd: Optional[float] = None
    stress_percent: Optional[float] = None
    avg_stress: Optional[float] = None
    stress: Optional[str] = None
    vo2_max: Optional[str] = None
    total_active_energy_expenditure: Optional[float] = None
    injury_risk: Optional[str] = None
    current_trimp: Optional[float] = None
    total_trimp: Optional[float] = None
    hr_rest: Optional[float] = None
    hr_max: Optional[float] = None
    g_impact_count: Optional[int] = None


class PredictionResponse(BaseModel):
    """Real-time prediction response."""
    timestamp: str
    device_id: str
    athlete_id: str
    athlete_profile: AthleteProfile
    velocity: float
    distance: float
    heart_rate: float
    heart_rate_source: str
    mode: str
    health_metrics: HealthMetrics
    g_impact: float
    g_impact_count: int
    g_impact_events: List[Dict[str, Any]]
    model_prediction: Optional[Dict[str, Any]] = None
    data_quality: Optional[Dict[str, Any]] = None


class PlayerInfo(BaseModel):
    """Player information."""
    player_id: str
    device_id: str
    status: str
    last_update: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    latest_metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow extra fields


class PlayerListResponse(BaseModel):
    """List of players response."""
    players: List[PlayerInfo]
    total: int


class TrainingStatus(BaseModel):
    """Training status information."""
    is_running: bool
    status: str
    message: Optional[str] = None


class SystemStatus(BaseModel):
    """System status information."""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    prediction_running: bool
    training_running: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionSummary(BaseModel):
    """Session summary data."""
    session_duration_minutes: float
    total_trimp: float
    trimp_zone: str
    zone_description: str
    avg_stress: float
    g_impact_count: int
    peak_heart_rate: Optional[float] = None
    avg_heart_rate: Optional[float] = None
    training_recommendations: Optional[List[str]] = None

