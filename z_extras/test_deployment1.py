import paho.mqtt.client as mqtt
import json
import time
import joblib
import numpy as np
from collections import deque
import warnings
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2R
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq
from datetime import datetime
from database.db import db
import os
import logging
warnings.filterwarnings("ignore")
import psutil
import yaml
import torch
from hummingbird.ml import convert
import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()

with open('config/jetson_orin_32gb_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logs_dir = config['logging']['logs_dir']

# --- Multi-device state ---
# Keep per-device processing context so we can handle multiple devices concurrently
device_contexts = {}

def _init_device_context(device_id_str):
    """Create and return a fresh processing context for a device."""
    try:
        device_id_int = int(device_id_str)
    except Exception:
        device_id_int = None

    # Try to resolve athlete profile from DB
    athlete = None
    if device_id_int is not None:
        try:
            rows = db.query("SELECT * FROM players WHERE id = %s", (device_id_int,))
            if rows:
                athlete = rows[0]
        except Exception:
            pass

    if athlete is None:
        try:
            rows = db.query("SELECT * FROM players WHERE device_id = %s", (device_id_str,))
            if rows:
                athlete = rows[0]
        except Exception:
            pass

    if athlete:
        athlete_id_val = athlete.get('id', device_id_int if device_id_int is not None else 0)
        name_val = athlete.get('name', f"Device_{device_id_str}")
        age_val = int(athlete.get('age', 25))
        weight_val = float(athlete.get('weight', 70))
        height_val = float(athlete.get('height', 175))
        gender_val = 1 if athlete.get('gender', 'M') == 'M' else 0
    else:
        athlete_id_val = device_id_int if device_id_int is not None else 0
        name_val = f"Device_{device_id_str}"
        age_val = 25
        weight_val = 70.0
        height_val = 175.0
        gender_val = 1

    context = {
        "device_id": device_id_str,
        "athlete_id": athlete_id_val,
        "name": name_val,
        "age": age_val,
        "weight": weight_val,
        "height": height_val,
        "gender": gender_val,
        "hr_rest": 60,
        "hr_max": 220 - age_val,
        "MQTT_PUBLISH_TOPIC": f"{device_id_str}/predictions",
        # Filters and state
        "madgwick_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "hr_buffer": deque(maxlen=N),
        "acc_buffer": deque(maxlen=N),
        "gyro_buffer": deque(maxlen=N),
        "acc_mag_buffer": deque(maxlen=30),
        "vel_buffer": deque([0.0], maxlen=1),
        "dist_buffer": deque([0.0], maxlen=1),
        "stress_buffer": [],
        "TEE_buffer": [],
        "g_impact_events": [],
        "g_impact_count": 0,
        "acc_mag_history": deque(maxlen=5),
        "gyro_mag_history": deque(maxlen=5),
        # Timing/session
        "session_start_time": None,
        "last_vo2_update_time": None,
        "last_data_time": time.time(),
        "last_warning_time": 0,
        "vo2_max_value": "-",
        "session_end_time": None,
        "session_ended": False,
        # Vitals
        "fitness_level": 100.0,
        "hydration_level": 100.0,
        # TRIMP
        "trimp_buffer": [],
        "total_trimp": 0.0,
    }

    device_contexts[device_id_str] = context
    return context

def _load_context_to_globals(ctx):
    """Populate module-level globals from context for reuse in existing code paths."""
    global device_id, athlete_id, name, age, weight, height, gender, hr_rest, hr_max
    global quaternion, hr_buffer, acc_buffer, gyro_buffer, acc_mag_buffer, vel_buffer, dist_buffer
    global stress_buffer, TEE_buffer, g_impact_events, g_impact_count, acc_mag_history, gyro_mag_history
    global session_start_time, last_vo2_update_time, last_data_time, last_warning_time, vo2_max_value
    global session_end_time, session_ended, fitness_level, hydration_level, trimp_buffer, total_trimp
    global MQTT_PUBLISH_TOPIC

    device_id = ctx["device_id"]
    athlete_id = ctx["athlete_id"]
    name = ctx["name"]
    age = ctx["age"]
    weight = ctx["weight"]
    height = ctx["height"]
    gender = ctx["gender"]
    hr_rest = ctx["hr_rest"]
    hr_max = ctx["hr_max"]
    MQTT_PUBLISH_TOPIC = ctx["MQTT_PUBLISH_TOPIC"]

    quaternion = ctx["madgwick_quaternion"]
    hr_buffer = ctx["hr_buffer"]
    acc_buffer = ctx["acc_buffer"]
    gyro_buffer = ctx["gyro_buffer"]
    acc_mag_buffer = ctx["acc_mag_buffer"]
    vel_buffer = ctx["vel_buffer"]
    dist_buffer = ctx["dist_buffer"]
    stress_buffer = ctx["stress_buffer"]
    TEE_buffer = ctx["TEE_buffer"]
    g_impact_events = ctx["g_impact_events"]
    g_impact_count = ctx["g_impact_count"]
    acc_mag_history = ctx["acc_mag_history"]
    gyro_mag_history = ctx["gyro_mag_history"]

    session_start_time = ctx["session_start_time"]
    last_vo2_update_time = ctx["last_vo2_update_time"]
    last_data_time = ctx["last_data_time"]
    last_warning_time = ctx["last_warning_time"]
    vo2_max_value = ctx["vo2_max_value"]
    session_end_time = ctx["session_end_time"]
    session_ended = ctx["session_ended"]
    fitness_level = ctx["fitness_level"]
    hydration_level = ctx["hydration_level"]
    trimp_buffer = ctx["trimp_buffer"]
    total_trimp = ctx["total_trimp"]

def _save_globals_to_context(ctx):
    """Persist module-level globals back into the device context after processing."""
    ctx["hr_rest"] = hr_rest
    ctx["hr_max"] = hr_max
    ctx["MQTT_PUBLISH_TOPIC"] = MQTT_PUBLISH_TOPIC

    ctx["madgwick_quaternion"] = quaternion
    ctx["hr_buffer"] = hr_buffer
    ctx["acc_buffer"] = acc_buffer
    ctx["gyro_buffer"] = gyro_buffer
    ctx["acc_mag_buffer"] = acc_mag_buffer
    ctx["vel_buffer"] = vel_buffer
    ctx["dist_buffer"] = dist_buffer
    ctx["stress_buffer"] = stress_buffer
    ctx["TEE_buffer"] = TEE_buffer
    ctx["g_impact_events"] = g_impact_events
    ctx["g_impact_count"] = g_impact_count
    ctx["acc_mag_history"] = acc_mag_history
    ctx["gyro_mag_history"] = gyro_mag_history

    ctx["session_start_time"] = session_start_time
    ctx["last_vo2_update_time"] = last_vo2_update_time
    ctx["last_data_time"] = last_data_time
    ctx["last_warning_time"] = last_warning_time
    ctx["vo2_max_value"] = vo2_max_value
    ctx["session_end_time"] = session_end_time
    ctx["session_ended"] = session_ended
    ctx["fitness_level"] = fitness_level
    ctx["hydration_level"] = hydration_level
    ctx["trimp_buffer"] = trimp_buffer
    ctx["total_trimp"] = total_trimp

# Setup logging
def setup_logging(athlete_id, device_id):
    """Setup comprehensive logging for the session"""
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    log_filename = f"{logs_dir}/A{athlete_id}_D{device_id}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== SESSION STARTED ===")
    logger.info(f"Athlete ID: {athlete_id}")
    logger.info(f"Device ID: {device_id}")
    logger.info(f"Log file: {log_filename}")
    
    return logger

# Memory monitoring
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def print_memory_usage(label=""):
    """Print memory usage with optional label"""
    memory_mb = get_memory_usage()
    print(f"Memory usage{label}: {memory_mb:.2f} MB")
    
def print_detailed_memory_usage(label=""):
    """Print detailed memory usage with additional system info"""
    memory_mb = get_memory_usage()
    total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Total system RAM in MB
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Available RAM in MB
    memory_percent = (memory_mb / total_memory) * 100
    
    print(f"   MEMORY STATUS{label}:")
    print(f"   Process Memory: {memory_mb:.2f} MB ({memory_percent:.1f}% of system)")
    print(f"   System Total: {total_memory:.0f} MB")
    print(f"   System Available: {available_memory:.0f} MB")
    print(f"   System Usage: {100 - (available_memory/total_memory)*100:.1f}%")

def create_prediction_lockfile():
    """Create a lockfile to indicate prediction is running."""
    try:
        with open('.prediction_running.lock', 'w') as f:
            f.write(str(os.getpid()))
        print(f"[INFO] Created prediction lockfile (PID: {os.getpid()})")
    except Exception as e:
        print(f"[WARN] Failed to create prediction lockfile: {e}")

def remove_prediction_lockfile():
    """Remove the prediction lockfile."""
    try:
        if os.path.exists('.prediction_running.lock'):
            os.remove('.prediction_running.lock')
            print("[INFO] Removed prediction lockfile")
    except Exception as e:
        print(f"[WARN] Failed to remove prediction lockfile: {e}")


# Initial memory usage
print_detailed_memory_usage(" (startup)")

# Broker config
MQTT_BROKER = "localhost" 
MQTT_PORT = 1883
idle_time = os.getenv("IDLE_TIME", 300)
# Load model
model = joblib.load("Athlete_RF_trained.pkl")
hb_model = convert(model, backend="pytorch")
if torch.cuda.is_available():
    hb_model.to("cuda")

# Initialize Madgwick filter
madgwick_filter = Madgwick()
quaternion = np.array([1.0, 0.0, 0.0, 0.0])

# Buffers
N = 30  # Rolling window size
hr_buffer = deque(maxlen=N)
acc_buffer = deque(maxlen=N)
gyro_buffer = deque(maxlen=N)
acc_mag_buffer = deque(maxlen=30)
vel_buffer = deque([0.0], maxlen=1)
dist_buffer = deque([0.0], maxlen=1)
stress_buffer = []
TEE_buffer = []
g_impact_events = []  # List to store (timestamp, g_impact)
g_impact_count = 0    # Counter for high-g impacts

# --- Add rolling history for sophisticated g-impact detection ---
acc_mag_history = deque(maxlen=5)
gyro_mag_history = deque(maxlen=5)

# Time tracking
session_start_time = None
last_vo2_update_time = None
last_data_time = time.time()
last_warning_time = 0  # Track when last warning was printed
vo2_max_value = "-"
session_end_time = None
session_ended = False

dt = 0.1  # time step
FS_HZ = 10.0
WINDOW_SECONDS = 3  # 3–5 s recommended; start with 3 s for responsiveness
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS_HZ)

# TRIMP variables (will be initialized after athlete profile is loaded)
trimp_buffer = []
total_trimp = 0.0
hr_rest = 60  # actual resting HR from database
hr_max = None  # Will be calculated after age is loaded

# Constants
s_a1, s_a2, s_a4, s_a5 = 0.4, 0.2, 0.2, 0.3
s_b = 10
s_epsilon = 1e-6

# Athlete profile
athlete_id = int(input("Enter an athlete id:")) # or any logic to determine which user to load
rows = db.query("SELECT * FROM players WHERE id = %s", (athlete_id,))

if rows:
    athlete = rows[0]
    name = athlete['name']
    age = int(athlete['age'])
    weight = float(athlete['weight'])
    height = float(athlete['height'])
    gender = 1 if athlete['gender'] == 'M' else 0
    device_id = athlete.get('device_id') or athlete.get('id')  # Try both device_id and id
    if not device_id:
        raise ValueError("No device is assigned to the player with the entered id.")
    # Calculate hr_max after age is loaded
    hr_max = 220 - age  # Estimated max heart rate using age formula
else:
    raise ValueError("Athlete not found")

Athlete_profile = {"Device_id": device_id, "Name": name, "Age": age, "Weight": weight, "Height": height, "Gender": gender}
#print(Athlete_profile)

# Setup logging after athlete profile is loaded
logger = setup_logging(athlete_id, device_id)
logger.info(f"Athlete Profile: {Athlete_profile}")

MQTT_PUBLISH_TOPIC = f"{device_id}/predictions" 
# Store latest sensor readings
sensor_data = {
    "acc": None,
    "gyro": None,
    "magno": None
}

# Initialize fitness, hydration
fitness_level = 100.0
hydration_level = 100.0

# Add position tracking
latest_position = {"x": None, "y": None}

# --- Window buffers for feature engineering (per-axis, 3 s @ 10 Hz) ---
acc_x_win = deque(maxlen=WINDOW_SAMPLES)
acc_y_win = deque(maxlen=WINDOW_SAMPLES)
acc_z_win = deque(maxlen=WINDOW_SAMPLES)
gyro_x_win = deque(maxlen=WINDOW_SAMPLES)
gyro_y_win = deque(maxlen=WINDOW_SAMPLES)
gyro_z_win = deque(maxlen=WINDOW_SAMPLES)

def compute_window_features(ax, ay, az, gx, gy, gz, fs=FS_HZ):
    try:
        ax_arr = np.asarray(ax, dtype=np.float32)
        ay_arr = np.asarray(ay, dtype=np.float32)
        az_arr = np.asarray(az, dtype=np.float32)
        gx_arr = np.asarray(gx, dtype=np.float32)
        gy_arr = np.asarray(gy, dtype=np.float32)
        gz_arr = np.asarray(gz, dtype=np.float32)
        if min(len(ax_arr), len(ay_arr), len(az_arr), len(gx_arr), len(gy_arr), len(gz_arr)) < WINDOW_SAMPLES:
            return None

        # Basic stats per channel
        def stats(arr):
            return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)), float(np.max(arr) - np.min(arr))

        ax_mean, ax_std, ax_min, ax_max, ax_range = stats(ax_arr)
        ay_mean, ay_std, ay_min, ay_max, ay_range = stats(ay_arr)
        az_mean, az_std, az_min, az_max, az_range = stats(az_arr)
        gx_mean, gx_std, gx_min, gx_max, gx_range = stats(gx_arr)
        gy_mean, gy_std, gy_min, gy_max, gy_range = stats(gy_arr)
        gz_mean, gz_std, gz_min, gz_max, gz_range = stats(gz_arr)

        # Resultant acceleration time series
        a_res = np.sqrt(ax_arr**2 + ay_arr**2 + az_arr**2)
        a_res_mean = float(np.mean(a_res))
        a_res_std = float(np.std(a_res))
        a_res_cov = float(a_res_std / (a_res_mean + 1e-6))

        # Mean absolute jerk (finite difference of resultant accel)
        jerk = np.diff(a_res) * fs
        mean_abs_jerk = float(np.mean(np.abs(jerk))) if jerk.size > 0 else 0.0

        # Player load index = sum |Δaccel|
        player_load = float(np.sum(np.abs(np.diff(a_res))))

        # Frequency-domain features (0.5–5 Hz band)
        spectrum = rfft(a_res - np.mean(a_res))
        freqs = rfftfreq(a_res.shape[0], d=1.0/fs)
        power = np.abs(spectrum)**2
        band_mask = (freqs >= 0.5) & (freqs <= 5.0)
        spectral_energy = float(np.sum(power[band_mask]))
        if np.any(power > 0):
            peak_idx = int(np.argmax(power))
            peak_freq_hz = float(freqs[peak_idx])
        else:
            peak_freq_hz = 0.0

        return {
            "ax_mean": ax_mean, "ax_std": ax_std, "ax_min": ax_min, "ax_max": ax_max, "ax_range": ax_range,
            "ay_mean": ay_mean, "ay_std": ay_std, "ay_min": ay_min, "ay_max": ay_max, "ay_range": ay_range,
            "az_mean": az_mean, "az_std": az_std, "az_min": az_min, "az_max": az_max, "az_range": az_range,
            "gx_mean": gx_mean, "gx_std": gx_std, "gx_min": gx_min, "gx_max": gx_max, "gx_range": gx_range,
            "gy_mean": gy_mean, "gy_std": gy_std, "gy_min": gy_min, "gy_max": gy_max, "gy_range": gy_range,
            "gz_mean": gz_mean, "gz_std": gz_std, "gz_min": gz_min, "gz_max": gz_max, "gz_range": gz_range,
            "a_res_mean": a_res_mean, "a_res_std": a_res_std, "a_res_cov": a_res_cov,
            "mean_abs_jerk": mean_abs_jerk,
            "spectral_energy_0p5_5hz": spectral_energy,
            "peak_frequency_hz": peak_freq_hz,
            "player_load_index": player_load
        }
    except Exception:
        return None

# === Butterworth Lowpass Filter ===
def butter_bandpass_filter(data, low_cutoff=0.3, high_cutoff=4.5, fs=10.0, order=2):
    nyq = 0.5 * fs
    low = low_cutoff / nyq
    high = high_cutoff/ nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def calculate_rmssd(hr_values):
    rr_intervals = 60000 / np.array(hr_values)
    rr_diffs = np.diff(rr_intervals)
    return round(np.sqrt(np.mean(rr_diffs ** 2)), 2)

def estimate_vo2_max(age, gender, current_hr, hrv):
    hr_mod = 220 - current_hr
    vo2 = 60.0 - (0.55 * age) + (0.2 * hrv) + (0.3 * hr_mod) + (5 * gender)
    return round(vo2, 2)

def training_energy_expenditure(velocity, duration_s, mass_kg):
    velocity_kmph = velocity * 3.6
    # (lower_bound_velocity, upper_bound_velocity, MET, comment)
    met_table = [
        (0, 3.21, 2.3, "very light walking"),
        (3.21, 4.18, 2.8, "light walking"),
        (4.18, 5.95, 3.3, "jogging"),
        (6.4, 6.8, 6.5, "running 4.0-4.2 mph"), # 4.0 mph = 6.4 km/h, 4.2 mph = 6.8 km/h
        (6.8, 7.7, 7.8, "running 4.3-4.8 mph"), # 4.3 mph = 6.9 km/h, 4.8 mph = 7.7 km/h
        (8.0, 8.4, 8.5, "running 5.0-5.2 mph"), # 5.0 mph = 8.0 km/h, 5.2 mph = 8.4 km/h
        (8.9, 9.3, 9.0, "running 5.5-5.8 mph"), # 5.5 mph = 8.9 km/h, 5.8 mph = 9.3 km/h
        (9.7, 10.1, 9.3, "running 6.0-6.3 mph"), # 6.0 mph = 9.7 km/h, 6.3 mph = 10.1 km/h
        (10.7, 11.3, 10.5, "running 6.7 mph"), # 6.7 mph = 10.7 km/h, 11.3 mph = 18.1 km/h
        (11.3, 12.1, 11.0, "running 7.0 mph"), # 7.0 mph = 11.3 km/h, 12.1 mph = 19.5 km/h
        (12.1, 12.9, 11.8, "running 7.5 mph"), # 7.5 mph = 12.1 km/h, 12.9 mph = 20.8 km/h
        (12.9, 13.7, 12.0, "running 8.0 mph"), # 8.0 mph = 12.9 km/h, 13.7 mph = 22.1 km/h
        (13.7, 14.5, 12.5, "running 8.6 mph"), # 8.6 mph = 13.7 km/h, 14.5 mph = 23.3 km/h
        (14.5, 15.4, 13.0, "running 9.0 mph"), # 9.0 mph = 14.5 km/h, 15.4 mph = 24.8 km/h
        (15.0, 15.4, 14.8, "running 9.3-10.0 mph"), # 9.3 mph = 15.0 km/h, 10.0 mph = 15.4 km/h
        # Add more as needed
    ]
    met = 14.8  # Default to highest if above all ranges
    for lower, upper, met_value, _ in met_table:
        if lower <= velocity_kmph < upper:
            met = met_value
            break

    bmr = 500 + 22 * mass_kg # kcal/day
    bmr_min = bmr / 1440 # kcal/min
    duration_min = duration_s / 60.0
    calories = met * bmr_min * duration_min
    return round(calories, 2)

def parse_sensor_payload(payload):
    try:
        return json.loads(payload.decode())
    except json.JSONDecodeError:
        try:
            parts = payload.decode().strip().split("/")
            if len(parts) == 3:
                return {
                    "x": float(parts[0].strip()),
                    "y": float(parts[1].strip()),
                    "z": float(parts[2].strip())
                }
        except Exception:
            return None
    return None

def calculate_stress(hr, hrv, acc_mag, gyro_mag, age, gender, hr_rest=60, hr_max=200):
    # Normalize features
    hrr = (hr - hr_rest) / (hr_max - hr_rest + 1e-6)
    hrv_norm = 1 - (hrv / 100)  # Lower HRV = higher stress
    acc_norm = min(acc_mag / 10, 1.0)  # scale to [0,1]
    gyro_norm = min(gyro_mag / 10, 1.0)
    # Weighted sum (tune weights as needed)
    score = (
        0.4 * hrr +
        0.2 * hrv_norm +
        0.2 * acc_norm +
        0.1 * gyro_norm +
        0.1 * (1 if gender == 0 else 0)  # e.g., add a small gender effect
    )
    # Nonlinear mapping (sigmoid)
    stress_percent = 100 * (1 / (1 + np.exp(-8 * (score - 0.5))))
    return round(stress_percent, 1)

def calculate_trimp(hr_avg, hr_rest, hr_max, duration_min, gender="male"):
    #Calculate TRIMP (Training Impulse) for a single session.
    # Heart rate reserve ratio
    HRr = (hr_avg - hr_rest) / (hr_max - hr_rest)
    # Weighting factor based on gender
    if gender.lower() == "male":
        y = 0.64 * np.exp(1.92 * HRr)
    elif gender.lower() == "female":
        y = 0.86 * np.exp(1.67 * HRr)
    else:
        raise ValueError("Gender must be 'male' or 'female'.")
    
    # TRIMP calculation
    trimp = duration_min * HRr * y
    return trimp

def get_trimp_zone(total_trimp):
    """Determine TRIMP training zone based on total TRIMP value"""
    if total_trimp < 50:
        return "Light", "Recovery/Warm-up intensity"
    elif total_trimp < 150:
        return "Moderate", "Standard training session"
    elif total_trimp < 300:
        return "High", "Intense training session"
    else:
        return "Very High", "Very intense session"

def get_recovery_recommendations(total_trimp, stress_percent, fitness_level):
    """Generate recovery recommendations based on TRIMP and other metrics"""
    recommendations = []
    
    # Recovery time based on TRIMP
    if total_trimp < 50:
        recovery_time = "0-1 days"
        recommendations.append("Light session - can train again tomorrow")
    elif total_trimp < 150:
        recovery_time = "1-2 days"
        recommendations.append("Moderate session - rest 1-2 days before next intense session")
    elif total_trimp < 300:
        recovery_time = "2-3 days"
        recommendations.append("High intensity - plan 2-3 days recovery")
    else:
        recovery_time = "3+ days"
        recommendations.append("Very high intensity - extended recovery needed (3+ days)")
    
    # Additional recommendations based on stress and fitness
    if stress_percent > 70:
        recommendations.append("High stress detected - consider stress management techniques")
    
    if fitness_level < 50:
        recommendations.append("Low fitness level - focus on gradual progression")
    
    if total_trimp > 200 and stress_percent > 60:
        recommendations.append("High load + stress - monitor for overtraining signs")
    
    return recovery_time, recommendations

def get_training_recommendations(trimp_zone, stress_percent):
    """Generate detailed training recommendations based on TRIMP zone and current metrics"""
    recommendations = []
    
    # Add zone-specific recommendations
    if trimp_zone == "Very High":
        recommendations.extend([
            "High intensity session - ensure adequate recovery",
            "Monitor for signs of overtraining",
            "Consider extended recovery period"
        ])
    elif trimp_zone == "High":
        recommendations.extend([
            "Intense training session - plan recovery accordingly",
            "Monitor stress and fatigue levels",
            "Maintain proper nutrition and hydration"
        ])
    elif trimp_zone == "Moderate":
        recommendations.extend([
            "Standard training session - continue current approach",
            "Monitor for signs of progression",
            "Maintain consistent recovery practices"
        ])
    elif trimp_zone == "Light":
        recommendations.extend([
            "Recovery session - good for active rest",
            "Focus on technique and form",
            "Use for active recovery or warm-up"
        ])
    
    # Add stress-based recommendations
    if stress_percent > 70:
        recommendations.append("High stress levels - consider stress management techniques")
    elif stress_percent > 50:
        recommendations.append("Moderate stress - monitor stress levels")
    
    return recommendations

def generate_session_summary():
    """Generate end-of-session summary with recommendations"""
    global session_end_time, total_trimp, stress_buffer, fitness_level, hydration_level, g_impact_count
    
    if session_end_time is None:
        return None
    
    session_duration = (session_end_time - session_start_time) / 60.0  # in minutes
    avg_stress = round(sum(stress_buffer) / len(stress_buffer), 2) if stress_buffer else 0
    
    # Calculate final TRIMP zone and recommendations
    trimp_zone, zone_description = get_trimp_zone(total_trimp)
    recovery_time, recovery_recommendations = get_recovery_recommendations(total_trimp, avg_stress, fitness_level)
    training_recommendations = get_training_recommendations(trimp_zone, avg_stress)
    
    summary = {
        "session_end_time": datetime.fromtimestamp(session_end_time).isoformat(),
        "session_duration_minutes": round(session_duration, 2),
        "total_trimp": total_trimp,
        "trimp_zone": trimp_zone,
        "zone_description": zone_description,
        "avg_stress": avg_stress,
        "final_fitness_level": fitness_level,
        "final_hydration_level": hydration_level,
        "g_impact_count": g_impact_count,
        "recovery_time": recovery_time,
        "recovery_recommendations": recovery_recommendations,
        "training_recommendations": training_recommendations
    }
    
    return summary


def process_data():
    global session_start_time, last_vo2_update_time, vo2_max_value
    global fitness_level, hydration_level
    global quaternion
    global g_impact_count
    global hrv_rmssd
    global trimp_buffer, total_trimp, current_trimp
    global sensor_data, session_end_time, session_ended

    acc = sensor_data["acc"]
    gyro = sensor_data["gyro"]
    magno = sensor_data['magno']
    if not (acc and gyro and magno):
        return

    acc_x = acc["x"] * 9.81
    acc_y = acc["y"] * 9.81
    acc_z = acc["z"] * 9.81
    acc_raw = np.array([acc_x, acc_y, acc_z])
    
    gyro_x, gyro_y, gyro_z = gyro["x"], gyro["y"], gyro["z"]
    gyro_raw = np.array([gyro["x"], gyro["y"], gyro["z"]])
    mag_raw = np.array([magno["x"], magno["y"], magno["z"]])

    quaternion = madgwick_filter.updateMARG(q=quaternion, gyr=gyro_raw, acc=acc_raw, mag=mag_raw)
    rotation_matrix = q2R(quaternion)
    
    gravity = rotation_matrix.T @ np.array([0.0, 0.0, 9.81])
    acc_linear = acc_raw - gravity
    # When sensor is stationary on a flat surface: acc_linear ≈ [0, 0, 0]
    acc_x_l, acc_y_l, acc_z_l = acc_linear
    acc_mag = np.linalg.norm(acc_linear)
    acc_mag_buffer.append(acc_mag)
    # --- Add to rolling history ---
    acc_mag_history.append(acc_mag)
    gyro_mag = np.linalg.norm(gyro_raw)
    gyro_mag_history.append(gyro_mag)
    # Compute jerk (rate of change of acc_mag)
    if len(acc_mag_history) >= 2:
        jerk = abs(acc_mag_history[-1] - acc_mag_history[-2]) / dt
    else:
        jerk = 0
    
    if len(acc_mag_buffer) >= 30:
        acc_filtered = butter_bandpass_filter(list(acc_mag_buffer))
        if acc_filtered is not None:
            acc_buffer.append(acc_filtered[-1])

    if len(acc_buffer) == N:
        prev_acc, curr_acc = abs(acc_buffer[0]), abs(acc_buffer[-1])
        prev_vel = vel_buffer[-1]
        new_vel = 0.5 * (prev_acc + curr_acc) * dt
        
        is_resting = all(a < 0.75 for a in list(acc_buffer)[-2:])

        if is_resting:
            new_vel = 0.0
            vel_buffer.clear()
            vel_buffer.append(0.0)
            new_dist = dist_buffer[-1] + 0.0
            dist_buffer.append(new_dist)
        else:
            vel_buffer.append(new_vel)
            prev_dist = dist_buffer[-1]
            new_dist = prev_dist + 0.5 * new_vel * dt
            dist_buffer.append(new_dist)
    else:
        new_vel, new_dist = 0.0, 0.0

    # --- Windowed feature engineering (does not change model inputs yet) ---
    acc_x_win.append(float(acc_x))
    acc_y_win.append(float(acc_y))
    acc_z_win.append(float(acc_z))
    gyro_x_win.append(float(gyro_x))
    gyro_y_win.append(float(gyro_y))
    gyro_z_win.append(float(gyro_z))
    window_features = None
    if len(acc_x_win) == WINDOW_SAMPLES:
        window_features = compute_window_features(acc_x_win, acc_y_win, acc_z_win, gyro_x_win, gyro_y_win, gyro_z_win, fs=FS_HZ)

    # Check if we're in training mode (use actual HR) or game mode (predict HR)
    mode = sensor_data.get("mode", "game")  # Default to game mode if not specified
    
    if mode == "training":
        # In training mode, use actual heart rate from sensors
        actual_hr = sensor_data.get("heart_rate_bpm")
        if actual_hr is not None:
            predicted_hr = round(float(actual_hr), 0)
            print(f"🏃 Training Mode: Using actual HR from sensors: {predicted_hr} bpm")
        else:
            # Fallback to prediction if no actual HR available
            print("⚠️  Training mode but no actual HR found, falling back to prediction")
            mode = "game"  # Switch to prediction mode
    
    if mode == "game":
        # In game mode, predict heart rate using ML models
        features = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, age, weight, height, gender]
        x = np.asarray([features], dtype=np.float32)
        predicted_hr = float(hb_model.predict(x)[0])
        predicted_hr = round(predicted_hr, 0)
        print(f"⚽ Game Mode: Predicted HR using ML model: {predicted_hr} bpm")
    
    hr_buffer.append(predicted_hr)

    print(f"Velocity: {round(new_vel, 2)} m/s | Distance: {round(new_dist, 2)} m")
    if mode == "training":
        print(f"Actual HR: {predicted_hr} bpm (from sensors)")
    else:
        print(f"Predicted HR: {predicted_hr} bpm (ML model)")

    # Robust stress calculation using rolling means
    if len(hr_buffer) == N:
        hrv_rmssd = calculate_rmssd(hr_buffer)
        acc_mean = np.mean(acc_buffer)
        gyro_mean = np.mean(gyro_buffer)
        hr_mean = np.mean(hr_buffer)
        stress_percent = calculate_stress(hr_mean, hrv_rmssd, acc_mean, gyro_mean, age, gender)
        stress_buffer.append(stress_percent)
        avg_stress = round(sum(stress_buffer) / len(stress_buffer), 2)
        stress_label = (
            "Low" if avg_stress < 40 else
            "Moderate" if avg_stress < 70 else
            "High"
        )
        print(f"Stress: {stress_percent}% (avg: {avg_stress}%) - {stress_label}")
    else:
        hrv_rmssd = 0
        stress_percent = 0
        avg_stress = 0
        stress_label = "-"

    now = time.time()
    if session_start_time is None:
        session_start_time = now
        logger.info(f"Session started at {datetime.fromtimestamp(session_start_time).isoformat()}")

    elapsed_time = now - session_start_time

    if elapsed_time >= 300:
        if last_vo2_update_time is None or (now - last_vo2_update_time >= 300):
            vo2_max_value = estimate_vo2_max(age, gender, np.mean(hr_buffer), hrv_rmssd)
            last_vo2_update_time = now
            logger.info(f"VO2 Max updated: {vo2_max_value}")
    else:
        vo2_max_value = "-"

    # --- Energy, Fitness, Hydration decay ---
    fitness_decay_rate = 0.0005
    hydration_decay_rate = 0.0005

    activity_intensity = abs(new_vel)
    stress_factor = stress_percent / 100

    fitness_loss = fitness_decay_rate * (1 + stress_factor)
    hydration_loss = hydration_decay_rate * (1 + stress_factor + activity_intensity * 0.3)

    fitness_level = max(0, round(fitness_level - fitness_loss, 2))
    hydration_level = max(0, round(hydration_level - hydration_loss, 2))

    # --- Total Energy Expenditure Calculation ---
    tte = training_energy_expenditure(new_vel, dt, weight)
    TEE_buffer.append(tte)
    active_tee = round(sum(TEE_buffer), 2)

    print(f"TEE: {active_tee}kcal, Fitness: {fitness_level}%, Hydration: {hydration_level}%")
    print(f"Training Energy Expenditure: {tte} kcal | Total: {active_tee} kcal")

    # --- TRIMP Calculation ---
    if len(hr_buffer) >= 5:  # Need at least 5 HR readings for meaningful TRIMP
        hr_avg = np.mean(list(hr_buffer)[-5:])  # Use last 5 HR readings
        duration_min = elapsed_time / 60.0  # Convert to minutes
        gender_str = "male" if gender == 1 else "female"
        
        current_trimp = calculate_trimp(hr_avg, hr_rest, hr_max, duration_min, gender_str)
        trimp_buffer.append(current_trimp)
        total_trimp = round(sum(trimp_buffer), 2)
        
        print(f"TRIMP: {round(current_trimp, 2)} | Total TRIMP: {total_trimp}")
    else:
        current_trimp = 0
        total_trimp = 0

    G_IMPACT_ACC_THRESHOLD = 8 * 9.81  # 8g threshold
    G_IMPACT_GYRO_THRESHOLD = 300      # deg/s, adjust as needed
    G_IMPACT_JERK_THRESHOLD = 100      # m/s^3, adjust as needed

    # Detect high-g impact: require all three criteria
    injury_risk = False
    if (
        acc_mag > G_IMPACT_ACC_THRESHOLD and
        gyro_mag > G_IMPACT_GYRO_THRESHOLD and
        jerk > G_IMPACT_JERK_THRESHOLD
    ):
        injury_risk = True
        g_impact_count += 1
        event_time = datetime.fromtimestamp(time.time()).isoformat()
        # Determine axis of max acceleration
        acc_axes = np.array([abs(acc_x), abs(acc_y), abs(acc_z)])
        max_axis = ['x', 'y', 'z'][np.argmax(acc_axes)]
        g_impact_events.append({
            "time": event_time,
            "g_impact": round(acc_mag, 2),
            "x": latest_position["x"],
            "y": latest_position["y"],
            "device_id": device_id,
            "athlete_id": athlete_id,
            "name": name,
            "gyro_mag": round(gyro_mag, 2),
            "jerk": round(jerk, 2),
            "max_axis": max_axis
        })
        impact_msg = f"⚠️ High G-Impact detected! ({acc_mag:.2f} m/s², gyro: {gyro_mag:.2f} deg/s, jerk: {jerk:.2f} m/s³) at {event_time} - Possible injury risk. Position: ({latest_position['x']}, {latest_position['y']}) Axis: {max_axis}"
        print(impact_msg)
        logger.warning(impact_msg)
        # When saving g-impact log, use a per-player folder:
        player_folder = str(device_id)
        os.makedirs(player_folder, exist_ok=True)
        with open(os.path.join(player_folder, f"{athlete_id}_g_impact_log.json"), "w") as f:
            json.dump(g_impact_events, f, indent=2)

    output = {
        "timestamp": datetime.fromtimestamp(time.time()).isoformat(),
        "velocity": round(new_vel, 2),
        "distance": round(new_dist, 2),
        "heart_rate": predicted_hr,
        "heart_rate_source": "actual_sensor" if mode == "training" else "ml_prediction",
        "mode": mode,
        "hrv_rmssd": hrv_rmssd,
        "stress_percent": stress_percent,
        "avg_stress": avg_stress,
        "stress": stress_label,
        "vo2_max": vo2_max_value,
        "Total_active_energy_expenditure": active_tee,
        "fitness_level": fitness_level,
        "hydration_level": hydration_level,
        "injury_risk": injury_risk,
        "g_impact": round(acc_mag, 2),
        "g_impact_count": g_impact_count,
        "g_impact_events": g_impact_events[-10:],  # Last 10 events for quick view
        "current_trimp": round(current_trimp, 2),
        "total_trimp": total_trimp,
        "hr_rest": hr_rest,
        "hr_max": hr_max,
        # Include engineered features snapshot for this window
        "window_features": window_features if window_features is not None else {},
    }

    # When saving realtime output, use organized folder structure:
    base_output_dir = "prediction_outputs"
    player_folder = f"A{str(athlete_id)}_{str(name)}"
    full_player_path = os.path.join(base_output_dir, player_folder)
    os.makedirs(full_player_path, exist_ok=True)

    with open(os.path.join(full_player_path, f"A{athlete_id}_D{device_id}_realtime_output.json"), "w") as f:
        json.dump(output, f, indent=2)

    client.publish(MQTT_PUBLISH_TOPIC, json.dumps(output))
    print(f"📤 Published metrics to MQTT topic '{MQTT_PUBLISH_TOPIC}'")
    logger.debug(f"Published metrics to MQTT topic '{MQTT_PUBLISH_TOPIC}'")
    
    # Real-time memory monitoring (every 10 data points)
    if len(hr_buffer) % 10 == 0 and len(hr_buffer) > 0:
        print_memory_usage(" (real-time)")
        print("-" * 40)  # Separator for better visibility
        logger.debug("Memory usage check completed")

# --- MQTT callbacks ---
def on_connect(client, userdata, flags, rc):
    print("✅ Connected to MQTT Broker:", rc)
    logger.info(f"Connected to MQTT Broker with result code: {rc}")
    # Subscribe to all per-player topics and keep legacy topic for compatibility
    client.subscribe("player/+/sensor/data")
    client.subscribe("sensor/data")
    logger.info("Subscribed to player/+/sensor/data and sensor/data topics")

def on_disconnect(client, userdata, rc):
    print("⚠️  Disconnected. Attempting to reconnect...")
    logger.warning(f"Disconnected from MQTT Broker with result code: {rc}")
    while True:
        try:
            client.reconnect()
            print("🔄 Reconnected.")
            logger.info("Successfully reconnected to MQTT Broker")
            break
        except:
            print("⏳ Retry in 5 seconds...")
            logger.warning("Reconnection attempt failed, retrying in 5 seconds...")
            time.sleep(5)

def on_message(client, userdata, msg):
    topic = msg.topic
    global sensor_data

    try:
        parsed_data = json.loads(msg.payload.decode())

        # Determine device id: prefer payload device_id, else parse from topic player/{id}/sensor/data
        device_id_str = str(parsed_data.get("device_id", "")).strip()
        if not device_id_str and topic.startswith("player/"):
            try:
                device_id_str = topic.split("/")[1]
            except Exception:
                device_id_str = ""
        if not device_id_str:
            # Unable to determine device id; skip
            return
        device_id_str = device_id_str.zfill(3)

        # Get or init per-device context and sync into globals
        ctx = device_contexts.get(device_id_str) or _init_device_context(device_id_str)
        _load_context_to_globals(ctx)
        
        # Update last data time for this device
        ctx["last_data_time"] = time.time()

        # Convert publisher.py format to the expected format
        sensor_data = {
            "acc": {
                "x": parsed_data.get("acc_x", 0),
                "y": parsed_data.get("acc_y", 0),
                "z": parsed_data.get("acc_z", 0)
            },
            "gyro": {
                "x": parsed_data.get("gyro_x", 0),
                "y": parsed_data.get("gyro_y", 0),
                "z": parsed_data.get("gyro_z", 0)
            },
            "magno": {
                "x": parsed_data.get("mag_x", 0),
                "y": parsed_data.get("mag_y", 0),
                "z": parsed_data.get("mag_z", 0)
            },
            # Include mode and heart rate data from publisher
            "mode": parsed_data.get("mode", "game"),
            "heart_rate_bpm": parsed_data.get("heart_rate_bpm")
        }

        # Process the data using globals mapped from this context
        process_data()

        # Persist updated globals back to context
        _save_globals_to_context(ctx)

    except Exception as e:
        error_msg = f"Error processing message: {e}"
        print(error_msg)
        logger.error(error_msg)

def check_session_end():
    """Per-device session end checks. Generate summaries without stopping the loop."""
    now_ts = time.time()
    for device_id_str, ctx in list(device_contexts.items()):
        if not ctx.get("session_ended") and (now_ts - ctx.get("last_data_time", 0)) > int(idle_time):
            # Load context into globals to reuse existing summary pipeline
            _load_context_to_globals(ctx)
            ctx["session_end_time"] = now_ts
            session_end_time_set = now_ts
            session_ended_set = True
            # Mirror into globals expected by generate_session_summary
            globals()["session_end_time"] = session_end_time_set
            globals()["session_ended"] = session_ended_set

            logger.info(f"Session end condition met for device {device_id_str} - no data for {idle_time} seconds")

            summary = generate_session_summary()
            if summary:
                print(f"\n📊 SESSION SUMMARY (device {device_id_str}):")
                print(f"Duration: {summary['session_duration_minutes']} minutes")
                print(f"Total TRIMP: {summary['total_trimp']}")
                print(f"TRIMP Zone: {summary['trimp_zone']} - {summary['zone_description']}")
                print(f"Average Stress: {summary['avg_stress']}%")
                print(f"Final Fitness Level: {summary['final_fitness_level']}%")
                print(f"Final Hydration Level: {summary['final_hydration_level']}%")
                print(f"G-Impact Events: {summary['g_impact_count']}")

                player_folder = f"A{str(athlete_id)}_{str(name)}"
                os.makedirs(player_folder, exist_ok=True)
                timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                summary_filename = f"A{athlete_id}_D{device_id}_session_summary_{timestamp}.json"
                summary_filepath = os.path.join(player_folder, summary_filename)
                with open(summary_filepath, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"\n Session summary saved to {player_folder}/{summary_filename}")

            # Mark ended and persist back to context
            globals()["session_ended"] = True
            _save_globals_to_context(ctx)

    # Never stops the loop; return False for compatibility
    return False

# --- MAIN ---
if __name__ == "__main__":
    # Create prediction lockfile to prevent training conflicts
    create_prediction_lockfile()
    
    # Setup cleanup handler
    def cleanup_handler(signum, frame):
        print("\n[INFO] Prediction interrupted. Cleaning up...")
        remove_prediction_lockfile()
        import sys
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    print(f"🚀 Starting test deployment in multi-device mode (initial athlete {athlete_id}, device {device_id})")
    print(f"📡 Subscribing to player/+/sensor/data and sensor/data topics")
    print(f"📤 Publishing to {MQTT_PUBLISH_TOPIC} topic")
    logger.info(f"Starting test deployment for athlete {athlete_id} (device {device_id})")
    logger.info(f"Subscribing to sensor/data topic")
    logger.info(f"Publishing to {MQTT_PUBLISH_TOPIC} topic")

    while True:
        client.loop(timeout=1.0)

        # Check for session end first
        if check_session_end():
            print("\n" + "="*60)
            print("🏁 SESSION ENDED - Generating Summary Report")
            print("="*60)
            logger.info("="*60)
            logger.info("SESSION ENDED - Generating Summary Report")
            logger.info("="*60)
            
            summary = generate_session_summary()
            if summary:
                print(f"\n📊 SESSION SUMMARY:")
                logger.info("📊 SESSION SUMMARY:")
                print(f"Duration: {summary['session_duration_minutes']} minutes")
                logger.info(f"Duration: {summary['session_duration_minutes']} minutes")
                print(f"Total TRIMP: {summary['total_trimp']}")
                logger.info(f"Total TRIMP: {summary['total_trimp']}")
                print(f"TRIMP Zone: {summary['trimp_zone']} - {summary['zone_description']}")
                logger.info(f"TRIMP Zone: {summary['trimp_zone']} - {summary['zone_description']}")
                print(f"Average Stress: {summary['avg_stress']}%")
                logger.info(f"Average Stress: {summary['avg_stress']}%")
                print(f"Final Fitness Level: {summary['final_fitness_level']}%")
                logger.info(f"Final Fitness Level: {summary['final_fitness_level']}%")
                print(f"Final Hydration Level: {summary['final_hydration_level']}%")
                logger.info(f"Final Hydration Level: {summary['final_hydration_level']}%")
                print(f"G-Impact Events: {summary['g_impact_count']}")
                logger.info(f"G-Impact Events: {summary['g_impact_count']}")
                
                print(f"\n⏰ RECOVERY RECOMMENDATIONS:")
                logger.info("⏰ RECOVERY RECOMMENDATIONS:")
                print(f"Recovery Time: {summary['recovery_time']}")
                logger.info(f"Recovery Time: {summary['recovery_time']}")
                for i, rec in enumerate(summary['recovery_recommendations'], 1):
                    print(f"  {i}. {rec}")
                    logger.info(f"  {i}. {rec}")
                
                print(f"\n💪 TRAINING RECOMMENDATIONS:")
                logger.info("💪 TRAINING RECOMMENDATIONS:")
                for i, rec in enumerate(summary['training_recommendations'], 1):
                    print(f"  {i}. {rec}")
                    logger.info(f"  {i}. {rec}")
                
                # Save session summary with timestamp
                base_output_dir = "prediction_outputs"
                player_folder = f"A{str(athlete_id)}_{str(name)}"
                full_player_path = os.path.join(base_output_dir, player_folder)
                os.makedirs(full_player_path, exist_ok=True)
                
                # Generate timestamp for unique filename
                timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                summary_filename = f"A{athlete_id}_D{device_id}_session_summary_{timestamp}.json"
                summary_filepath = os.path.join(full_player_path, summary_filename)
                
                with open(summary_filepath, "w") as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\n Session summary saved to {player_folder}/{summary_filename}")
                logger.info(f" Session summary saved to {player_folder}/{summary_filename}")
                
                # Final memory usage
                print_detailed_memory_usage(" (session end)")
                print("="*60)
                logger.info("="*60)
                logger.info("=== SESSION COMPLETED ===")
                
                # Cleanup prediction lockfile before exiting
                remove_prediction_lockfile()
                break  # Exit the loop after generating summary
        
        # Check for 5-second warning (but don't reset timer)
        elif time.time() - last_data_time > 5:
            current_time = time.time()
            # Only print warning every 5 seconds, not every loop iteration
            if current_time - last_warning_time >= 5:
                print("⚠️  No sensor data received in the last 5 seconds.")
                print_memory_usage(" (waiting)")
                logger.warning("No sensor data received in the last 5 seconds")
                last_warning_time = current_time
    
# Initial memory usage
print_detailed_memory_usage(" (startup)")