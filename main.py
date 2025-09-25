import paho.mqtt.client as mqtt
import json
import time
import numpy as np
from collections import deque
import warnings
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2R
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq
from datetime import datetime
import os
import logging
warnings.filterwarnings("ignore")
import psutil
import yaml
import torch
from hummingbird.ml import convert, load
import numpy as np
import joblib
from dotenv import load_dotenv
from dynamic_model_loader import DynamicModelLoader

load_dotenv()

with open('jetson_nano_4gb_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logs_dir = config['logging']['logs_dir']

# --- Multi-device state ----
# Keep per-device processing context so we can handle multiple devices concurrently
device_contexts = {}
current_mode = "unknown"  # Track current mode (training/game)
prediction_logged_devices = set()  # Track which devices have had their first prediction logged

def _update_log_filename_if_needed(mode):
    """Create log file with mode when first detected."""
    global current_mode
    if current_mode == "unknown" and mode in ["training", "game"]:
        current_mode = mode
        # Create log filename with mode
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        log_filename = f"{logs_dir}/system_{mode}_{timestamp}.log"
        
        # Add file handler for mode-specific log (no need to remove since we start without one)
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Log all the startup information that was previously only in console
        logger.info(f"=== SYSTEM STARTUP ===")
        logger.info(f"CUDA not requested, using CPU processing")
        logger.info(f"Model Discovery: Found {len(discovered)} models in athlete_models_tensors_updated/")
        
        # Log discovered models dynamically
        for name, _ in discovered:
            logger.info(f"Model Loaded: {name}")
        
        # Log model registry creation dynamically
        logger.info(f"Model Registry: Created with {len(model_registry)} models (indices 1-{max_available_models})")
        logger.info(f"Model Mapping: Each device will use its corresponding model (Device 1 -> Model 1, Device 2 -> Model 2, etc.)")
        logger.info(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
        logger.info(f"Starting test deployment in multi-device mode")
        logger.info(f"MQTT subscription topics: player/+/sensor/data, sensor/data")
        logger.info(f"Connected to MQTT Broker with result code: 0")
        logger.info(f"Subscribed to topics: player/+/sensor/data, sensor/data")
        
        logger.info(f"Log file created with mode: {log_filename}")
        logger.info(f"Mode detected: {mode.upper()}")

def _init_device_context(device_id_str):
    """Create and return a fresh processing context for a device."""
    try:
        device_id_int = int(device_id_str)
    except Exception:
        device_id_int = None

    # Initialize athlete profile without DB; use sensible defaults
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
        "hr_buffer": deque(maxlen=ROLLING_WINDOW_SIZE),
        "acc_buffer": deque(maxlen=ROLLING_WINDOW_SIZE),
        "gyro_buffer": deque(maxlen=ROLLING_WINDOW_SIZE),
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
        # TRIMP
        "trimp_buffer": [],
        "total_trimp": 0.0,
    }

    device_contexts[device_id_str] = context

    # Ensure a logger exists; set up per first device
    global logger
    try:
        _ = logger  # type: ignore[name-defined]
    except NameError:
        logger = setup_logging(athlete_id_val, device_id_str)
    return context

def _load_context_to_globals(ctx):
    """Populate module-level globals from context for reuse in existing code paths."""
    global device_id, athlete_id, name, age, weight, height, gender, hr_rest, hr_max
    global quaternion, hr_buffer, acc_buffer, gyro_buffer, acc_mag_buffer, vel_buffer, dist_buffer
    global stress_buffer, TEE_buffer, g_impact_events, g_impact_count, acc_mag_history, gyro_mag_history
    global session_start_time, last_vo2_update_time, last_data_time, last_warning_time, vo2_max_value
    global session_end_time, session_ended, trimp_buffer, total_trimp
    global MQTT_PUBLISH_TOPIC

    # Safely load context with None checks
    device_id = ctx.get("device_id", "000")
    athlete_id = ctx.get("athlete_id", 0)
    name = ctx.get("name", "Unknown")
    age = ctx.get("age", 25)
    weight = ctx.get("weight", 70.0)
    height = ctx.get("height", 175.0)
    gender = ctx.get("gender", 1)
    hr_rest = ctx.get("hr_rest", 60)
    hr_max = ctx.get("hr_max", 195)
    MQTT_PUBLISH_TOPIC = ctx.get("MQTT_PUBLISH_TOPIC", "000/predictions")

    quaternion = ctx.get("madgwick_quaternion", np.array([1.0, 0.0, 0.0, 0.0]))
    hr_buffer = ctx.get("hr_buffer", deque(maxlen=ROLLING_WINDOW_SIZE))
    acc_buffer = ctx.get("acc_buffer", deque(maxlen=ROLLING_WINDOW_SIZE))
    gyro_buffer = ctx.get("gyro_buffer", deque(maxlen=ROLLING_WINDOW_SIZE))
    acc_mag_buffer = ctx.get("acc_mag_buffer", deque(maxlen=30))
    vel_buffer = ctx.get("vel_buffer", deque([0.0], maxlen=1))
    dist_buffer = ctx.get("dist_buffer", deque([0.0], maxlen=1))
    stress_buffer = ctx.get("stress_buffer", [])
    TEE_buffer = ctx.get("TEE_buffer", [])
    g_impact_events = ctx.get("g_impact_events", [])
    g_impact_count = ctx.get("g_impact_count", 0)
    acc_mag_history = ctx.get("acc_mag_history", deque(maxlen=5))
    gyro_mag_history = ctx.get("gyro_mag_history", deque(maxlen=5))

    session_start_time = ctx.get("session_start_time", None)
    last_vo2_update_time = ctx.get("last_vo2_update_time", None)
    last_data_time = ctx.get("last_data_time", time.time())
    last_warning_time = ctx.get("last_warning_time", 0)
    vo2_max_value = ctx.get("vo2_max_value", "-")
    session_end_time = ctx.get("session_end_time", None)
    session_ended = ctx.get("session_ended", False)
    trimp_buffer = ctx.get("trimp_buffer", [])
    total_trimp = ctx.get("total_trimp", 0.0)

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
def get_memory_status():
    """
    Get comprehensive memory status for both CPU and GPU.
    
    Returns:
        dict: Memory status information including CPU and GPU details
    """
    try:
        # CPU Memory
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        virtual_mem = psutil.virtual_memory()
        total_memory = virtual_mem.total / (1024 * 1024)  # MB
        available_memory = virtual_mem.available / (1024 * 1024)  # MB
        memory_percent = (process_memory / total_memory) * 100
        system_usage = 100 - (available_memory / total_memory) * 100
        
        cpu_status = {
            "process_memory_mb": round(process_memory, 2),
            "process_memory_percent": round(memory_percent, 3),
            "total_system_mb": round(total_memory, 0),
            "available_mb": round(available_memory, 0),
            "system_usage_percent": round(system_usage, 1)
        }
        
        # GPU Memory
        gpu_status = None
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                gpu_props = torch.cuda.get_device_properties(device)
                allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
                total_gpu = gpu_props.total_memory / (1024 * 1024)
                
                gpu_status = {
                    "device_name": gpu_props.name,
                    "total_memory_mb": round(total_gpu, 0),
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2),
                    "free_mb": round(total_gpu - reserved, 2),
                    "usage_percent": round((reserved / total_gpu) * 100, 1)
                }
            except Exception as e:
                gpu_status = {"error": f"GPU monitoring failed: {str(e)}"}
        else:
            gpu_status = {"status": "CUDA not available"}
        
        return {
            "cpu": cpu_status,
            "gpu": gpu_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {"error": f"Memory monitoring failed: {str(e)}"}

def print_memory_usage(label="", detailed=False):
    """
    Print memory usage information.
    
    Args:
        label: Optional label for the memory report
        detailed: If True, prints detailed CPU+GPU info; if False, prints simple process memory
    """
    try:
        if detailed:
            status = get_memory_status()
            
            if "error" in status:
                print(f"❌ Memory monitoring error: {status['error']}")
                return
            
            cpu = status["cpu"]
            gpu = status["gpu"]
            
            print(f"\n==== MEMORY STATUS {label} ====")
            print(f"CPU Process Memory: {cpu['process_memory_mb']} MB ({cpu['process_memory_percent']}% of system)")
            print(f"CPU Total System:  {cpu['total_system_mb']} MB")
            print(f"CPU Available:     {cpu['available_mb']} MB")
            print(f"CPU Usage:         {cpu['system_usage_percent']}%")
            
            if "error" in gpu:
                print(f"\nGPU Error: {gpu['error']}")
            elif "status" in gpu:
                print(f"\n{gpu['status']}")
            else:
                print(f"\nGPU Device:        {gpu['device_name']}")
                print(f"GPU Total Memory:  {gpu['total_memory_mb']} MB")
                print(f"GPU Allocated:     {gpu['allocated_mb']} MB")
                print(f"GPU Reserved:      {gpu['reserved_mb']} MB")
                print(f"GPU Free:          {gpu['free_mb']} MB")
                print(f"GPU Usage:         {gpu['usage_percent']}%")
        else:
            # Simple memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            print(f"💾 Memory{label}: {memory_mb:.0f} MB")
            
    except Exception as e:
        print(f"❌ Memory monitoring failed: {e}")

def print_detailed_memory_usage(label=""):
    """Print detailed CPU + GPU memory usage with system info (legacy function for compatibility)"""
    print_memory_usage(label, detailed=True)

# Prepare a fallback logger before any device context is created
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure logging with only console output initially
    # The mode-specific log file will be created when first device data arrives
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Only console output initially
        ]
    )
    
    logger.info(f"=== SYSTEM STARTUP ===")
    logger.info(f"Waiting for mode detection to create log file...")

# Broker config
MQTT_BROKER = "localhost" 
MQTT_PORT = 1883
idle_time = os.getenv("IDLE_TIME", 300)
USE_CUDA = os.getenv("USE_CUDA", "0")

# Enhanced CUDA detection with better error handling
def get_device():
    """Safely determine the best device to use."""
    if USE_CUDA in {"1", "true", "True"}:
        try:
            if torch.cuda.is_available():
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                print("CUDA is available and working")
                logger.info("CUDA is available and working - GPU processing enabled")
                return "cuda"
            else:
                print("CUDA requested but not available, falling back to CPU")
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        except Exception as e:
            print(f"CUDA error detected: {e}")
            print("Falling back to CPU processing")
            logger.error(f"CUDA error detected: {e} - Falling back to CPU processing")
            return "cpu"
    else:
        print("Using CPU processing")
        logger.info("CUDA not requested, using CPU processing")
        return "cpu"

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

DEVICE = get_device()

# Initialize dynamic model loader instead of loading all models at startup
print("Initializing dynamic model loader...")
logger.info("Initializing dynamic model loader for memory-efficient model management")

# Create dynamic model loader with Jetson Nano 4GB optimized settings
# Read configuration from jetson_nano_4gb_config.yaml
CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", str(config.get('model_loading', {}).get('cache_size', 3))))
MODEL_DEVICE = os.getenv("MODEL_DEVICE", config.get('model_loading', {}).get('device', 'cpu'))
MODELS_DIR = os.getenv("MODEL_DIRECTORY", config.get('model_loading', {}).get('models_directory', 'athlete_models_tensors_updated'))
ENABLE_MONITORING = config.get('model_loading', {}).get('enable_memory_monitoring', True)

model_loader = DynamicModelLoader(
    models_dir=MODELS_DIR,
    cache_size=CACHE_SIZE,
    device=MODEL_DEVICE,
    enable_memory_monitoring=ENABLE_MONITORING
)

# Get available models info
available_models = model_loader.get_available_player_ids()
max_available_models = len(available_models)

print(f"Dynamic model loader initialized (Jetson Nano 4GB optimized)")
print(f"   📁 Models directory: {MODELS_DIR}/")
print(f"   💾 Cache size: {CACHE_SIZE} models (optimized for 4GB RAM)")
print(f"   Device: {MODEL_DEVICE} (GPU acceleration enabled)")
print(f"   Available models: {max_available_models}")
print(f"   🎮 Available player IDs: {available_models}")
print(f"   🔧 Memory monitoring: {'Enabled' if ENABLE_MONITORING else 'Disabled'}")

logger.info(f"Dynamic Model Loader: Initialized with Jetson Nano 4GB optimized settings")
logger.info(f"Dynamic Model Loader: Cache size {CACHE_SIZE}, Device {MODEL_DEVICE}")
logger.info(f"Dynamic Model Loader: Found {max_available_models} available models")
logger.info(f"Dynamic Model Loader: Available player IDs: {available_models}")

# Load a default model for fallback (first available model)
default_model = None
if available_models:
    default_model = model_loader.get_model(available_models[0])
    if default_model:
        print(f"Default fallback model loaded for player {available_models[0]}")
        logger.info(f"Default fallback model loaded for player {available_models[0]}")
    else:
        print(f"Failed to load default fallback model")
        logger.warning("Failed to load default fallback model")

if default_model is None:
    raise RuntimeError("No models found under athlete_models_tensors_updated/. Please add models.")

print(f"Models will be loaded on-demand based on player/device IDs")
print(f"Memory usage will be optimized with LRU cache eviction")
logger.info("Dynamic model loading system ready - models will be loaded on-demand")

# =============================================================================
# STARTUP STATUS SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print(f"JETSON NANO ML PREDICTION ENGINE - STARTUP COMPLETE")
print(f"{'='*60}")
print(f"System Configuration:")
print(f"   - Mode: Dynamic Model Loading (Memory Optimized)")
print(f"   - Cache Size: {CACHE_SIZE} models (LRU eviction)")
print(f"   - Device: {MODEL_DEVICE} (GPU acceleration enabled)")
print(f"   - Available Models: {max_available_models} players")
print(f"   - Memory Monitoring: {'Enabled' if ENABLE_MONITORING else 'Disabled'}")
print(f"   - Health Metrics: Every 10 predictions")
print(f"   - Technical Reports: Every 100 predictions")
print(f"")
print(f"Ready for:")
print(f"   - Multi-device predictions (1-{max_available_models} players)")
print(f"   - Training mode (actual HR from sensors)")
print(f"   - Game mode (ML predictions)")
print(f"   - Real-time MQTT data processing")
print(f"")
print(f"Display Strategy:")
print(f"   - Health metrics: Every 10 predictions (HR, HRV, Stress, Recovery)")
print(f"   - Technical info: Every 100 predictions (Memory, Cache)")
print(f"   - Errors/warnings: Always logged to file")
print(f"{'='*60}")
print(f"Waiting for MQTT data...")
print(f"{'='*60}\n")

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Sampling and Processing Parameters
DT = 0.1                                        # Time step (seconds)
FS_HZ = 10.0                                    # Sampling frequency (Hz)
WINDOW_SECONDS = 3                              # Feature window duration (seconds)
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS_HZ)    # Samples per window
ROLLING_WINDOW_SIZE = 30                        # Main rolling window size
HRV_WINDOW_SECONDS = 10                         # HRV window duration (seconds)
HRV_WINDOW_SAMPLES = int(HRV_WINDOW_SECONDS * FS_HZ)

# G-Impact Detection Parameters
G_IMPACT_ACC_THRESHOLD = 8 * 9.81               # 8g threshold (m/s²)
G_IMPACT_GYRO_THRESHOLD = 300                   # Gyro threshold (deg/s)
G_IMPACT_JERK_THRESHOLD = 100                   # Jerk threshold (m/s³)

# =============================================================================
# SENSOR PROCESSING INITIALIZATION
# =============================================================================

# Madgwick Filter for Orientation Estimation
madgwick_filter = Madgwick()
quaternion = np.array([1.0, 0.0, 0.0, 0.0])

# =============================================================================
# DATA BUFFERS
# =============================================================================

# Primary Sensor Data Buffers
hr_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
acc_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
gyro_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
acc_mag_buffer = deque(maxlen=30)

# Motion Analysis Buffers
vel_buffer = deque([0.0], maxlen=1)
dist_buffer = deque([0.0], maxlen=1)

# Feature Engineering Window Buffers (per-axis, 3s @ 10Hz)
acc_x_win = deque(maxlen=WINDOW_SAMPLES)
acc_y_win = deque(maxlen=WINDOW_SAMPLES)
acc_z_win = deque(maxlen=WINDOW_SAMPLES)
gyro_x_win = deque(maxlen=WINDOW_SAMPLES)
gyro_y_win = deque(maxlen=WINDOW_SAMPLES)
gyro_z_win = deque(maxlen=WINDOW_SAMPLES)

# G-Impact Detection History
acc_mag_history = deque(maxlen=5)
gyro_mag_history = deque(maxlen=5)

# =============================================================================
# HEALTH METRICS BUFFERS
# =============================================================================

# Stress and Energy Tracking
stress_buffer = []                              # Stress Buffer    
TEE_buffer = []                                 # Total Energy Expenditure
hrv_hr_buffer = deque(maxlen=HRV_WINDOW_SAMPLES) # Dedicated HRV buffer (10s)

# TRIMP (Training Impulse) Tracking
trimp_buffer = []                                # TRIMP Buffer
total_trimp = 0.0                                # Total TRIMP

# G-Impact Event Tracking
g_impact_events = []                            # List of (timestamp, g_impact) tuples
g_impact_count = 0                              # Counter for high-g impacts

# =============================================================================
# SESSION AND TIMING TRACKING
# =============================================================================

# Session Lifecycle
session_start_time = None
session_end_time = None
session_ended = False

# Update Timers
last_vo2_update_time = None
last_data_time = time.time()
last_warning_time = 0

# Health Metrics Values
vo2_max_value = "-"

# =============================================================================
# MQTT CONNECTION TRACKING
# =============================================================================

mqtt_connected = False
mqtt_last_connect_time = 0
mqtt_last_disconnect_time = 0
mqtt_reconnect_attempts = 0
mqtt_last_status_report = 0

# =============================================================================
# ATHLETE PROFILE AND DATA
# =============================================================================

# Default Athlete Profile (will be initialized per-device)
hr_rest = 60                                    # Resting heart rate (BPM)
hr_max = None                                   # Maximum heart rate (calculated from age)

# Global Athlete Profile Storage
Athlete_profile = {}

# Current Sensor Data
sensor_data = {
    "acc": None,
    "gyro": None,
    "magno": None
}

# Position Tracking
latest_position = {"x": None, "y": None}

# MQTT Configuration
MQTT_PUBLISH_TOPIC = "predictions"


# --- Model input dimension cache and helpers ---
model_input_dims = {}

def _safe_prepare_input(base_features, required_dim):
    """Pad or truncate feature vector to required_dim."""
    arr = np.asarray(base_features, dtype=np.float32)
    if arr.size < required_dim:
        padded = np.zeros(required_dim, dtype=np.float32)
        padded[:arr.size] = arr
        return padded
    elif arr.size > required_dim:
        return arr[:required_dim]
    return arr

def _infer_input_dim_for_model(model, base_len, max_try=64):
    """Empirically find an input dimension that the model accepts."""
    for k in range(base_len, max_try + 1):
        try:
            trial = np.zeros((1, k), dtype=np.float32)
            _ = model.predict(trial)
            return k
        except Exception:
            continue
    # Fall back to base_len if nothing succeeded
    return base_len

def predict_with_adaptive_input(model, base_features):
    """Run model.predict with features padded/truncated to model's expected size."""
    try:
        mid = model_input_dims.get(id(model))
        base_len = len(base_features)
        if mid is None:
            mid = _infer_input_dim_for_model(model, base_len)
            model_input_dims[id(model)] = mid
        x_vec = _safe_prepare_input(base_features, mid)
        x_batch = np.asarray([x_vec], dtype=np.float32)
        
        # Try prediction with error handling
        try:
            result = model.predict(x_batch)
            return result
        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Check for specific CUDA/Hummingbird errors
            error_str = str(e).lower()
            is_cuda_error = ("cuda" in error_str or "gpu" in error_str or 
                           "indices element is out of data bounds" in error_str or
                           "out of bounds" in error_str)
            
            if DEVICE == "cuda" and is_cuda_error:
                try:
                    print("Attempting CPU fallback for prediction")
                    # Create a temporary CPU copy of the model
                    cpu_model = model
                    if hasattr(cpu_model, 'to'):
                        cpu_model = cpu_model.to('cpu')
                    result = cpu_model.predict(x_batch)
                    print("CPU fallback successful")
                    return result
                except Exception as cpu_e:
                    print(f"CPU fallback also failed: {cpu_e}")
                    # Return a default prediction
                    return np.array([60.0])  # Default heart rate
            else:
                # For non-CUDA errors, return default
                return np.array([60.0])
                
    except Exception as e:
        print(f"Error in predict_with_adaptive_input: {e}")
        return np.array([60.0])  # Default heart rate

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
            "acc_x_mean": ax_mean, "acc_x_std": ax_std, "acc_x_min": ax_min, "acc_x_max": ax_max, "acc_x_range": ax_range,
            "acc_y_mean": ay_mean, "acc_y_std": ay_std, "acc_y_min": ay_min, "acc_y_max": ay_max, "acc_y_range": ay_range,
            "acc_z_mean": az_mean, "acc_z_std": az_std, "acc_z_min": az_min, "acc_z_max": az_max, "acc_z_range": az_range,
            "gyro_x_mean": gx_mean, "gyro_x_std": gx_std, "gyro_x_min": gx_min, "gyro_x_max": gx_max, "gyro_x_range": gx_range,
            "gyro_y_mean": gy_mean, "gyro_y_std": gy_std, "gyro_y_min": gy_min, "gyro_y_max": gy_max, "gyro_y_range": gy_range,
            "gyro_z_mean": gz_mean, "gyro_z_std": gz_std, "gyro_z_min": gz_min, "gyro_z_max": gz_max, "gyro_z_range": gz_range,
            "resultant_mean": a_res_mean, "resultant_std": a_res_std, "resultant_cov": a_res_cov,
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
    """
    Calculate RMSSD (Root Mean Square of Successive Differences) for HRV analysis.
    
    Args:
        hr_values: List/array of heart rate values in BPM
        
    Returns:
        RMSSD value in milliseconds, or 0.0 if insufficient/invalid data
    """
    if len(hr_values) < 5:
        return 0.0  # Need at least 5 HR values for reliable RMSSD
    
    # Convert to numpy array and filter invalid values
    hr_array = np.array(hr_values)
    
    # Filter out physiologically impossible HR values
    valid_hr = hr_array[(hr_array > 30) & (hr_array < 220)]
    
    if len(valid_hr) < 3:
        return 0.0  # Need at least 3 valid values
    
    # Convert HR to RR intervals (milliseconds)
    rr_intervals = 60000 / valid_hr
    
    # Calculate successive differences
    rr_diffs = np.diff(rr_intervals)
    
    # Remove extreme outliers (>3 standard deviations from mean)
    if len(rr_diffs) > 2:
        mean_diff = np.mean(rr_diffs)
        std_diff = np.std(rr_diffs)
        
        # Keep only differences within 3 standard deviations
        valid_diffs = rr_diffs[np.abs(rr_diffs - mean_diff) <= 3 * std_diff]
        
        if len(valid_diffs) > 0:
            # Calculate RMSSD
            rmssd = np.sqrt(np.mean(valid_diffs ** 2))
            return round(rmssd, 2)
    
    return 0.0

def estimate_vo2_max(age, gender, current_hr, hrv, hr_rest=60, hr_max=None):
    """
    Estimate VO2 max based on heart rate, HRV, age, and gender.
    
    Uses a simplified estimation method based on:
    - Heart Rate Reserve (HRR) as primary indicator
    - HRV as secondary indicator of cardiovascular fitness
    - Age and gender adjustments based on physiological norms
    
    Args:
        age: Age in years
        gender: 1 for male, 0 for female
        current_hr: Current heart rate (BPM)
        hrv: Heart rate variability (RMSSD in ms)
        hr_rest: Resting heart rate (BPM)
        hr_max: Maximum heart rate (BPM), calculated if None
        
    Returns:
        Estimated VO2 max in ml/kg/min (clamped to realistic range 20-80)
    """
    if hr_max is None:
        hr_max = 220 - age  # Standard age-based formula
    
    # Heart Rate Reserve (HRR) - correct calculation
    hrr = (current_hr - hr_rest) / (hr_max - hr_rest + 1e-6)
    hrr = max(0, min(1, hrr))  # Clamp to [0,1]
    
    # Base VO2 max estimation using HRR
    # This is a simplified model - in reality, VO2 max requires lab testing
    base_vo2 = 15.0 + (hrr * 45.0)  # Range: 15-60 ml/kg/min based on HRR
    
    # Age adjustment (VO2 max declines with age)
    age_factor = -0.4 * (age - 20)  # Decline after age 20
    
    # Gender adjustment (males typically have higher VO2 max)
    gender_factor = 8.0 if gender == 1 else 0.0  # Males typically 8-12 ml/kg/min higher
    
    # HRV adjustment (higher HRV indicates better cardiovascular fitness)
    if hrv > 0:
        # Normalize HRV: 50ms = 0 adjustment, 100ms+ = +5 adjustment
        hrv_factor = max(0, min(5, (hrv - 50) / 10))
    else:
        hrv_factor = 0
    
    # Calculate final VO2 max
    vo2_max = base_vo2 + age_factor + gender_factor + hrv_factor
    
    # Clamp to physiologically realistic range
    return round(max(20, min(80, vo2_max)), 1)

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
    """
    Calculate stress level based on physiological and activity indicators.
    
    Args:
        hr: Current heart rate (BPM)
        hrv: Heart rate variability (RMSSD in ms)
        acc_mag: Acceleration magnitude (m/s²)
        gyro_mag: Gyroscope magnitude (deg/s)
        age: Age in years
        gender: 1 for male, 0 for female
        hr_rest: Resting heart rate (BPM)
        hr_max: Maximum heart rate (BPM)
        
    Returns:
        Stress percentage (0-100)
    """
    # Heart Rate Reserve (correct calculation)
    hrr = (hr - hr_rest) / (hr_max - hr_rest + 1e-6)
    hrr = max(0, min(1, hrr))  # Clamp to [0,1]
    
    # HRV normalization (improved)
    # Typical RMSSD range: 20-200ms, with 50-100ms being normal
    # Higher HRV = lower stress, so invert the relationship
    if hrv <= 0:
        hrv_norm = 1.0  # No HRV data = assume high stress
    else:
        # Normalize: 100ms = 0 stress, 20ms = 1 stress
        hrv_norm = max(0, min(1, (100 - hrv) / 80))
    
    # Activity normalization (improved thresholds)
    acc_norm = min(acc_mag / 15, 1.0)  # Increased threshold from 10 to 15
    gyro_norm = min(gyro_mag / 15, 1.0)  # Increased threshold from 10 to 15
    
    # Weighted score (more balanced, evidence-based)
    score = (
        0.5 * hrr +           # Heart rate is primary stress indicator
        0.3 * hrv_norm +      # HRV is secondary indicator
        0.1 * acc_norm +      # Activity contributes less
        0.1 * gyro_norm       # Gyro contributes less
    )
    
    # Gentler sigmoid curve (less steep transition)
    # Adjust parameters for more realistic stress response curve
    stress_percent = 100 * (1 / (1 + np.exp(-4 * (score - 0.3))))
    
    # Clamp to valid range
    return round(max(0, min(100, stress_percent)), 1)

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

def get_recovery_recommendations(total_trimp, stress_percent):
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
    
    # Additional recommendations based on stress
    if stress_percent > 70:
        recommendations.append("High stress detected - consider stress management techniques")
    
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
            "Maintain proper nutrition"
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
    global session_end_time, total_trimp, stress_buffer, g_impact_count
    
    if session_end_time is None:
        return None
    
    session_duration = (session_end_time - session_start_time) / 60.0  # in minutes
    avg_stress = round(sum(stress_buffer) / len(stress_buffer), 2) if stress_buffer else 0
    
    # Calculate final TRIMP zone and recommendations
    trimp_zone, zone_description = get_trimp_zone(total_trimp)
    recovery_time, recovery_recommendations = get_recovery_recommendations(total_trimp, avg_stress)
    training_recommendations = get_training_recommendations(trimp_zone, avg_stress)
    
    summary = {
        "session_end_time": datetime.fromtimestamp(session_end_time).isoformat(),
        "session_duration_minutes": round(session_duration, 2),
        "total_trimp": total_trimp,
        "trimp_zone": trimp_zone,
        "zone_description": zone_description,
        "avg_stress": avg_stress,
        "g_impact_count": g_impact_count,
        "recovery_time": recovery_time,
        "recovery_recommendations": recovery_recommendations,
        "training_recommendations": training_recommendations
    }
    
    return summary


def process_data():
    global session_start_time, last_vo2_update_time, vo2_max_value
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
        jerk = abs(acc_mag_history[-1] - acc_mag_history[-2]) / DT
    else:
        jerk = 0
    
    if len(acc_mag_buffer) >= 30:
        acc_filtered = butter_bandpass_filter(list(acc_mag_buffer))
        if acc_filtered is not None:
            acc_buffer.append(acc_filtered[-1])

    # Calculate velocity and distance based on current acceleration
    if len(acc_buffer) >= 2:  # Need at least 2 readings for velocity calculation
        # Use the most recent acceleration magnitude for velocity calculation
        current_acc = acc_buffer[-1] if len(acc_buffer) > 0 else 0.0
        prev_vel = vel_buffer[-1] if len(vel_buffer) > 0 else 0.0
        
        # Simple velocity calculation: v = v0 + a * dt
        new_vel = prev_vel + current_acc * DT
        
        # Check if resting (lower threshold for more sensitivity)
        is_resting = current_acc < 0.5  # Reduced from 0.75 to 0.5
        
        if is_resting:
            new_vel = 0.0
            vel_buffer.clear()
            vel_buffer.append(0.0)
            new_dist = dist_buffer[-1] if len(dist_buffer) > 0 else 0.0
            dist_buffer.append(new_dist)
        else:
            vel_buffer.append(new_vel)
            prev_dist = dist_buffer[-1] if len(dist_buffer) > 0 else 0.0
            new_dist = prev_dist + new_vel * DT  # Simplified distance calculation
            dist_buffer.append(new_dist)
    else:
        # Use raw acceleration magnitude if filtered buffer is not ready
        if len(acc_mag_buffer) > 0:
            current_acc = acc_mag_buffer[-1]
            new_vel = current_acc * DT  # Simple velocity from acceleration
            new_dist = dist_buffer[-1] + new_vel * DT if len(dist_buffer) > 0 else new_vel * DT
            dist_buffer.append(new_dist)
            vel_buffer.append(new_vel)
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

    # --- Additional rolling (last 10) feature engineering per user spec ---
    try:
        roll_n = 10
        if len(acc_x_win) >= roll_n:
            ax_arr_10 = np.asarray(list(acc_x_win)[-roll_n:], dtype=np.float32)
            ay_arr_10 = np.asarray(list(acc_y_win)[-roll_n:], dtype=np.float32)
            az_arr_10 = np.asarray(list(acc_z_win)[-roll_n:], dtype=np.float32)

            resultant_10 = np.sqrt(ax_arr_10**2 + ay_arr_10**2 + az_arr_10**2)

            # Basic stats
            acc_x_mean_10 = float(np.mean(ax_arr_10))
            acc_y_mean_10 = float(np.mean(ay_arr_10))
            acc_z_mean_10 = float(np.mean(az_arr_10))
            acc_x_std_10 = float(np.std(ax_arr_10, ddof=0))
            acc_y_std_10 = float(np.std(ay_arr_10, ddof=0))
            acc_z_std_10 = float(np.std(az_arr_10, ddof=0))
            resultant_mean_10 = float(np.mean(resultant_10))
            resultant_std_10 = float(np.std(resultant_10, ddof=0))

            # Jerk features
            jerk_10 = np.abs(np.diff(resultant_10)) * FS_HZ
            jerk_val = float(np.mean(jerk_10)) if jerk_10.size > 0 else 0.0  # use as 'jerk'
            jerk_mean_10 = jerk_val  # same as jerk_mean for now

            # FFT features
            res_centered = resultant_10 - np.mean(resultant_10)
            spec = np.abs(rfft(res_centered))
            freqs = rfftfreq(resultant_10.shape[0], d=1.0/FS_HZ)
            if spec.size > 1:
                peak_idx = int(np.argmax(spec[1:]) + 1)
                fft_peak_freq_10 = float(freqs[peak_idx])
            else:
                fft_peak_freq_10 = 0.0
            fft_energy_10 = float(np.sum(spec**2))

            # Collect window features
            window_features = {
                "acc_x_mean": acc_x_mean_10,
                "acc_y_mean": acc_y_mean_10,
                "acc_z_mean": acc_z_mean_10,
                "acc_x_std": acc_x_std_10,
                "acc_y_std": acc_y_std_10,
                "acc_z_std": acc_z_std_10,
                "resultant_mean": resultant_mean_10,
                "resultant_std": resultant_std_10,
                "jerk": jerk_val,
                "jerk_mean": jerk_mean_10,
                "fft_peak_freq": fft_peak_freq_10,
                "fft_energy": fft_energy_10,
            }
    except Exception:
        # In case of any failure, set defaults
        window_features = {
            "acc_x_mean": 0.0, "acc_y_mean": 0.0, "acc_z_mean": 0.0,
            "acc_x_std": 0.0, "acc_y_std": 0.0, "acc_z_std": 0.0,
            "resultant_mean": 0.0, "resultant_std": 0.0,
            "jerk": 0.0, "jerk_mean": 0.0,
            "fft_peak_freq": 0.0, "fft_energy": 0.0
        }

# Prepare features in the exact order the model expects
    features = [
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
        window_features["acc_x_mean"],
        window_features["acc_y_mean"],
        window_features["acc_z_mean"],
        float(np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)),  # resultant (single value)
        window_features["acc_x_std"],
        window_features["acc_y_std"],
        window_features["acc_z_std"],
        window_features["resultant_mean"],
        window_features["resultant_std"],
        window_features["jerk"],
        window_features["jerk_mean"],
        window_features["fft_peak_freq"],
        window_features["fft_energy"],
        age, weight, height, gender
    ]


    # Determine mode early and avoid ML usage entirely in training mode
    mode = sensor_data.get("mode", "game")

    if mode == "training":
        actual_hr = sensor_data.get("heart_rate_bpm")
        if actual_hr is None:
            print(f"Training mode: no actual HR for Player {athlete_id} (Device {device_id}); skipping this sample")
            logger.warning(f"Training mode: missing heart_rate_bpm for device {device_id}")
            return
        predicted_hr = round(float(actual_hr), 0)
    else:
        # Dynamic model loading - only predict for the current device's model
        # This saves significant memory compared to batch predictions across all models
        device_model = None
        try:
            device_idx = int(str(device_id).lstrip("0") or "0")
        except Exception:
            device_idx = 0

        # Only proceed if a model exists for this specific device/player
        if device_idx <= 0 or not model_loader.is_model_available(device_idx):
            print(f"Skipping Player {athlete_id} (Device {device_id}) - no model available")
            logger.info(f"Skipping device {device_id} - no model available")
            return

        # Get model for this device (loads on-demand if not in cache)
        device_model = model_loader.get_model(device_idx)
        if device_model is None:
            print(f"Skipping Player {athlete_id} (Device {device_id}) - model failed to load")
            logger.warning(f"Device {device_id} model failed to load")
            return

        # In game mode, predict heart rate using the dynamically loaded model
        device_type = "CUDA" if DEVICE == "cuda" else "CPU"
        try:
            predicted_hr = float(predict_with_adaptive_input(device_model, features)[0])
            predicted_hr = round(predicted_hr, 0)
        except Exception as e:
            print(f"Prediction failed for Player {athlete_id} (Device {device_id}): {e}")
            logger.error(f"Prediction failed for device {device_id}: {e}")
            return


    hr_buffer.append(predicted_hr)
    hrv_hr_buffer.append(predicted_hr)

    # Display activity metrics for every prediction (moved above heart rate)
    current_acc_display = acc_buffer[-1] if len(acc_buffer) > 0 else (acc_mag_buffer[-1] if len(acc_mag_buffer) > 0 else 0.0)
    print(f"Activity: Velocity: {round(new_vel, 2)} m/s | Distance: {round(new_dist, 2)} m | Acc: {round(current_acc_display, 2)} m/s²")

    # Show heart rate and HRV information for every prediction
    # Get actual HR if available (training mode)
    actual_hr = sensor_data.get("heart_rate_bpm") if mode == "training" else None
    
    # Display heart rate information for every prediction
    if actual_hr is not None:
        print(f"Heart Rate: {actual_hr} bpm (actual) | {predicted_hr} bpm (predicted) | Mode: {mode}")
    else:
        print(f"Heart Rate: {predicted_hr} bpm (predicted) | Mode: {mode}")
    
    # Display HRV (single source): 10-second window if available
    if len(hrv_hr_buffer) >= 5:
        current_hrv = calculate_rmssd(hrv_hr_buffer)
        print(f"HRV (RMSSD, 10s): {current_hrv:.1f} ms")
    else:
        print(f"HRV (RMSSD): Calculating... (need {5 - len(hr_buffer)} more readings)")
    
    # Show comprehensive health metrics every 5 predictions for better visibility
    if len(hr_buffer) % 5 == 0 and len(hr_buffer) > 0:  # Show detailed metrics every 5 predictions
        # Display stress if available
        if len(hr_buffer) >= ROLLING_WINDOW_SIZE:
            current_stress = calculate_stress(np.mean(hr_buffer), hrv_rmssd, np.mean(acc_buffer), np.mean(gyro_buffer), age, gender, hr_rest, hr_max)
            print(f"Stress Level: {current_stress:.1f}%")
        

    # Robust stress calculation using rolling means
    if len(hr_buffer) == ROLLING_WINDOW_SIZE:
        hrv_rmssd = calculate_rmssd(hrv_hr_buffer)
        acc_mean = np.mean(acc_buffer)
        gyro_mean = np.mean(gyro_buffer)
        hr_mean = np.mean(hr_buffer)
        stress_percent = calculate_stress(hr_mean, hrv_rmssd, acc_mean, gyro_mean, age, gender, hr_rest, hr_max)
        stress_buffer.append(stress_percent)
        avg_stress = round(sum(stress_buffer) / len(stress_buffer), 2)
        stress_label = (
            "Low" if avg_stress < 40 else
            "Moderate" if avg_stress < 70 else
            "High"
        )
        print(f"Stress: {stress_percent:.1f}% (avg: {avg_stress:.1f}%) - {stress_label}")
    else:
        hrv_rmssd = 0
        stress_percent = 0
        avg_stress = 0
        stress_label = "-"

    now = time.time()
    if session_start_time is None:
        session_start_time = now
        # Only log session start once

    elapsed_time = now - session_start_time

    if elapsed_time >= 300:
        if last_vo2_update_time is None or (now - last_vo2_update_time >= 300):
            if len(hr_buffer) > 0:  # Ensure we have HR data
                vo2_max_value = estimate_vo2_max(age, gender, np.mean(hr_buffer), hrv_rmssd, hr_rest, hr_max)
                last_vo2_update_time = now
                # VO2 updates logged to file only, no console output
                logger.info(f"VO2 Max Updated: {vo2_max_value} ml/kg/min (HR: {np.mean(hr_buffer):.0f}, HRV: {hrv_rmssd:.1f})")
            else:
                # VO2 warnings logged to file only, no console output
                pass
    else:
        vo2_max_value = "-"
        # VO2 waiting message removed to reduce verbosity


    # --- Total Energy Expenditure Calculation ---
    tte = training_energy_expenditure(new_vel, DT, weight)
    TEE_buffer.append(tte)
    active_tee = round(sum(TEE_buffer), 2)

    print(f"Energy: {tte:.2f} kcal | Total: {active_tee} kcal")

    # --- TRIMP Calculation ---
    if len(hr_buffer) >= 5:                             # Need at least 5 HR readings for meaningful TRIMP
        hr_avg = np.mean(list(hr_buffer)[-5:])          # Use last 5 HR readings
        duration_min = elapsed_time / 60.0              # Convert to minutes
        gender_str = "male" if gender == 1 else "female"
        
        current_trimp = calculate_trimp(hr_avg, hr_rest, hr_max, duration_min, gender_str)
        trimp_buffer.append(current_trimp)
        total_trimp = round(sum(trimp_buffer), 2)
        
        # Get TRIMP zone and recovery recommendations
        trimp_zone, zone_description = get_trimp_zone(total_trimp)
        recovery_time, recovery_recommendations = get_recovery_recommendations(total_trimp, avg_stress)
        
        print(f"TRIMP: {round(current_trimp, 2)} | Total: {total_trimp} | Zone: {trimp_zone}")
        print(f"Recovery: {recovery_time} | {zone_description}")
    else:
        current_trimp = 0
        total_trimp = 0

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
        impact_msg = f"High G-Impact detected! ({acc_mag:.2f} m/s², gyro: {gyro_mag:.2f} deg/s, jerk: {jerk:.2f} m/s³) at {event_time} - Possible injury risk. Position: ({latest_position['x']}, {latest_position['y']}) Axis: {max_axis}"
        print(impact_msg)
        logger.warning(impact_msg)
        # When saving g-impact log, use organized folder structure:
        base_output_dir = "prediction_outputs"
        player_folder = f"A{str(athlete_id)}_{str(name)}"
        full_player_path = os.path.join(base_output_dir, player_folder)
        os.makedirs(full_player_path, exist_ok=True)
        with open(os.path.join(full_player_path, f"{athlete_id}_g_impact_log.json"), "w") as f:
            json.dump(g_impact_events, f, indent=2)

    output = {
        "timestamp": datetime.fromtimestamp(time.time()).isoformat(),
        "device_id": device_id,
        "athlete_id": athlete_id,
        "athlete_profile": {
            "name": name,
            "age": age,
            "weight": weight,
            "height": height,
            "gender": "Male" if gender == 1 else "Female"
        },
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
        "injury_risk": injury_risk,
        "g_impact": round(acc_mag, 2),
        "g_impact_count": g_impact_count,
        "g_impact_events": g_impact_events[-10:],       # Last 10 events for quick view
        "current_trimp": round(current_trimp, 2),
        "total_trimp": total_trimp,
        "hr_rest": hr_rest,
        "hr_max": hr_max,
        # Include engineered features snapshot for this window
        "window_features": window_features if window_features is not None else {},
        # Only include the prediction for this specific device's model (for game mode)
        "model_prediction": {
            "model_id": device_idx if mode == "game" else None,
            "predicted_hr": predicted_hr if mode == "game" else None,
            "model_loaded_dynamically": True,
            "cache_info": model_loader.get_cache_info() if mode == "game" else None
        } if mode == "game" else None
    }

    # When saving realtime output, use organized folder structure:
    base_output_dir = "prediction_outputs"
    player_folder = f"A{str(athlete_id)}_{str(name)}"
    full_player_path = os.path.join(base_output_dir, player_folder)
    os.makedirs(full_player_path, exist_ok=True)

    with open(os.path.join(full_player_path, f"A{athlete_id}_D{device_id}_realtime_output.json"), "w") as f:
        json.dump(output, f, indent=2)

    client.publish(MQTT_PUBLISH_TOPIC, json.dumps(output))
    # MQTT publishing logged to file only, no console output
    if device_id not in prediction_logged_devices:
        logger.info(f"Published prediction to MQTT topic: {MQTT_PUBLISH_TOPIC}")
        prediction_logged_devices.add(device_id)
    
    # Memory monitoring and cache status (every 100 data points to reduce noise)
    if len(hr_buffer) % 100 == 0 and len(hr_buffer) > 0:
        print_memory_usage(" (periodic check)")
        
        # Show dynamic model loader cache status
        cache_info = model_loader.get_cache_info()
        if cache_info["total_requests"] > 0:
            print(f"🧠 Model Cache: {cache_info['cache_size']}/{cache_info['max_cache_size']} models, "
                  f"Hit rate: {cache_info['cache_hit_rate']:.1%}, "
                  f"Requests: {cache_info['total_requests']}")
            print(f"Cache Details: Loaded: {cache_info['models_loaded']}, "
                  f"Evicted: {cache_info['models_evicted']}, "
                  f"Avg load time: {cache_info['average_load_time']:.2f}s")

# --- MQTT callbacks ---
def on_connect(client, userdata, flags, rc):
    global mqtt_connected, mqtt_last_connect_time, mqtt_reconnect_attempts
    mqtt_last_connect_time = time.time()
    
    if rc == 0:
        mqtt_connected = True
        logger.info(f"Connected to MQTT Broker with result code: {rc}")
        # Subscribe to all per-player topics and keep legacy topic for compatibility
        client.subscribe("player/+/sensor/data")
        client.subscribe("sensor/data")
        logger.info("Subscribed to player/+/sensor/data and sensor/data topics")
        mqtt_reconnect_attempts = 0  # Reset counter on successful connection
    else:
        mqtt_connected = False
        logger.error(f"Failed to connect to MQTT Broker with result code {rc}")
        mqtt_reconnect_attempts += 1

def on_disconnect(client, userdata, rc):
    global mqtt_connected, mqtt_last_disconnect_time, mqtt_reconnect_attempts
    mqtt_connected = False
    mqtt_last_disconnect_time = time.time()    
    logger.warning(f"Disconnected from MQTT Broker with result code: {rc}")
    
    while not mqtt_connected:
        try:
            mqtt_reconnect_attempts += 1
            # Reconnection attempt logged to file only
            client.reconnect()
            logger.info("Successfully reconnected to MQTT Broker")
            break
        except Exception as e:
            logger.warning(f"Reconnection attempt failed: {e}, retrying in 5 seconds...")
            time.sleep(5)

def on_message(client, userdata, msg):
    topic = msg.topic
    global sensor_data, unique_mqtt_topics

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
        
        # Track unique MQTT topics
        unique_mqtt_topics.add(topic)

        # Show that we received sensor data (less verbose)
        athlete_id = parsed_data.get("athlete_id", "unknown")
        print(f"📥 Player {athlete_id} (Device {device_id_str}) - New sensor data")
        
        # Log device activity and detect mode
        mode = parsed_data.get("mode", "game")
        _update_log_filename_if_needed(mode)
        
        # Only log device activity on first activation, not every message
        if device_id_str not in device_contexts:
            logger.info(f"Device {device_id_str} (Player {athlete_id}) - First activation, Mode: {mode}")
            logger.info(f"Device {device_id_str} - MQTT prediction topic: {device_id_str}/predictions")

        # Get or init per-device context and sync into globals
        if device_id_str not in device_contexts:
            _init_device_context(device_id_str)
        ctx = device_contexts[device_id_str]
        _load_context_to_globals(ctx)
        
        # Optionally update athlete profile from payload if provided
        name_in = parsed_data.get("name")
        age_in = parsed_data.get("age")
        weight_in = parsed_data.get("weight")
        height_in = parsed_data.get("height")
        gender_in = parsed_data.get("gender")          # 'M'/'F' or 1/0
        updated = False
        if name_in is not None:
            ctx["name"] = str(name_in)
            updated = True
        if age_in is not None:
            try:
                ctx["age"] = int(age_in)
                updated = True
            except Exception:
                pass
        if weight_in is not None:
            try:
                ctx["weight"] = float(weight_in)
                updated = True
            except Exception:
                pass
        if height_in is not None:
            try:
                ctx["height"] = float(height_in)
                updated = True
            except Exception:
                pass
        if gender_in is not None:
            try:
                if isinstance(gender_in, str):
                    ctx["gender"] = 1 if gender_in.upper().startswith("M") else 0
                else:
                    ctx["gender"] = 1 if int(gender_in) == 1 else 0
                updated = True
            except Exception:
                pass
        if updated:
            # update derived fields
            ctx["hr_max"] = 220 - int(ctx.get("age", 25))
            _load_context_to_globals(ctx)
            print(f"Player {athlete_id}: Age={ctx['age']}, Weight={ctx['weight']}kg, Height={ctx['height']}cm, Gender={'Male' if ctx['gender'] == 1 else 'Female'}, HR_max={ctx['hr_max']}")
            # Only log profile updates, not every data point
        
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
        print(f"Processing Player {athlete_id} data...")
        process_data()
        print(f"Player {athlete_id} processing complete")
        print(" ")                                      # Noticeable line break between players
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

            # Session end conditions are logged at summary level

            summary = generate_session_summary()
            if summary:
                print(f"\nSESSION SUMMARY (device {device_id_str}):")
                print(f"Duration: {summary['session_duration_minutes']} minutes")
                print(f"Total TRIMP: {summary['total_trimp']}")
                print(f"TRIMP Zone: {summary['trimp_zone']} - {summary['zone_description']}")
                print(f"Average Stress: {summary['avg_stress']}%")
                print(f"G-Impact Events: {summary['g_impact_count']}")

                base_output_dir = "prediction_outputs"
                player_folder = f"A{str(athlete_id)}_{str(name)}"
                full_player_path = os.path.join(base_output_dir, player_folder)
                os.makedirs(full_player_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                summary_filename = f"A{athlete_id}_D{device_id}_session_summary_{timestamp}.json"
                summary_filepath = os.path.join(full_player_path, summary_filename)
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
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    logger.info(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    logger.info(f"Starting test deployment in multi-device mode")
    logger.info(f"MQTT subscription topics: player/+/sensor/data, sensor/data")
    # Reduced logging for subscription messages

    # Initialize separate timers for different purposes
    last_status_report_time = 0  # For status updates every 30 seconds
    last_warning_report_time = 0  # For warnings every 5 seconds
    
    # Simple MQTT topic tracking
    unique_mqtt_topics = set()

    while True:
        client.loop(timeout=1.0)  # Non-blocking with 1-second timeout
        current_time = time.time()

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
                print(f"\nSESSION SUMMARY:")
                logger.info("SESSION SUMMARY:")
                print(f"Duration: {summary['session_duration_minutes']} minutes")
                logger.info(f"Duration: {summary['session_duration_minutes']} minutes")
                print(f"Total TRIMP: {summary['total_trimp']}")
                logger.info(f"Total TRIMP: {summary['total_trimp']}")
                print(f"TRIMP Zone: {summary['trimp_zone']} - {summary['zone_description']}")
                logger.info(f"TRIMP Zone: {summary['trimp_zone']} - {summary['zone_description']}")
                print(f"Average Stress: {summary['avg_stress']}%")
                logger.info(f"Average Stress: {summary['avg_stress']}%")
                print(f"G-Impact Events: {summary['g_impact_count']}")
                logger.info(f"G-Impact Events: {summary['g_impact_count']}")
                
                print(f"\nRECOVERY RECOMMENDATIONS:")
                logger.info("RECOVERY RECOMMENDATIONS:")
                print(f"Recovery Time: {summary['recovery_time']}")
                logger.info(f"Recovery Time: {summary['recovery_time']}")
                for i, rec in enumerate(summary['recovery_recommendations'], 1):
                    print(f"  {i}. {rec}")
                    logger.info(f"  {i}. {rec}")
                
                print(f"\nTRAINING RECOMMENDATIONS:")
                logger.info("TRAINING RECOMMENDATIONS:")
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
                
                # Final memory usage and cache statistics
                print_detailed_memory_usage(" (session end)")
                
                # Show final dynamic model loader statistics
                final_cache_info = model_loader.get_cache_info()
                print(f"\n🧠 DYNAMIC MODEL LOADER FINAL STATISTICS:")
                print(f"   Models loaded: {final_cache_info['models_loaded']}")
                print(f"   Models evicted: {final_cache_info['models_evicted']}")
                print(f"   Cache hit rate: {final_cache_info['cache_hit_rate']:.1%}")
                print(f"   Total requests: {final_cache_info['total_requests']}")
                print(f"   Average load time: {final_cache_info['average_load_time']:.2f}s")
                print(f"   Final cache size: {final_cache_info['cache_size']}/{final_cache_info['max_cache_size']}")
                print(f"   Available models: {final_cache_info['available_models']}")
                
                logger.info(f"Dynamic Model Loader Final Stats: {final_cache_info}")
                
                print("="*60)
                logger.info("="*60)
                logger.info("=== SESSION COMPLETED ===")
                
                # Cleanup prediction lockfile before exiting
                remove_prediction_lockfile()
                break  # Exit the loop after generating summary
        
        # Active device monitoring (runs regardless of session end status)
        # MQTT status reporting removed to reduce verbosity
        
        # Active device monitoring removed to reduce verbosity
    
# Initial memory usage
print_detailed_memory_usage(" (startup)")