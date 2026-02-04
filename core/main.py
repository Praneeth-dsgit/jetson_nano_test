import paho.mqtt.client as mqtt
import json
import time
import numpy as np
from collections import deque
import warnings
import re
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
from typing import Dict, List, Optional, Union, Any, Tuple, Deque
from numpy.typing import NDArray
import signal
import sys
# Handle both relative and absolute imports
try:
    # Try relative imports first (when run as module)
    from .dynamic_model_loader import DynamicModelLoader
    from .data_quality_assessor import SensorDataQualityAssessor
    from .system_health_monitor import SystemHealthMonitor
    from ..communication.mqtt_message_queue import MQTTMessageQueue
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.dynamic_model_loader import DynamicModelLoader
    from core.data_quality_assessor import SensorDataQualityAssessor
    from core.system_health_monitor import SystemHealthMonitor
    from communication.mqtt_message_queue import MQTTMessageQueue

load_dotenv()

# Update config path for new folder structure
try:
    # Try relative path first (when run as module)
    with open('../config/jetson_orin_32gb_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Fall back to absolute path (when run directly)
    import os
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'jetson_orin_32gb_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

# Handle logs directory path for new folder structure
try:
    # Try relative path first (when run as module)
    logs_dir = config['logging']['logs_dir']
    if not os.path.isabs(logs_dir):
        logs_dir = f"../{logs_dir}"
except:
    # Fall back to absolute path (when run directly)
    import os
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'monitoring', 'logs')

# --- Multi-device state ----
# Keep per-device processing context so we can handle multiple devices concurrently
device_contexts = {}
current_mode = "unknown"  # Track current mode (training/game)
prediction_logged_devices = set()  # Track which devices have had their first prediction logged
active_game_devices = set()  # Track unique devices in game mode to scale cache size

def _update_log_filename_if_needed(mode: str) -> None:
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
        try:
            device_info = get_device()
            logger.info(f"Processing Device: {device_info}")
        except:
            logger.info(f"CUDA not requested, using CPU processing")
        
        # Log dynamic model loader info (using actual variables that exist)
        try:
            available_models = model_loader.get_available_player_ids()
            max_available_models = len(available_models)
            logger.info(f"Dynamic Model Loader: Found {max_available_models} available models")
            logger.info(f"Dynamic Model Loader: Available player IDs: {available_models}")
            logger.info(f"Dynamic Model Loader: Cache size {CACHE_SIZE}, Device {MODEL_DEVICE}")
            logger.info(f"Models will be loaded on-demand based on player/device IDs")
        except NameError:
            # Fallback if model_loader not initialized yet
            logger.info(f"Model Discovery: Models directory: {MODELS_DIR}/")
            logger.info(f"Dynamic model loading system will initialize on first device")
        
        logger.info(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
        logger.info(f"Starting test deployment in multi-device mode")
        logger.info(f"MQTT subscription topics: player/+/sensor/data, sensor/data")
        logger.info(f"Connected to MQTT Broker with result code: 0")
        logger.info(f"Subscribed to topics: player/+/sensor/data, sensor/data")
        
        logger.info(f"Log file created with mode: {log_filename}")
        logger.info(f"Mode detected: {mode.upper()}")

def _normalize_device_id(raw_device_id: str) -> str:
    """Normalize device ID to PM### format."""
    raw_str = str(raw_device_id).strip() if raw_device_id is not None else ""
    if not raw_str:
        return "PM000"

    match = re.search(r'(\d+)', raw_str)
    if match:
        return f"PM{int(match.group(1)):03d}"

    return "PM000"

def _init_device_context(device_id_str: str) -> Dict[str, Any]:
    """Create and return a fresh processing context for a device."""
    normalized_device_id = _normalize_device_id(device_id_str)
    # Extract numeric part from device_id (handles both "001" and "PM001" formats)
    device_id_int = None
    try:
        device_id_int = int(normalized_device_id[2:])
    except ValueError:
        device_id_int = None
    
    # Initialize athlete profile without DB; use sensible defaults
    athlete_id_val = device_id_int if device_id_int is not None else 0
    name_val = f"Device_{normalized_device_id}"
    age_val = 25
    weight_val = 70.0
    height_val = 175.0
    gender_val = 1

    context = {
        "device_id": normalized_device_id,
        "athlete_id": athlete_id_val,
        "name": name_val,
        "age": age_val,
        "weight": weight_val,
        "height": height_val,
        "gender": gender_val,
        "hr_rest": 60,
        "hr_max": 220 - age_val,
        "MQTT_PUBLISH_TOPIC": f"{MQTT_TOPIC_PREDICTIONS_PREFIX}/{normalized_device_id}",
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
        "last_trimp_update_time": None,
        "last_data_time": time.time(),
        "last_warning_time": 0,
        "vo2_max_value": "-",
        "session_end_time": None,
        "session_ended": False,
        # TRIMP
        "trimp_buffer": [],
        "total_trimp": 0.0,
    }

    device_contexts[normalized_device_id] = context

    # Ensure a logger exists; set up per first device
    global logger
    try:
        _ = logger  # type: ignore[name-defined]
    except NameError:
        logger = setup_logging(athlete_id_val, normalized_device_id)
    return context

def _load_context_to_globals(ctx: Dict[str, Any]) -> None:
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
    MQTT_PUBLISH_TOPIC = ctx.get("MQTT_PUBLISH_TOPIC", "predictions/000")

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
    last_trimp_update_time = ctx.get("last_trimp_update_time", None)
    last_data_time = ctx.get("last_data_time", time.time())
    last_warning_time = ctx.get("last_warning_time", 0)
    vo2_max_value = ctx.get("vo2_max_value", "-")
    session_end_time = ctx.get("session_end_time", None)
    session_ended = ctx.get("session_ended", False)
    trimp_buffer = ctx.get("trimp_buffer", [])
    total_trimp = ctx.get("total_trimp", 0.0)

def _save_globals_to_context(ctx: Dict[str, Any]) -> None:
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
    ctx["last_trimp_update_time"] = last_trimp_update_time

# Setup logging
def setup_logging(athlete_id: int, device_id: str) -> logging.Logger:
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
def get_memory_status() -> Dict[str, Any]:
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

def print_memory_usage(label: str = "", detailed: bool = False) -> None:
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
                print(f"[ERR] Memory monitoring error: {status['error']}")
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
        print(f"[ERR] Memory monitoring failed: {e}")

def print_detailed_memory_usage(label: str = "") -> None:
    """Print detailed CPU + GPU memory usage with system info (legacy function for compatibility)"""
    print_memory_usage(label, detailed=True)

# Prepare a fallback logger before any device context is created
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Create logs directory if it doesn't exist
    # Handle logs directory path for new folder structure
    try:
        logs_dir = "../monitoring/logs"
    except:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'monitoring', 'logs')
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

# MQTT Broker Configuration
# SUBSCRIBE broker: Local broker for receiving sensor data (from publisher)
MQTT_SUBSCRIBE_BROKER = os.getenv("MQTT_SUBSCRIBE_BROKER", "localhost")
MQTT_SUBSCRIBE_PORT = int(os.getenv("MQTT_SUBSCRIBE_PORT", "1883"))

# PUBLISH broker: Network broker for publishing predictions (to frontend)
MQTT_PUBLISH_BROKER = os.getenv("MQTT_PUBLISH_BROKER", "localhost")
MQTT_PUBLISH_PORT = int(os.getenv("MQTT_PUBLISH_PORT", "1883"))
idle_time = os.getenv("IDLE_TIME", 300)
USE_CUDA = os.getenv("USE_CUDA", "1")

# Enhanced CUDA detection with better error handling
def get_device() -> str:
    """Prefer CUDA automatically when available unless explicitly disabled."""
    try:
        ## Respect explicit disable/force-CPU if provided
        disable_cuda = os.getenv("DISABLE_CUDA", "").lower() in {"1", "true", "yes"}
        force_cpu = (USE_CUDA or "").lower() in {"0", "false"}
        if disable_cuda or force_cpu:
            print("CUDA explicitly disabled; using CPU processing")
            logger.info("CUDA explicitly disabled; using CPU processing")
            return "cpu"

        if torch.cuda.is_available():
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0], device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            print("CUDA is available and working")
            logger.info("CUDA is available and working - GPU processing enabled")
            return "cuda"
        else:
            print("CUDA not available; using CPU processing")
            logger.warning("CUDA not available; using CPU processing")
            return "cpu"
    except Exception as e:
        print(f"CUDA error detected: {e}")
        print("Falling back to CPU processing")
        logger.error(f"CUDA error detected: {e} - Falling back to CPU processing")
        return "cpu"

def create_prediction_lockfile() -> None:
    """Create a lockfile to indicate prediction is running."""
    try:
        with open('.prediction_running.lock', 'w') as f:
            f.write(str(os.getpid()))
        print(f"[INFO] Created prediction lockfile (PID: {os.getpid()})")
    except Exception as e:
        print(f"[WARN] Failed to create prediction lockfile: {e}")

def remove_prediction_lockfile() -> None:
    """Remove the prediction lockfile."""
    try:
        if os.path.exists('.prediction_running.lock'):
            os.remove('.prediction_running.lock')
            print("[INFO] Removed prediction lockfile")
    except Exception as e:
        print(f"[WARN] Failed to remove prediction lockfile: {e}")

DEVICE = get_device()

# Lazy initialization variables - will be initialized on first use
model_loader = None
quality_assessor = None
message_queue = None
health_monitor = None

# Configuration variables (will be set during initialization)
CACHE_SIZE = None
MODEL_DEVICE = None
MODELS_DIR = None

def _initialize_components():
    """Initialize all components lazily (only when needed)."""
    global model_loader, quality_assessor, message_queue, health_monitor
    global CACHE_SIZE, MODEL_DEVICE, MODELS_DIR
    
    # Only initialize if not already initialized
    if model_loader is not None:
        return
    
    # Initialize dynamic model loader instead of loading all models at startup
    print("Initializing dynamic model loader...")
    logger.info("Initializing dynamic model loader for memory-efficient model management")

    # Create dynamic model loader with Jetson Orin 32GB optimized settings
    # Read configuration from jetson_orin_32gb_config.yaml
    CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", str(config.get('model_loading', {}).get('cache_size', 3))))
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", config.get('model_loading', {}).get('device', 'cpu'))
    # Handle models directory path for new folder structure
    try:
        # Try relative path first (when run as module)
        MODELS_DIR = os.getenv("MODEL_DIRECTORY", config.get('model_loading', {}).get('models_directory', '../models/athlete_models_tensors_updated'))
    except:
        # Fall back to absolute path (when run directly)
        # os is already imported at module level, no need to import again
        MODELS_DIR = os.getenv("MODEL_DIRECTORY", config.get('model_loading', {}).get('models_directory', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'athlete_models_tensors_updated')))
    ENABLE_MONITORING = config.get('model_loading', {}).get('enable_memory_monitoring', True)

    model_loader = DynamicModelLoader(
        models_dir=MODELS_DIR,
        cache_size=CACHE_SIZE,
        device=MODEL_DEVICE,
        enable_memory_monitoring=ENABLE_MONITORING
    )

    # Initialize data quality assessor
    print("Initializing data quality assessor...")
    logger.info("Initializing data quality assessor for sensor data reliability monitoring")

    quality_assessor = SensorDataQualityAssessor(
        window_size=30,
        enable_logging=True,
        quality_threshold=0.7
    )

    # Initialize MQTT message queue for reliable delivery
    print("Initializing MQTT message queue...")
    logger.info("Initializing MQTT message queue for reliable message delivery")

    message_queue = MQTTMessageQueue(
        db_path="mqtt_message_queue.db",
        max_retries=5,
        enable_logging=True
    )

# Initialize components immediately when run directly, lazily when imported
if __name__ == "__main__":
    _initialize_components()
    _initialize_health_monitor()

# Global variable to track if publish client is initialized
_publish_client_initialized = False
client = None  # Global MQTT client for fallback publishing

def _initialize_publish_client():
    """
    Initialize MQTT publish client for publishing predictions.
    This function is called at module level to ensure the client is available
    when main.py is imported (e.g., by subscriber.py).
    """
    global _publish_client_initialized, client
    
    # Only initialize once
    if _publish_client_initialized:
        return
    
    try:
        # Log the broker configuration being used
        broker_from_env = os.getenv("MQTT_PUBLISH_BROKER")
        logger.info(f"🔌 Initializing PUBLISH MQTT client")
        logger.info(f"   Broker from env: {broker_from_env}")
        logger.info(f"   Broker being used: {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}")
        
        # Create PUBLISH client for network broker (publishing predictions)
        publish_client = mqtt.Client()
        publish_client.on_connect = on_connect_publish
        publish_client.on_disconnect = on_disconnect_publish
        
        # Add error callback for better diagnostics
        def on_publish_error(client, userdata, error):
            logger.error(f"[ERR] PUBLISH client error: {error}")
        publish_client.on_socket_open = lambda client, userdata, sock: logger.debug("PUBLISH socket opened")
        publish_client.on_socket_close = lambda client, userdata, sock: logger.warning("PUBLISH socket closed")
        
        # Try to connect
        try:
            logger.info(f"   Attempting to connect to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}...")
            result = publish_client.connect(MQTT_PUBLISH_BROKER, MQTT_PUBLISH_PORT, 60)
            if result == mqtt.MQTT_ERR_SUCCESS:
                logger.info("[OK] PUBLISH client connection initiated (will connect asynchronously)")
            else:
                logger.warning(f"[WARN]  PUBLISH client connect() returned code: {result}")
                logger.warning(f"   Error codes: 0=Success, 1=Connection refused, 2=Identifier rejected, 3=Server unavailable, 4=Bad credentials, 5=Not authorized")
            publish_client.loop_start()
            
            # Wait longer for connection to establish (async connection)
            time.sleep(2)
            if publish_client.is_connected():
                logger.info("[OK] PUBLISH client is connected")
            else:
                logger.warning(f"[WARN]  PUBLISH client not yet connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}")
                logger.warning("   This is normal - connection happens asynchronously. Messages will be queued until connection is established.")
                logger.warning(f"   Verify broker is running: Test-NetConnection -ComputerName {MQTT_PUBLISH_BROKER} -Port {MQTT_PUBLISH_PORT}")
        except Exception as e:
            logger.error(f"[ERR] Failed to connect to PUBLISH MQTT broker ({MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}): {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
            logger.warning("[WARN]  Predictions will be queued and published when broker becomes available")
            # Still start the loop - it will retry connection
            publish_client.loop_start()
        
        # Store publish_client globally for fallback publishing
        client = publish_client
        
        # Set the client in message queue immediately (will be updated when connected)
        # This allows the queue to start processing once connection is established
        # Make sure message_queue is initialized first
        if message_queue is None:
            logger.warning("[WARN]  Message queue not initialized yet, initializing components...")
            _initialize_components()
        
        if message_queue is not None:
            message_queue.set_mqtt_client(publish_client)
            logger.info("[OK] Message queue configured with publish client")
            
            # Check connection status after a moment
            time.sleep(1)
            if publish_client.is_connected():
                logger.info("[OK] PUBLISH client is connected - ready to publish")
            else:
                logger.warning(f"[WARN]  PUBLISH client not yet connected - messages will be queued")
                logger.warning(f"   Connection will be established asynchronously")
                logger.warning(f"   Check broker accessibility: {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}")
        else:
            logger.error("[ERR] Message queue is still None after initialization attempt")
        
        _publish_client_initialized = True
        
        logger.info(f"[OK] PUBLISH client initialized (broker: {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT})")
        logger.info(f"   Connection status will be updated when broker connects")
        
    except Exception as e:
        logger.error(f"[ERR] Error initializing PUBLISH client: {e}")
        logger.warning("[WARN]  Predictions will be queued but may not be published until client is initialized")

# Initialize publish client only when main.py is run directly, not when imported
# This allows subscriber.py to import process_raw_sensor_data without initializing MQTT clients
if __name__ == "__main__":
    _initialize_publish_client()
else:
    # When imported, initialize lazily (only when process_raw_sensor_data is called)
    # This allows subscriber.py to run independently
    _publish_client_initialized = False

def _initialize_health_monitor():
    """Initialize health monitor lazily."""
    global health_monitor
    
    if health_monitor is not None:
        return
    
    # Initialize system health monitor
    print("Initializing system health monitor...")
    logger.info("Initializing system health monitor for comprehensive system monitoring")
    
    health_monitor = SystemHealthMonitor(
        db_path="system_health.db",
        collection_interval=60,  # Collect metrics every minute
        history_retention_days=7,  # Keep 7 days of history
        enable_logging=True
    )

    # Add alert callback for health monitoring
    def health_alert_callback(alert: Any) -> None:
        """
        Callback function for system health alerts.
        
        Args:
            alert: HealthAlert object from SystemHealthMonitor
        """
        logger.warning(f"System Health Alert: {alert.message}")
        print(f"🚨 System Alert: {alert.message}")

    health_monitor.add_alert_callback(health_alert_callback)

    # Start system health monitoring
    health_monitor.start_monitoring()
    logger.info("System health monitoring started")

# Initialize health monitor immediately when run directly, lazily when imported
if __name__ == "__main__":
    _initialize_components()
    _initialize_health_monitor()
    
    # Get available models info (only when run directly)
    available_models = model_loader.get_available_player_ids()
    max_available_models = len(available_models)
    
    # Get MODELS_DIR for display
    try:
        MODELS_DIR = os.getenv("MODEL_DIRECTORY", config.get('model_loading', {}).get('models_directory', '../models/athlete_models_tensors_updated'))
    except:
        MODELS_DIR = os.getenv("MODEL_DIRECTORY", config.get('model_loading', {}).get('models_directory', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'athlete_models_tensors_updated')))
    
    print(f"Dynamic model loader initialized (Jetson Nano 4GB optimized)")
    print(f"   📁 Models directory: {MODELS_DIR}/")
else:
    # When imported, set defaults (will be initialized lazily)
    available_models = []
    max_available_models = 0
    # Don't print startup messages when imported

    # =============================================================================
    # STARTUP STATUS SUMMARY
    # =============================================================================
    print(f"\n{'='*60}")
    print(f"JETSON NANO ML PREDICTION ENGINE - STARTUP COMPLETE")
    print(f"{'='*60}")
    print(f"System Configuration:")
    print(f"   - Mode: Dynamic Model Loading (Memory Optimized)")
    if CACHE_SIZE is not None:
        print(f"   - Cache Size: {CACHE_SIZE} models (LRU eviction)")
    if MODEL_DEVICE is not None:
        print(f"   - Device: {MODEL_DEVICE} (GPU acceleration enabled)")
    print(f"   - Available Models: {max_available_models} players")
    try:
        ENABLE_MONITORING = config.get('model_loading', {}).get('enable_memory_monitoring', True)
        print(f"   - Memory Monitoring: {'Enabled' if ENABLE_MONITORING else 'Disabled'}")
    except:
        pass
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
HR_MAX_BPM = 220                        # Elite athlete physiological max; clamp displayed HR to this

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
last_trimp_update_time = None
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

# Simple MQTT topic tracking (initialized early for use in callbacks)
unique_mqtt_topics = set()

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
# Publishing topic prefixes from .env (topic = prefix/device_id; context may overwrite MQTT_PUBLISH_TOPIC)
MQTT_TOPIC_PREDICTIONS_PREFIX = os.getenv("MQTT_PUBLISH_TOPIC_PREDICTIONS", "predictions")
LPS_POSITION_TOPIC_PREFIX = os.getenv("MQTT_PUBLISH_TOPIC_LPS", "lps")
MQTT_PUBLISH_TOPIC = "predictions"  # default; often overwritten by context with full topic per device


def _publish_position_immediate(device_id: str, x: float, y: float) -> None:
    """
    Publish position to lps/{device_id} immediately so the visualization
    can update the plot without waiting for the full ML prediction pipeline.
    Reduces perceived latency when using real sensor data.
    """
    global client
    if client is None or not (hasattr(client, "is_connected") and client.is_connected()):
        return
    try:
        topic = f"{LPS_POSITION_TOPIC_PREFIX}/{_normalize_device_id(device_id)}"
        x_val, y_val = round(x, 4), round(y, 4)
        payload = json.dumps({
            "device_id": device_id,
            "positions": {"x": x_val, "y": y_val},
            "x": x_val,
            "y": y_val
        })
        client.publish(topic, payload, qos=0)  # qos=0 for lowest latency
        logger.info(f"[OK] Publishing to {topic} (lps/+) completed for device {device_id}: x={x:.2f}, y={y:.2f}")
    except Exception as e:
        logger.debug(f"Could not publish immediate position: {e}")


# --- Model input dimension cache and helpers ---
model_input_dims = {}

def _safe_prepare_input(base_features: Union[List[float], NDArray[np.float32]], required_dim: int) -> NDArray[np.float32]:
    """Pad or truncate feature vector to required_dim."""
    arr = np.asarray(base_features, dtype=np.float32)
    if arr.size < required_dim:
        padded = np.zeros(required_dim, dtype=np.float32)
        padded[:arr.size] = arr
        return padded
    elif arr.size > required_dim:
        return arr[:required_dim]
    return arr

def _infer_input_dim_for_model(model: Any, base_len: int, max_try: int = 64) -> int:
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

def predict_with_adaptive_input(model: Any, base_features: List[float]) -> NDArray[np.float32]:
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
            error_msg = f"Model prediction failed: {e}"
            logger.error(error_msg)
            print(f"[ERR] {error_msg}")
            
            # Check for specific CUDA/Hummingbird errors
            error_str = str(e).lower()
            is_cuda_error = ("cuda" in error_str or "gpu" in error_str or 
                           "indices element is out of data bounds" in error_str or
                           "out of bounds" in error_str)
            
            if DEVICE == "cuda" and is_cuda_error:
                try:
                    fallback_msg = "Attempting CPU fallback for prediction"
                    logger.warning(f"{fallback_msg} - CUDA error: {e}")
                    print(f"[WARN] {fallback_msg}")
                    
                    # Create a temporary CPU copy of the model
                    cpu_model = model
                    if hasattr(cpu_model, 'to'):
                        cpu_model = cpu_model.to('cpu')
                    result = cpu_model.predict(x_batch)
                    
                    success_msg = "CPU fallback successful"
                    logger.info(success_msg)
                    print(f"[OK] {success_msg}")
                    return result
                except Exception as cpu_e:
                    cpu_error_msg = f"CPU fallback also failed: {cpu_e}"
                    logger.error(cpu_error_msg)
                    print(f"[ERR] {cpu_error_msg}")
                    
                    # Log the fallback failure and return default
                    logger.warning("Returning default heart rate due to prediction failure")
                    return np.array([60.0])  # Default heart rate
            else:
                # For non-CUDA errors, log and return default
                logger.error(f"Non-CUDA prediction error: {e}")
                logger.warning("Returning default heart rate due to prediction failure")
                return np.array([60.0])
                
    except Exception as e:
        error_msg = f"Error in predict_with_adaptive_input: {e}"
        logger.error(error_msg)
        print(f"[ERR] {error_msg}")
        logger.warning("Returning default heart rate due to critical prediction error")
        return np.array([60.0])  # Default heart rate

def compute_window_features(
    ax: Deque[float],
    ay: Deque[float],
    az: Deque[float],
    gx: Deque[float],
    gy: Deque[float],
    gz: Deque[float],
    fs: float = FS_HZ
) -> Optional[Dict[str, float]]:
    """
    Compute windowed features from sensor data.
    
    Args:
        ax: X-axis acceleration window
        ay: Y-axis acceleration window
        az: Z-axis acceleration window
        gx: X-axis gyroscope window
        gy: Y-axis gyroscope window
        gz: Z-axis gyroscope window
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of computed features, or None if insufficient data
    """
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
        def stats(arr: NDArray[np.float32]) -> Tuple[float, float, float, float, float]:
            """
            Calculate basic statistics for an array.
            
            Args:
                arr: Input array of float values
                
            Returns:
                Tuple of (mean, std, min, max, range)
            """
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
def butter_bandpass_filter(
    data: Union[List[float], NDArray[np.float32]],
    low_cutoff: float = 0.3,
    high_cutoff: float = 4.5,
    fs: float = 10.0,
    order: int = 2
) -> NDArray[np.float32]:
    """
    Apply Butterworth bandpass filter to sensor data.
    
    Filters data to remove noise outside the specified frequency range,
    commonly used for processing accelerometer and gyroscope data.
    
    Args:
        data: Array/list of sensor data values to filter
        low_cutoff: Lower cutoff frequency in Hz (default: 0.3)
        high_cutoff: Upper cutoff frequency in Hz (default: 4.5)
        fs: Sampling frequency in Hz (default: 10.0)
        order: Filter order (default: 2)
        
    Returns:
        Filtered data array with same length as input
    """
    nyq = 0.5 * fs
    low = low_cutoff / nyq
    high = high_cutoff/ nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def calculate_rmssd(hr_values: Union[List[float], NDArray[np.float32]]) -> float:
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

def estimate_vo2_max(
    age: int,
    gender: int,
    current_hr: float,
    hrv: float,
    hr_rest: int = 60,
    hr_max: Optional[int] = None
) -> float:
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

def training_energy_expenditure(velocity: float, duration_s: float, mass_kg: float) -> float:
    """
    Calculate training energy expenditure based on velocity and body mass.
    
    Args:
        velocity: Velocity in m/s
        duration_s: Duration in seconds
        mass_kg: Body mass in kilograms
        
    Returns:
        Energy expenditure in kilocalories (rounded to 2 decimal places)
    """
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

def parse_sensor_payload(payload: bytes) -> Optional[Dict[str, Any]]:
    """
    Parse sensor data from MQTT payload.
    
    Attempts to parse JSON payload first, falls back to simple format parsing
    if JSON parsing fails.
    
    Args:
        payload: Raw bytes from MQTT message
        
    Returns:
        Parsed dictionary with sensor data, or None if parsing fails
    """
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

def calculate_stress(
    hr: float,
    hrv: float,
    acc_mag: float,
    gyro_mag: float,
    age: int,
    gender: int,
    hr_rest: int = 60,
    hr_max: int = 200
) -> float:
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
        hrv_norm = 0.5  # No HRV data = neutral (don't assume high stress)
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
    
    # Gentler sigmoid curve; cap at 95% so we don't show 100% from model/clamped inputs
    stress_percent = 100 * (1 / (1 + np.exp(-4 * (score - 0.3))))
    stress_percent = min(95.0, stress_percent)
    
    # Clamp to valid range
    return round(max(0, min(100, stress_percent)), 1)

def calculate_trimp(
    hr_avg: float,
    hr_rest: int,
    hr_max: int,
    duration_min: float,
    gender: str = "male"
) -> float:
    """
    Calculate TRIMP (Training Impulse) for a single session.
    
    Args:
        hr_avg: Average heart rate (BPM)
        hr_rest: Resting heart rate (BPM)
        hr_max: Maximum heart rate (BPM)
        duration_min: Duration in minutes
        gender: Gender string ("male" or "female")
        
    Returns:
        TRIMP value
        
    Raises:
        ValueError: If gender is not "male" or "female"
    """
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

def get_trimp_zone(total_trimp: float) -> Tuple[str, str]:
    """
    Determine TRIMP training zone based on total TRIMP value.
    
    Args:
        total_trimp: Total TRIMP value for the session
        
    Returns:
        Tuple of (zone_name, zone_description)
    """
    if total_trimp < 50:
        return "Light", "Recovery/Warm-up intensity"
    elif total_trimp < 150:
        return "Moderate", "Standard training session"
    elif total_trimp < 300:
        return "High", "Intense training session"
    else:
        return "Very High", "Very intense session"

def get_recovery_recommendations(total_trimp: float, stress_percent: float) -> Tuple[str, List[str]]:
    """
    Generate recovery recommendations based on TRIMP and other metrics.
    
    Args:
        total_trimp: Total TRIMP value for the session
        stress_percent: Average stress percentage
        
    Returns:
        Tuple of (recovery_time_string, list_of_recommendations)
    """
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

def get_training_recommendations(trimp_zone: str, stress_percent: float) -> List[str]:
    """
    Generate detailed training recommendations based on TRIMP zone and current metrics.
    
    Args:
        trimp_zone: TRIMP zone string ("Light", "Moderate", "High", "Very High")
        stress_percent: Current stress percentage
        
    Returns:
        List of recommendation strings
    """
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

def generate_session_summary() -> Optional[Dict[str, Any]]:
    """
    Generate end-of-session summary with recommendations.
    
    Returns:
        Dictionary containing session summary data, or None if session not ended
    """
    global session_end_time, session_start_time, total_trimp, stress_buffer, g_impact_count
    
    if session_end_time is None or session_start_time is None:
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


def process_data() -> None:
    """
    Main data processing pipeline for sensor data.
    
    Processes incoming sensor data through multiple stages:
    1. Data quality assessment
    2. Sensor fusion (Madgwick filter)
    3. Feature engineering
    4. ML prediction (if in game mode)
    5. Health metrics calculation (stress, VO2, TRIMP)
    6. Output generation and MQTT publishing
    
    Reads from global sensor_data and updates global buffers and metrics.
    This function is called by the MQTT message handler for each incoming message.
    
    Note:
        This function uses global variables extensively. Consider refactoring
        to use a class-based approach for better encapsulation.
    """
    global session_start_time, last_vo2_update_time, last_trimp_update_time, vo2_max_value
    global quaternion
    global g_impact_count
    global hrv_rmssd
    global trimp_buffer, total_trimp, current_trimp
    global sensor_data, session_end_time, session_ended

    acc = sensor_data["acc"]
    gyro = sensor_data["gyro"]
    magno = sensor_data['magno']
    if not (acc and gyro and magno):
        logger.warning("Missing sensor data - skipping processing")
        return
    
    # Assess data quality before processing
    try:
        hr_data = sensor_data.get("heart_rate_bpm", 0.0)
        quality_report = quality_assessor.assess_sensor_data_quality(
            acc_data=acc,
            gyro_data=gyro,
            hr_data=hr_data,
            mag_data=magno,
            timestamp=datetime.now()
        )
        
        # Log quality issues if any
        if quality_report['overall_quality_score'] < 0.7:
            logger.warning(f"Low data quality detected: {quality_report['overall_quality_score']:.3f} - {quality_report['quality_status']}")
            for recommendation in quality_report['recommendations']:
                logger.info(f"Quality recommendation: {recommendation}")
        
        # Skip processing if quality is critically low
        if quality_report['overall_quality_score'] < 0.3:
            logger.error("Critical data quality issues - skipping data processing")
            return
            
    except Exception as e:
        logger.error(f"Data quality assessment failed: {e}")
        # Continue processing even if quality assessment fails

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
            # Extract numeric part from device_id (handles both "001" and "PM001" formats)
            device_id_str = str(device_id)
            match = re.search(r'(\d+)', device_id_str)
            if match:
                device_idx = int(match.group(1))
            else:
                device_idx = int(device_id_str.lstrip("0") or "0")
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


    # Clamp HR to physiological range (30-220 BPM) for buffers so RMSSD and stress get valid inputs.
    # Model can output out-of-range values (e.g. 500+ bpm); using raw values would zero out HRV and max stress.
    hr_for_buffers = max(30.0, min(220.0, float(predicted_hr)))
    hr_buffer.append(hr_for_buffers)
    hrv_hr_buffer.append(hr_for_buffers)

    # Display activity metrics for every prediction (moved above heart rate)
    current_acc_display = acc_buffer[-1] if len(acc_buffer) > 0 else (acc_mag_buffer[-1] if len(acc_mag_buffer) > 0 else 0.0)
    print(f"Activity: Velocity: {round(new_vel, 2)} m/s | Distance: {round(new_dist, 2)} m | Acc: {round(current_acc_display, 2)} m/s²")

    # Show heart rate and HRV information for every prediction
    # Get actual HR if available (training mode)
    actual_hr = sensor_data.get("heart_rate_bpm") if mode == "training" else None
    
    # Display heart rate information for every prediction
    if actual_hr is not None and mode == "training":
        print(f"Heart Rate: {actual_hr} bpm (actual) | Mode: {mode}")
    elif mode == "game":
        max_hr = min(float(predicted_hr), HR_MAX_BPM)
        print(f"Heart Rate: {max_hr:.1f} bpm (predicted) | Mode: {mode}")
    else:
        print(f"Heart Rate: - bpm | Mode: {mode}")

    # Show cache info every prediction only in game mode
    if mode == "game":
        try:
            cache_info = model_loader.get_cache_info()
            print(
                f"🧠 Cache: {cache_info['cache_size']}/{cache_info['max_cache_size']} | "
                f"Hit: {cache_info['cache_hit_rate']:.1%} | "
                f"Loaded: {cache_info['models_loaded']} | "
                f"Evicted: {cache_info['models_evicted']} | "
                f"Cached: {cache_info['cached_models']}"
            )
        except Exception:
            pass
    
    # Display HRV (single source): 10-second window if available
    if len(hrv_hr_buffer) >= 5:
        current_hrv = calculate_rmssd(hrv_hr_buffer)
        print(f"HRV (RMSSD, 10s): {current_hrv:.1f} ms")
    else:
        print(f"HRV (RMSSD): Calculating... (need {5 - len(hrv_hr_buffer)} more readings)")
    
    # Show comprehensive health metrics every 5 predictions for better visibility
    if len(hr_buffer) % 5 == 0 and len(hr_buffer) > 0:  # Show detailed metrics every 5 predictions
        # Display stress if available (use freshly computed HRV so stress matches displayed HRV)
        if len(hr_buffer) >= ROLLING_WINDOW_SIZE and len(hrv_hr_buffer) >= 5:
            stress_hrv = calculate_rmssd(hrv_hr_buffer)
            current_stress = calculate_stress(np.mean(hr_buffer), stress_hrv, np.mean(acc_buffer), np.mean(gyro_buffer), age, gender, hr_rest, hr_max)
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

    # VO2 max: show as soon as we have enough HR data (5+ samples), then refresh every 5 min
    if len(hr_buffer) >= 5:
        if last_vo2_update_time is None or (now - last_vo2_update_time >= 300):
            # Use fresh HRV when available so VO2 estimate matches current HRV (same as stress)
            vo2_hrv = calculate_rmssd(hrv_hr_buffer) if len(hrv_hr_buffer) >= 5 else hrv_rmssd
            vo2_max_value = estimate_vo2_max(age, gender, np.mean(hr_buffer), vo2_hrv, hr_rest, hr_max)
            last_vo2_update_time = now
            logger.info(f"VO2 Max Updated: {vo2_max_value} ml/kg/min (HR: {np.mean(hr_buffer):.0f}, HRV: {vo2_hrv:.1f})")
    else:
        vo2_max_value = "-"


    # --- Total Energy Expenditure Calculation ---
    tte = training_energy_expenditure(new_vel, DT, weight)
    TEE_buffer.append(tte)
    active_tee = round(sum(TEE_buffer), 2)

    print(f"Energy: {tte:.2f} kcal | Total: {active_tee} kcal")

    # --- TRIMP Calculation ---
    if len(hr_buffer) >= 5:                             # Need at least 5 HR readings for meaningful TRIMP
        hr_avg = np.mean(list(hr_buffer)[-5:])          # Use last 5 HR readings
        gender_str = "male" if gender == 1 else "female"

        # Ensure session start and last update timestamps are initialized
        if session_start_time is None:
            session_start_time = now
            # initialize last_trimp_update_time to avoid counting whole session repeatedly
            last_trimp_update_time = now

        # Compute delta minutes since last TRIMP update (incremental TRIMP)
        if last_trimp_update_time is None:
            delta_min = elapsed_time / 60.0
        else:
            delta_min = max(0.0, (now - last_trimp_update_time) / 60.0)

        # Only compute if there's a measurable delta
        if delta_min > 0:
            current_trimp = calculate_trimp(hr_avg, hr_rest, hr_max, delta_min, gender_str)
            trimp_buffer.append(current_trimp)
            # update last update timestamp
            last_trimp_update_time = now
        else:
            current_trimp = 0.0

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
        # Handle prediction outputs directory path for new folder structure
        try:
            base_output_dir = "../data/prediction_outputs"
        except:
            base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prediction_outputs')
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
        #--"hr_rest": hr_rest,
        #--"hr_max": hr_max,
        # Position (x, y) is published only to lps/+ for low-latency visualization; not included in predictions/+
        # Include engineered features snapshot for this window
        #--"window_features": window_features if window_features is not None else {},
        # Data quality assessment
        #--"data_quality": quality_report if 'quality_report' in locals() else None,
        # Only include the prediction for this specific device's model (for game mode)
        "model_prediction": {
            "model_id": device_idx if mode == "game" else None,
            "predicted_hr": predicted_hr if mode == "game" else None,
            "model_loaded_dynamically": True,
            "cache_info": model_loader.get_cache_info() if mode == "game" else None
        } if mode == "game" else None
    }

    # When saving realtime output, use organized folder structure:
    # Handle prediction outputs directory path for new folder structure
    try:
        base_output_dir = "../data/prediction_outputs"
    except:
        base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prediction_outputs')
    player_folder = f"A{str(athlete_id)}_{str(name)}"
    full_player_path = os.path.join(base_output_dir, player_folder)
    os.makedirs(full_player_path, exist_ok=True)

    with open(os.path.join(full_player_path, f"A{athlete_id}_D{device_id}_realtime_output.json"), "w") as f:
        json.dump(output, f, indent=2)

    # MQTT publishing with reliable message queue
    publish_topic = f"{MQTT_TOPIC_PREDICTIONS_PREFIX}/{_normalize_device_id(device_id)}"
    try:
        # Queue message for reliable delivery
        message_id = message_queue.queue_message(
            topic=publish_topic,
            payload=json.dumps(output),
            qos=1,
            retain=False
        )
        logger.info(f"[OK] Publishing to {publish_topic} (predictions/+) completed.")
        
        if device_id not in prediction_logged_devices:
            # Check connection status for logging
            queue_client = message_queue.mqtt_client if hasattr(message_queue, 'mqtt_client') else None
            is_connected = queue_client.is_connected() if queue_client else False
            status = "[OK] Connected" if is_connected else "[WARN] Queued (waiting for connection)"
            logger.info(f"Queued prediction message {message_id} for reliable delivery to topic: {publish_topic} | Status: {status}")
            prediction_logged_devices.add(device_id)
            
    except Exception as e:
        logger.error(f"Failed to queue MQTT message: {e}")
        # Fallback to direct publishing if queue fails
        try:
            if client and hasattr(client, "is_connected") and client.is_connected():
                client.publish(publish_topic, json.dumps(output), qos=1)
                logger.info(f"[OK] Publishing to {publish_topic} (predictions/+) completed (fallback).")
                logger.warning("Used fallback direct MQTT publishing")
            else:
                logger.error("Fallback publish skipped: MQTT client not connected")
        except Exception as fallback_e:
            logger.error(f"Fallback MQTT publishing also failed: {fallback_e}")
    
    # Memory monitoring and cache status (every 100 data points to reduce noise)
    if len(hr_buffer) % 100 == 0 and len(hr_buffer) > 0:
        print_memory_usage(" (periodic check)")
        
        # Check memory pressure and auto-manage if needed
        try:
            pressure_info = model_loader.check_memory_pressure()
            if pressure_info["pressure_level"] in ["high", "critical"]:
                logger.warning(f"Memory pressure detected: {pressure_info['pressure_level']}")
                for recommendation in pressure_info["recommendations"]:
                    logger.info(f"Memory recommendation: {recommendation}")
                
                # Auto-manage memory if pressure is high
                if model_loader.auto_manage_memory():
                    logger.info("Memory auto-management triggered")
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
        
        # Show dynamic model loader cache status
        cache_info = model_loader.get_cache_info()
        if cache_info["total_requests"] > 0:
            print(f"🧠 Model Cache: {cache_info['cache_size']}/{cache_info['max_cache_size']} models, "
                  f"Hit rate: {cache_info['cache_hit_rate']:.1%}, "
                  f"Requests: {cache_info['total_requests']}")
            print(f"Cache Details: Loaded: {cache_info['models_loaded']}, "
                  f"Evicted: {cache_info['models_evicted']}, "
                  f"Avg load time: {cache_info['average_load_time']:.2f}s")
            
            # Show memory status if available
            if "memory_status" in cache_info and "cpu" in cache_info["memory_status"]:
                cpu_mem = cache_info["memory_status"]["cpu"]
                print(f"💾 Memory: Process {cpu_mem['process_memory_mb']}MB ({cpu_mem['process_memory_percent']:.1f}%), "
                      f"System {cpu_mem['system_usage_percent']:.1f}% used")
                
                if "gpu" in cache_info["memory_status"] and "usage_percent" in cache_info["memory_status"]["gpu"]:
                    gpu_mem = cache_info["memory_status"]["gpu"]
                    print(f"🎮 GPU: {gpu_mem['usage_percent']:.1f}% used ({gpu_mem['allocated_mb']:.1f}MB allocated)")
        
        # Show message queue statistics
        try:
            queue_stats = message_queue.get_queue_stats()
            if "total_messages" in queue_stats and queue_stats["total_messages"] > 0:
                print(f"📨 Message Queue: {queue_stats['queue_size']} pending, "
                      f"{queue_stats['stats']['messages_sent']} sent, "
                      f"{queue_stats['stats']['messages_delivered']} delivered, "
                      f"{queue_stats['stats']['messages_failed']} failed")
        except Exception as e:
            logger.error(f"Failed to get message queue stats: {e}")
        
        # Show system health status
        try:
            health_status = health_monitor.get_current_health_status()
            if "overall_status" in health_status:
                status_emoji = {
                    "healthy": "ok",
                    "warning": "warn", 
                    "critical": "🚨",
                    "unknown": "❓"
                }.get(health_status["overall_status"], "❓")
                
                print(f"{status_emoji} System Health: {health_status['overall_status'].upper()} "
                      f"(Critical: {health_status.get('critical_metrics', 0)}, "
                      f"Warning: {health_status.get('warning_metrics', 0)}, "
                      f"Healthy: {health_status.get('healthy_metrics', 0)})")
                
                # Show uptime
                uptime_hours = health_status.get('uptime_hours', 0)
                print(f"⏱️ Health Monitor Uptime: {uptime_hours:.1f} hours")
        except Exception as e:
            logger.error(f"Failed to get system health status: {e}")

# --- MQTT callbacks ---
def on_connect_subscribe(client: mqtt.Client, userdata: Any, flags: Dict[str, Any], rc: int) -> None:
    """
    MQTT connection callback handler for SUBSCRIBE client (local broker).
    
    Called automatically when connection to local MQTT broker is established.
    Subscribes to sensor data topics from publisher.
    
    Args:
        client: MQTT client instance (subscribe client)
        userdata: User-defined data (not used)
        flags: Connection flags dictionary
        rc: Result code (0 = success)
    """
    global mqtt_connected, mqtt_last_connect_time, mqtt_reconnect_attempts
    mqtt_last_connect_time = time.time()
    
    if rc == 0:
        mqtt_connected = True
        logger.info(f"Connected to SUBSCRIBE MQTT Broker ({MQTT_SUBSCRIBE_BROKER}:{MQTT_SUBSCRIBE_PORT}) with result code: {rc}")
        # Subscribe to all per-player topics and keep legacy topic for compatibility
        client.subscribe("player/+/sensor/data")
        client.subscribe("sensor/data")
        logger.info("Subscribed to player/+/sensor/data and sensor/data topics")
        mqtt_reconnect_attempts = 0  # Reset counter on successful connection
    else:
        mqtt_connected = False
        logger.error(f"Failed to connect to SUBSCRIBE MQTT Broker with result code: {rc}")
        mqtt_reconnect_attempts += 1

def on_connect_publish(mqtt_client: mqtt.Client, userdata: Any, flags: Dict[str, Any], rc: int) -> None:
    """
    MQTT connection callback handler for PUBLISH client (network broker).
    
    Called automatically when connection to network MQTT broker is established.
    Sets up message queue for publishing predictions.
    
    Args:
        mqtt_client: MQTT client instance (publish client)
        userdata: User-defined data (not used)
        flags: Connection flags dictionary
        rc: Result code (0 = success)
    """
    global client
    if rc == 0:
        logger.info(f"[OK] Connected to PUBLISH MQTT Broker ({MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}) with result code: {rc}")
        # Set up message queue with publish client (ensure it's set even if already set)
        message_queue.set_mqtt_client(mqtt_client)
        # Update global client reference
        client = mqtt_client
        
        # Check queue status
        try:
            queue_stats = message_queue.get_queue_stats()
            pending = queue_stats.get('status_counts', {}).get('pending', 0)
            sent = queue_stats.get('status_counts', {}).get('sent', 0)
            if pending > 0:
                logger.info(f"   {pending} queued messages will be published now")
            if sent > 0:
                logger.info(f"   {sent} messages already sent")
        except Exception as e:
            logger.debug(f"Could not get queue stats: {e}")
        
        logger.info("[OK] MQTT message queue configured with publish client - messages will now be published")
        
        # Force immediate processing attempt
        try:
            # Trigger message processing by checking if there are pending messages
            if hasattr(message_queue, '_process_batch'):
                logger.debug("Triggering message queue processing")
        except:
            pass
    else:
        logger.error(f"[ERR] Failed to connect to PUBLISH MQTT Broker ({MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}) with result code: {rc}")
        logger.error(f"   Connection error codes: 1=Connection refused, 2=Identifier rejected, 3=Server unavailable, 4=Bad credentials, 5=Not authorized")
        logger.warning("[WARN]  Messages will be queued and published when broker connection is established")

def on_disconnect_subscribe(client: mqtt.Client, userdata: Any, rc: int) -> None:
    """
    MQTT disconnection callback handler for SUBSCRIBE client.
    
    Called automatically when connection to local MQTT broker is lost.
    Attempts automatic reconnection with exponential backoff.
    
    Args:
        client: MQTT client instance (subscribe client)
        userdata: User-defined data (not used)
        rc: Result code (0 = normal disconnect)
    """
    global mqtt_connected, mqtt_last_disconnect_time, mqtt_reconnect_attempts
    mqtt_connected = False
    mqtt_last_disconnect_time = time.time()    
    logger.warning(f"Disconnected from SUBSCRIBE MQTT Broker with result code: {rc}")
    
    while not mqtt_connected:
        try:
            mqtt_reconnect_attempts += 1
            # Reconnection attempt logged to file only
            client.reconnect()
            logger.info("Successfully reconnected to SUBSCRIBE MQTT Broker")
            break
        except Exception as e:
            logger.warning(f"Reconnection attempt failed: {e}, retrying in 5 seconds...")
            time.sleep(5)

def on_disconnect_publish(client: mqtt.Client, userdata: Any, rc: int) -> None:
    """
    MQTT disconnection callback handler for PUBLISH client.
    
    Called automatically when connection to publish MQTT broker is lost.
    Clears the message queue client so it knows to wait for reconnection.
    
    Args:
        client: MQTT client instance (publish client)
        userdata: User-defined data (not used)
        rc: Result code (0 = normal disconnect)
    """
    if rc == 0:
        logger.info("ℹ️  Disconnected from PUBLISH MQTT Broker (normal disconnect)")
    else:
        logger.warning(f"[WARN]  Unexpected disconnect from PUBLISH MQTT Broker (code: {rc})")
    
    # Don't clear the client - let it reconnect and the queue will resume publishing
    logger.info("Messages will continue to be queued and published when connection is restored")
    """
    MQTT disconnection callback handler for PUBLISH client.
    
    Called automatically when connection to network MQTT broker is lost.
    
    Args:
        client: MQTT client instance (publish client)
        userdata: User-defined data (not used)
        rc: Result code (0 = normal disconnect)
    """
    logger.warning(f"Disconnected from PUBLISH MQTT Broker with result code: {rc}")

def on_message_subscribe(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
    """
    MQTT message callback handler.
    
    Called automatically when a message is received on subscribed topics.
    Parses sensor data, initializes device context if needed, and triggers
    data processing pipeline.
    
    Args:
        client: MQTT client instance
        userdata: User-defined data (not used)
        msg: MQTT message object containing topic and payload
        
    Topics handled:
        - player/{device_id}/sensor/data
        - sensor/data (legacy)
        
    Side effects:
        - Updates device_contexts dictionary
        - Calls process_data() for each message
        - Updates global sensor_data variable
    """
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
        device_id_str = _normalize_device_id(device_id_str)
        
        # Track unique MQTT topics
        unique_mqtt_topics.add(topic)

        # Show that we received sensor data (less verbose)
        athlete_id = parsed_data.get("athlete_id", "unknown")
        print(f"📥 Player {athlete_id} (Device {device_id_str}) - New sensor data")
        
        # Log device activity and detect mode
        mode = parsed_data.get("mode", "game")
        _update_log_filename_if_needed(mode)
        # Dynamically scale model cache size based on active game-mode devices
        try:
            if mode == "game":
                active_game_devices.add(device_id_str)
                desired_cache_size = len(active_game_devices)
                # Bound by number of available models
                if desired_cache_size > model_loader.cache_size:
                    new_cache_size = min(desired_cache_size, max_available_models)
                    if new_cache_size != model_loader.cache_size:
                        logger.info(
                            f"Adjusting model cache size from {model_loader.cache_size} to {new_cache_size} (active game devices: {desired_cache_size})"
                        )
                        model_loader.cache_size = new_cache_size
                        # Print cache info to console after resizing
                        cache_info = model_loader.get_cache_info()
                        print(
                            f"🧠 Cache: {cache_info['cache_size']}/{cache_info['max_cache_size']} | "
                            f"Hit: {cache_info['cache_hit_rate']:.1%} | "
                            f"Loaded: {cache_info['models_loaded']} | "
                            f"Evicted: {cache_info['models_evicted']} | "
                            f"Cached: {cache_info['cached_models']}"
                        )
        except Exception:
            # Do not interrupt message processing if scaling fails
            pass
        
        # Only log device activity on first activation, not every message
        if device_id_str not in device_contexts:
            logger.info(f"Device {device_id_str} (Player {athlete_id}) - First activation, Mode: {mode}")
            logger.info(f"Device {device_id_str} - MQTT prediction topic: predictions/{device_id_str}")

        # Get or init per-device context and sync into globals
        if device_id_str not in device_contexts:
            _init_device_context(device_id_str)
        ctx = device_contexts[device_id_str]
        # Persist last seen mode for this device to support mode-specific summaries
        try:
            ctx["last_mode"] = mode
        except Exception:
            pass
        _load_context_to_globals(ctx)
        
        # Optionally update athlete profile from payload if provided
        name_in = parsed_data.get("name")
        age_in = parsed_data.get("age")
        weight_in = parsed_data.get("weight")
        height_in = parsed_data.get("height")
        gender_in = parsed_data.get("gender")          ## 'M'/'F' or 1/0
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
        
        # Extract and update position from parsed_data (x, y from LPS or publisher)
        global latest_position
        pos_x = parsed_data.get("x")
        pos_y = parsed_data.get("y")
        if pos_x is not None and pos_y is not None:
            try:
                latest_position["x"] = float(pos_x)
                latest_position["y"] = float(pos_y)
                # Debug: Log position updates for PM001
                if athlete_id == "PM001":
                    print(f"[PM001] PM001 position updated in on_message_subscribe: ({latest_position['x']:.2f}, {latest_position['y']:.2f})")
            except (ValueError, TypeError) as e:
                print(f"[WARN]  Error converting position for {athlete_id}: x={pos_x}, y={pos_y}, error={e}")
        else:
            # Debug: Log when position is not updated
            if athlete_id == "PM001" and (pos_x is None or pos_y is None):
                print(f"[WARN]  PM001 position NOT updated in on_message_subscribe: x={pos_x}, y={pos_y} (from parsed_data)")

        # Convert publisher.py format to the expected format
        # Handle magno as single value (from subscriber.py) or as dict with x, y, z (from publisher.py)
        magno_value = parsed_data.get("magno")
        if magno_value is not None:
            if isinstance(magno_value, (int, float)):
                # Single value from subscriber.py - use for all axes
                magno_dict = {"x": magno_value, "y": magno_value, "z": magno_value}
            elif isinstance(magno_value, dict):
                # Dict format from publisher.py
                magno_dict = magno_value
            else:
                magno_dict = {"x": 0, "y": 0, "z": 0}
        else:
            # Fallback to mag_x, mag_y, mag_z if magno not present
            magno_dict = {
                "x": parsed_data.get("mag_x", 0),
                "y": parsed_data.get("mag_y", 0),
                "z": parsed_data.get("mag_z", 0)
            }
        
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
            "magno": magno_dict,
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

def process_raw_sensor_data(parsed_data: Dict[str, Any], device_id: str) -> None:
    """
    Process raw sensor data from subscriber.py (direct function call).
    
    This function is called directly from subscriber.py to process parsed sensor data
    without going through MQTT message handling.
    
    Args:
        parsed_data: Dictionary containing parsed sensor data with fields:
            - device_id, athlete_id, subhost_id, pm_id
            - acc_x, acc_y, acc_z
            - gyro_x, gyro_y, gyro_z
            - magno (single value)
            - pressure, temperature, btyVolt
            - mode
        device_id: Device ID string (e.g., "4" for PM004)
    """
    global sensor_data, unique_mqtt_topics
    
    # Lazy initialization: Initialize all components if not already initialized
    # This allows subscriber.py to run independently and initialize components when needed
    _initialize_components()
    _initialize_health_monitor()
    
    # Initialize publish client if not already initialized
    if not _publish_client_initialized:
        _initialize_publish_client()
    
    try:
        # Use device_id from parameter, fallback to parsed_data
        device_id_str = device_id or str(parsed_data.get("device_id", "")).strip()
        if not device_id_str:
            logger.warning("No device_id provided, skipping processing")
            return
        device_id_str = _normalize_device_id(device_id_str)
        
        # Track unique data sources
        source_key = f"subscriber_{device_id_str}"
        unique_mqtt_topics.add(source_key)
        
        # Show that we received sensor data (less verbose)
        athlete_id = parsed_data.get("athlete_id", device_id_str)
        print(f"📥 Player {athlete_id} (Device {device_id_str}) - New sensor data from subscriber")
        
        # Log device activity and detect mode
        mode = parsed_data.get("mode", "game")
        _update_log_filename_if_needed(mode)
        
        # Dynamically scale model cache size based on active game-mode devices
        try:
            if mode == "game":
                active_game_devices.add(device_id_str)
                desired_cache_size = len(active_game_devices)
                # Bound by number of available models
                if desired_cache_size > model_loader.cache_size:
                    new_cache_size = min(desired_cache_size, max_available_models)
                    if new_cache_size != model_loader.cache_size:
                        logger.info(
                            f"Adjusting model cache size from {model_loader.cache_size} to {new_cache_size} (active game devices: {desired_cache_size})"
                        )
                        model_loader.cache_size = new_cache_size
                        # Print cache info to console after resizing
                        cache_info = model_loader.get_cache_info()
                        print(
                            f"🧠 Cache: {cache_info['cache_size']}/{cache_info['max_cache_size']} | "
                            f"Hit: {cache_info['cache_hit_rate']:.1%} | "
                            f"Loaded: {cache_info['models_loaded']} | "
                            f"Evicted: {cache_info['models_evicted']} | "
                            f"Cached: {cache_info['cached_models']}"
                        )
        except Exception:
            # Do not interrupt message processing if scaling fails
            pass
        
        # Only log device activity on first activation, not every message
        if device_id_str not in device_contexts:
            logger.info(f"Device {device_id_str} (Player {athlete_id}) - First activation, Mode: {mode}")
            logger.info(f"Device {device_id_str} - MQTT prediction topic: predictions/{device_id_str}")

        # Get or init per-device context and sync into globals
        if device_id_str not in device_contexts:
            _init_device_context(device_id_str)
        ctx = device_contexts[device_id_str]
        # Persist last seen mode for this device to support mode-specific summaries
        try:
            ctx["last_mode"] = mode
        except Exception:
            pass
        _load_context_to_globals(ctx)
        
        # Optionally update athlete profile from payload if provided
        name_in = parsed_data.get("name")
        age_in = parsed_data.get("age")
        weight_in = parsed_data.get("weight")
        height_in = parsed_data.get("height")
        gender_in = parsed_data.get("gender")          ## 'M'/'F' or 1/0
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
        
        # Extract and update position from parsed_data (x, y from LPS or publisher)
        global latest_position
        pos_x = parsed_data.get("x")
        pos_y = parsed_data.get("y")
        if pos_x is not None and pos_y is not None:
            try:
                latest_position["x"] = float(pos_x)
                latest_position["y"] = float(pos_y)
                # Publish position immediately to lps/data/{device_id} so visualization
                # updates the plot without waiting for the full ML pipeline (reduces latency).
                _publish_position_immediate(device_id, latest_position["x"], latest_position["y"])
                # Debug: Log position updates for PM001
                if device_id == "PM001":
                    print(f"[PM001] PM001 position updated in process_raw_sensor_data: ({latest_position['x']:.2f}, {latest_position['y']:.2f})")
            except (ValueError, TypeError) as e:
                print(f"[WARN]  Error converting position for {device_id}: x={pos_x}, y={pos_y}, error={e}")
        else:
            # Debug: Log when position is not updated
            if device_id == "PM001" and (pos_x is None or pos_y is None):
                print(f"[WARN]  PM001 position NOT updated in process_raw_sensor_data: x={pos_x}, y={pos_y} (from parsed_data)")
        
        # Store subhost_id and pm_id in context if provided
        if "subhost_id" in parsed_data:
            ctx["subhost_id"] = parsed_data["subhost_id"]
        if "pm_id" in parsed_data:
            ctx["pm_id"] = parsed_data["pm_id"]
        
        # Convert subscriber.py format to the expected format
        # Handle magno as single value - convert to dict with same value for all axes
        magno_value = parsed_data.get("magno", 0.0)
        if isinstance(magno_value, (int, float)):
            magno_dict = {"x": magno_value, "y": magno_value, "z": magno_value}
        else:
            magno_dict = {"x": 0, "y": 0, "z": 0}
        
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
            "magno": magno_dict,
            # Include mode and heart rate data
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
        error_msg = f"Error processing raw sensor data: {e}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()

def check_session_end() -> bool:
    """
    Per-device session end checks. Generate summaries without stopping the loop.
    
    Checks all devices for inactivity timeout and generates session summaries
    for devices that have ended their sessions.
    
    Returns:
        False (always returns False to never stop the main loop)
    """
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

                # Handle prediction outputs directory path for new folder structure
                try:
                    base_output_dir = "../data/prediction_outputs"
                except:
                    base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prediction_outputs')
                player_folder = f"A{str(athlete_id)}_{str(name)}"
                full_player_path = os.path.join(base_output_dir, player_folder)
                os.makedirs(full_player_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                device_mode = ctx.get("last_mode", "game")
                mode_tag = "TR" if device_mode == "training" else "GM"
                summary_filename = f"A{athlete_id}_D{device_id}_{mode_tag}_session_summary_{timestamp}.json"
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
    def cleanup_handler(signum: int, frame: Any) -> None:
        """
        Signal handler for cleanup on process termination.
        
        Handles SIGINT and SIGTERM signals to gracefully shutdown the system,
        generate session summaries for all active devices, and clean up resources.
        
        Args:
            signum: Signal number (SIGINT=2, SIGTERM=15)
            frame: Current stack frame (for signal handlers)
        """
        print("\n[INFO] Prediction interrupted. Generating session summaries and cleaning up...")
        try:
            now_ts = time.time()
            # Generate and save session summary for all devices that had any activity
            for device_id_str, ctx in list(device_contexts.items()):
                try:
                    # Load context into globals to reuse summary pipeline
                    _load_context_to_globals(ctx)
                    # If session not marked ended, set end time to now
                    if not ctx.get("session_ended", False):
                        ctx["session_end_time"] = now_ts
                        globals()["session_end_time"] = now_ts
                        globals()["session_ended"] = True
                        _save_globals_to_context(ctx)
                    summary = generate_session_summary()
                    if summary:
                        # Handle prediction outputs directory path for new folder structure
                        try:
                            base_output_dir = "../data/prediction_outputs"
                        except:
                            base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prediction_outputs')
                        player_folder = f"A{str(athlete_id)}_{str(name)}"
                        full_player_path = os.path.join(base_output_dir, player_folder)
                        os.makedirs(full_player_path, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                        device_mode = ctx.get("last_mode", "game")
                        mode_tag = "TR" if device_mode == "training" else "GM"
                        summary_filename = f"A{athlete_id}_D{device_id}_{mode_tag}_session_summary_{timestamp}.json"
                        summary_filepath = os.path.join(full_player_path, summary_filename)
                        with open(summary_filepath, "w") as f:
                            json.dump(summary, f, indent=2)
                        print(f"Saved session summary for device {device_id_str} to {player_folder}/{summary_filename}")
                except Exception as e:
                    print(f"[WARN] Failed to save session summary for device {device_id_str}: {e}")
        finally:
            # Stop message queue processing
            try:
                message_queue.stop_processing()
                print("[INFO] Message queue processing stopped")
            except Exception as e:
                print(f"[WARN] Error stopping message queue: {e}")
            
            # Stop system health monitoring
            try:
                health_monitor.stop_monitoring()
                print("[INFO] System health monitoring stopped")
            except Exception as e:
                print(f"[WARN] Error stopping health monitor: {e}")
            
            remove_prediction_lockfile()
            sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Create SUBSCRIBE client for local broker (receiving sensor data)
    subscribe_client = mqtt.Client()
    subscribe_client.on_connect = on_connect_subscribe
    subscribe_client.on_disconnect = on_disconnect_subscribe
    subscribe_client.on_message = on_message_subscribe
    logger.info(f"Connecting to SUBSCRIBE MQTT broker: {MQTT_SUBSCRIBE_BROKER}:{MQTT_SUBSCRIBE_PORT}")
    try:
        result = subscribe_client.connect(MQTT_SUBSCRIBE_BROKER, MQTT_SUBSCRIBE_PORT, 60)
        if result == mqtt.MQTT_ERR_SUCCESS:
            logger.info("[OK] SUBSCRIBE client connection initiated (will connect asynchronously)")
        else:
            logger.warning(f"[WARN]  SUBSCRIBE client connect() returned code: {result}")
        subscribe_client.loop_start()
        
        # Wait a moment and check connection status
        time.sleep(2)
        if subscribe_client.is_connected():
            logger.info("[OK] SUBSCRIBE client is connected")
        else:
            logger.warning(f"[WARN]  SUBSCRIBE client not yet connected to {MQTT_SUBSCRIBE_BROKER}:{MQTT_SUBSCRIBE_PORT}")
            logger.warning("   Check if broker is running and accessible. Will retry connection automatically.")
    except Exception as e:
        logger.error(f"[ERR] Failed to connect to SUBSCRIBE MQTT broker ({MQTT_SUBSCRIBE_BROKER}:{MQTT_SUBSCRIBE_PORT}): {e}")
        logger.warning("[WARN]  Will retry connection automatically. Make sure the MQTT broker is running.")
        # Still start the loop - it will retry connection
        subscribe_client.loop_start()
    
    # Ensure PUBLISH client is initialized (may already be initialized if imported)
    _initialize_publish_client()
    
    # Get the global publish client (already initialized by _initialize_publish_client)
    publish_client = client
    
    logger.info(f"Starting test deployment in multi-device mode")
    logger.info(f"SUBSCRIBE broker: {MQTT_SUBSCRIBE_BROKER}:{MQTT_SUBSCRIBE_PORT} (topics: player/+/sensor/data, sensor/data)")
    logger.info(f"PUBLISH broker: {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} (topics: predictions/PM001, predictions/PM002, ...)")
    # Reduced logging for subscription messages

    # Initialize separate timers for different purposes
    last_status_report_time = 0  # For status updates every 30 seconds
    last_warning_report_time = 0  # For warnings every 5 seconds
    last_inactivity_check_time = 0  # For checking inactive devices
    last_publish_status_check = 0  # For checking publish client connection status
    
    # Track active/inactive device states
    device_activity_states = {}  # {device_id: {'active': True/False, 'last_report_time': timestamp}}
    DATA_TIMEOUT_SECONDS = 10  # Report if no data received for 10 seconds
    INACTIVITY_CHECK_INTERVAL = 5  # Check every 5 seconds
    all_devices_inactive_reported = False  # Track if we've reported all devices inactive

    while True:
        current_time = time.time()
        
        # Check publish client connection status every 30 seconds
        if current_time - last_publish_status_check >= 30:
            last_publish_status_check = current_time
            if publish_client.is_connected():
                # Get queue stats
                try:
                    queue_stats = message_queue.get_queue_stats()
                    pending = queue_stats.get('status_counts', {}).get('pending', 0)
                    sent = queue_stats.get('status_counts', {}).get('sent', 0)
                    failed = queue_stats.get('status_counts', {}).get('failed', 0)
                    if pending > 0 or sent > 0:
                        logger.info(f"[STAT] Publish Status: Connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} | Queue: {pending} pending, {sent} sent, {failed} failed")
                except Exception as e:
                    logger.debug(f"Error getting queue stats: {e}")
            else:
                try:
                    queue_stats = message_queue.get_queue_stats()
                    pending = queue_stats.get('status_counts', {}).get('pending', 0)
                    logger.warning(f"[WARN]  Publish client NOT connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} | {pending} messages queued")
                    logger.warning(f"   Check if broker at {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} is running and accessible")
                    logger.warning(f"   Test connection: mosquitto_sub -h {MQTT_PUBLISH_BROKER} -p {MQTT_PUBLISH_PORT} -t '#' -v")
                except Exception as e:
                    logger.warning(f"[WARN]  Publish client NOT connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} (error getting stats: {e})")
        
        # Check publish client connection status every 30 seconds
        if current_time - last_publish_status_check >= 30:
            last_publish_status_check = current_time
            if publish_client.is_connected():
                # Get queue stats
                queue_stats = message_queue.get_queue_stats()
                pending = queue_stats.get('status_counts', {}).get('pending', 0)
                sent = queue_stats.get('status_counts', {}).get('sent', 0)
                failed = queue_stats.get('status_counts', {}).get('failed', 0)
                if pending > 0 or sent > 0:
                    logger.info(f"[STAT] Publish Status: Connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} | Queue: {pending} pending, {sent} sent, {failed} failed")
            else:
                queue_stats = message_queue.get_queue_stats()
                pending = queue_stats.get('status_counts', {}).get('pending', 0)
                logger.warning(f"[WARN]  Publish client NOT connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} | {pending} messages queued")
                logger.warning(f"   Check if broker at {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} is running and accessible")
        # Both clients use loop_start() so we just need to keep the main loop running
        time.sleep(1.0)  # Sleep to prevent CPU spinning
        current_time = time.time()
        
        # Check for inactive devices periodically
        if current_time - last_inactivity_check_time >= INACTIVITY_CHECK_INTERVAL:
            last_inactivity_check_time = current_time
            
            for device_id_str, ctx in device_contexts.items():
                last_data_time = ctx.get("last_data_time", 0)
                athlete_id = ctx.get("athlete_id", "unknown")
                
                # Initialize device state if not present
                if device_id_str not in device_activity_states:
                    device_activity_states[device_id_str] = {
                        'active': True,
                        'last_report_time': current_time
                    }
                
                device_state = device_activity_states[device_id_str]
                time_since_last_data = current_time - last_data_time
                
                # Check if device has become inactive
                if device_state['active'] and time_since_last_data > DATA_TIMEOUT_SECONDS:
                    # Device was active but now inactive
                    print(f"\n[WARN]  WARNING: No data received from Player {athlete_id} (Device {device_id_str}) for {time_since_last_data:.1f} seconds")
                    print(f"   Last data received at: {datetime.fromtimestamp(last_data_time).strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.warning(f"Device {device_id_str} (Player {athlete_id}) inactive - no data for {time_since_last_data:.1f}s")
                    device_state['active'] = False
                    device_state['last_report_time'] = current_time
                
                # Report periodically if device remains inactive (every 30 seconds)
                elif not device_state['active'] and time_since_last_data > DATA_TIMEOUT_SECONDS:
                    if current_time - device_state['last_report_time'] >= 30:
                        print(f"[WARN]  Player {athlete_id} (Device {device_id_str}) still inactive - {time_since_last_data:.1f}s since last data")
                        device_state['last_report_time'] = current_time
                
                # Check if device has become active again
                elif not device_state['active'] and time_since_last_data <= DATA_TIMEOUT_SECONDS:
                    print(f"\n[OK] Player {athlete_id} (Device {device_id_str}) is now ACTIVE again")
                    logger.info(f"Device {device_id_str} (Player {athlete_id}) active again")
                    device_state['active'] = True
            
            # Check if all devices are inactive (publisher stopped)
            if device_activity_states:
                all_inactive = all(not state['active'] for state in device_activity_states.values())
                
                if all_inactive and not all_devices_inactive_reported:
                    print("\n" + "="*60)
                    print("📡 PUBLISHER STATUS: ALL DEVICES INACTIVE")
                    print("="*60)
                    print(f"[WARN]  No data received from any device for {DATA_TIMEOUT_SECONDS}+ seconds")
                    print(f"   Total devices tracked: {len(device_activity_states)}")
                    print(f"   This likely means the publisher has been stopped.")
                    print("\nDevice Last Activity Times:")
                    for dev_id, ctx in device_contexts.items():
                        athlete_id = ctx.get("athlete_id", "unknown")
                        last_data_time = ctx.get("last_data_time", 0)
                        time_since = current_time - last_data_time
                        last_time_str = datetime.fromtimestamp(last_data_time).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"   Player {athlete_id} (Device {dev_id}): {time_since:.1f}s ago (at {last_time_str})")
                    print("="*60)
                    logger.warning(f"All {len(device_activity_states)} devices inactive - publisher likely stopped")
                    all_devices_inactive_reported = True
                
                elif not all_inactive and all_devices_inactive_reported:
                    # At least one device is active again
                    print("\n" + "="*60)
                    print("[OK] PUBLISHER RESUMED: Data flow detected")
                    print("="*60)
                    logger.info("Publisher resumed - receiving data again")
                    all_devices_inactive_reported = False

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
                # Handle prediction outputs directory path for new folder structure
                try:
                    base_output_dir = "../data/prediction_outputs"
                except:
                    base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prediction_outputs')
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