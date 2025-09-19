import os
import time
import glob
import yaml
import logging
from typing import Dict, List, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from hummingbird.ml import convert
import torch
import onnx
from scipy.fft import rfft, rfftfreq
import psutil
import signal
import sys

# -----------------------------
# Configuration Loading
# -----------------------------
def load_config(config_path: str = "jetson_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"[INFO] Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"[WARN] Config file {config_path} not found, using default values")
        return get_default_config()
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}, using default values")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration if config file is not available."""
    return {
        'paths': {
            'data_root': 'athlete_training_data',
            'models_root': 'athlete_models_pkl',
            'hb_tensors_updated': 'athlete_models_tensors_updated',
            'hb_tensors_previous': 'athlete_models_tensors_previous',
            'data_file_pattern': '*.csv',
            'model_path': 'rf_model.pkl',
            'hb_path': 'hb_rf_model.zip',
            'onnx_path': 'rf_model.onnx'
        },
        'training': {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': 2,
            'sessions_to_use': 3,
            'min_sessions_required': 3
        },
        'monitoring': {
            'poll_interval_secs': 10,
            'debounce_secs': 3,
            'auto_update_enabled': True
        },
        'jetson': {
            'device_detection': True,
            'memory_limit_mb': 2048,
            'batch_size': 1000,
            'use_mixed_precision': False,
            'gpu_memory_fraction': 0.7
        },
        'backup': {
            'enabled': True,
            'keep_previous_versions': 3,
            'backup_on_update': True
        },
        'retraining': {
            'accuracy_threshold': 0.85,
            'min_samples_threshold': 500,
            'force_retrain_on_new_data': True
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': True,
            'log_file': 'jetson_training.log',
            'console_output': True,
            'detailed_progress': True
        },
        'feature_engineering': {
            'enabled': True,
            'rolling_window': 10,
            'sampling_frequency': 10,
            'features': {
                'resultant': True,
                'rolling_stats': True,
                'jerk': True,
                'fft_features': True,
                'acc_components': ['acc_x', 'acc_y', 'acc_z']
            }
        },
        'prediction_check': {
            'enabled': True,
            'prediction_script_names': ['test_30_players.py', 'test_deployment1.py'],
            'lockfile_path': '.prediction_running.lock',
            'check_interval_secs': 5,
            'max_wait_time_secs': 300,
            'force_training': False
        }
    }

# Load configuration
CONFIG = load_config()

# -----------------------------
# Prediction Running Check Functions
# -----------------------------
def is_prediction_script_running() -> bool:
    """
    Check if any live prediction scripts are currently running.
    Uses both process detection and lockfile mechanism.
    """
    prediction_config = CONFIG.get('prediction_check', {})
    
    if not prediction_config.get('enabled', True):
        return False
    
    # Method 1: Check running processes
    script_names = prediction_config.get('prediction_script_names', ['test_30_players.py', 'test_deployment1.py'])
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline:
                # Check if any of the command line arguments contain our script names
                cmdline_str = ' '.join(cmdline)
                for script_name in script_names:
                    if script_name in cmdline_str:
                        print(f"[INFO] Found running prediction script: {script_name} (PID: {proc.info['pid']})")
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # Method 2: Check lockfile
    lockfile_path = prediction_config.get('lockfile_path', '.prediction_running.lock')
    if os.path.exists(lockfile_path):
        try:
            # Check if the process in the lockfile is still running
            with open(lockfile_path, 'r') as f:
                pid_str = f.read().strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    if psutil.pid_exists(pid):
                        print(f"[INFO] Prediction script running according to lockfile (PID: {pid})")
                        return True
                    else:
                        # Stale lockfile, remove it
                        os.remove(lockfile_path)
                        print(f"[INFO] Removed stale lockfile: {lockfile_path}")
        except Exception as e:
            print(f"[WARN] Error checking lockfile {lockfile_path}: {e}")
    
    return False

def wait_for_prediction_to_stop() -> bool:
    """
    Wait for live prediction to stop running.
    Returns True if prediction stopped, False if timeout exceeded.
    """
    prediction_config = CONFIG.get('prediction_check', {})
    max_wait_time = prediction_config.get('max_wait_time_secs', 300)
    check_interval = prediction_config.get('check_interval_secs', 5)
    
    print(f"[INFO] Waiting for live prediction to stop (max wait: {max_wait_time}s)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if not is_prediction_script_running():
            print("[INFO] Live prediction has stopped. Training can proceed.")
            return True
        
        print(f"[INFO] Live prediction still running. Waiting {check_interval}s before next check...")
        time.sleep(check_interval)
    
    print(f"[WARN] Timeout exceeded ({max_wait_time}s). Live prediction is still running.")
    return False

def create_training_lockfile() -> None:
    """Create a lockfile to indicate training is in progress."""
    try:
        with open('.training_running.lock', 'w') as f:
            f.write(str(os.getpid()))
        print(f"[INFO] Created training lockfile (PID: {os.getpid()})")
    except Exception as e:
        print(f"[WARN] Failed to create training lockfile: {e}")

def remove_training_lockfile() -> None:
    """Remove the training lockfile."""
    try:
        if os.path.exists('.training_running.lock'):
            os.remove('.training_running.lock')
            print("[INFO] Removed training lockfile")
    except Exception as e:
        print(f"[WARN] Failed to remove training lockfile: {e}")

def check_prediction_before_training() -> bool:
    """
    Main function to check if training can proceed.
    Returns True if training can start, False otherwise.
    """
    prediction_config = CONFIG.get('prediction_check', {})
    
    if not prediction_config.get('enabled', True):
        print("[INFO] Prediction checking disabled. Training will proceed.")
        return True
    
    if prediction_config.get('force_training', False) or os.getenv('FORCE_TRAINING', '').lower() in ['true', '1', 'yes']:
        print("[INFO] Force training enabled. Skipping prediction check.")
        return True
    
    if not is_prediction_script_running():
        print("[INFO] No live prediction detected. Training can proceed.")
        return True
    
    print("[WARN] Live prediction is currently running!")
    print("[WARN] ML model training cannot start while live prediction is active.")
    print("[WARN] This prevents model conflicts and ensures prediction stability.")
    
    # Ask user if they want to wait
    try:
        response = input("Do you want to wait for prediction to stop? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return wait_for_prediction_to_stop()
        else:
            print("[INFO] Training cancelled by user.")
            return False
    except KeyboardInterrupt:
        print("\n[INFO] Training cancelled by user (Ctrl+C).")
        return False

# Extract commonly used config values
MODEL_PATH = CONFIG['paths']['model_path']
HB_PATH = CONFIG['paths']['hb_path']
ONNX_PATH = CONFIG['paths']['onnx_path']
DATA_ROOT = CONFIG['paths']['data_root']
MODELS_ROOT = CONFIG['paths']['models_root']
HB_TENSORS_ROOT = CONFIG['paths']['hb_tensors_updated']
HB_TENSORS_PREVIOUS = CONFIG['paths']['hb_tensors_previous']
DATA_FILE_GLOB = CONFIG['paths']['data_file_pattern']
POLL_INTERVAL_SECS = CONFIG['monitoring']['poll_interval_secs']
DEBOUNCE_SECS = CONFIG['monitoring']['debounce_secs']
SESSIONS_TO_USE = CONFIG['training']['sessions_to_use']
MIN_SESSIONS_REQUIRED = CONFIG['training']['min_sessions_required']

N_ESTIMATORS = CONFIG['training']['n_estimators']
MAX_DEPTH = CONFIG['training']['max_depth']
MIN_SAMPLES_SPLIT = CONFIG['training']['min_samples_split']
MIN_SAMPLES_LEAF = CONFIG['training']['min_samples_leaf']
RANDOM_STATE = CONFIG['training']['random_state']
N_JOBS = CONFIG['training']['n_jobs']

# -----------------------------
# Training Function
# -----------------------------
def train_rf(X, y, n_estimators=None, max_depth=None, 
             min_samples_split=None, min_samples_leaf=None):
    """Train Random Forest with configurable parameters."""
    # Use config values if not specified
    n_estimators = n_estimators or N_ESTIMATORS
    max_depth = max_depth or MAX_DEPTH
    min_samples_split = min_samples_split or MIN_SAMPLES_SPLIT
    min_samples_leaf = min_samples_leaf or MIN_SAMPLES_LEAF
    
    print(f"[INFO] Training RandomForest with n_estimators={n_estimators}, max_depth={max_depth}")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


# -----------------------------
# Feature Engineering Functions
# -----------------------------
def fft_peak(x, fs=10):
    """Calculate peak frequency from FFT."""
    if len(x) < 2:
        return 0.0
    x = x - np.mean(x)
    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1/fs)
    return float(xf[np.argmax(yf[1:])+1]) if len(yf) > 1 else 0.0

def fft_energy(x, fs=10):
    """Calculate energy from FFT."""
    if len(x) < 2:
        return 0.0
    x = x - np.mean(x)
    yf = np.abs(rfft(x))
    return float(np.sum(yf**2))

def engineer_features(athlete_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to accelerometer data.
    Creates additional features from raw acc_x, acc_y, acc_z data.
    """
    if not CONFIG['feature_engineering']['enabled']:
        return athlete_data
    
    print("[INFO] Applying feature engineering...")
    
    # Get configuration
    rolling_window = CONFIG['feature_engineering']['rolling_window']
    fs = CONFIG['feature_engineering']['sampling_frequency']
    features_config = CONFIG['feature_engineering']['features']
    acc_components = features_config['acc_components']
    
    # Check if required columns exist
    missing_cols = [col for col in acc_components if col not in athlete_data.columns]
    if missing_cols:
        print(f"[WARN] Missing accelerometer columns: {missing_cols}")
        return athlete_data
    
    # Create a copy to avoid modifying original data
    data = athlete_data.copy()
    
    try:
        # 1. Resultant acceleration
        if features_config['resultant']:
            data["resultant"] = np.sqrt(
                data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2
            )
        
        # 2. Rolling statistics for accelerometer components
        if features_config['rolling_stats']:
            for component in acc_components:
                data[f'{component}_mean'] = data[component].rolling(window=rolling_window).mean()
                data[f'{component}_std'] = data[component].rolling(window=rolling_window).std()
            
            # Rolling stats for resultant
            if features_config['resultant']:
                data["resultant_mean"] = data["resultant"].rolling(window=rolling_window).mean()
                data["resultant_std"] = data["resultant"].rolling(window=rolling_window).std()
        
        # 3. Jerk (derivative of resultant)
        if features_config['jerk'] and features_config['resultant']:
            data["jerk"] = data["resultant"].diff().abs() * fs
            data["jerk_mean"] = data["jerk"].rolling(window=rolling_window).mean()
        
        # 4. FFT features
        if features_config['fft_features'] and features_config['resultant']:
            data["fft_peak_freq"] = data["resultant"].rolling(window=rolling_window).apply(
                lambda x: fft_peak(x, fs=fs), raw=True
            )
            data["fft_energy"] = data["resultant"].rolling(window=rolling_window).apply(
                lambda x: fft_energy(x, fs=fs), raw=True
            )
        
        print(f"[INFO] Feature engineering completed. New shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        return athlete_data

# -----------------------------
# Per-player dataset loading
# -----------------------------
def list_player_ids(data_root: str = DATA_ROOT) -> List[str]:
    if not os.path.isdir(data_root):
        return []
    return [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]


def get_session_files(player_id: str, data_root: str = DATA_ROOT) -> List[str]:
    """
    Get all session CSV files for a player, sorted by session number.
    Returns list of file paths sorted by TR number (TR1, TR2, TR3, etc.)
    """
    player_dir = os.path.join(data_root, player_id)
    csv_paths = glob.glob(os.path.join(player_dir, DATA_FILE_GLOB))
    
    # Filter and sort by session number (TR1, TR2, TR3, etc.)
    session_files = []
    for path in csv_paths:
        filename = os.path.basename(path)
        if filename.startswith('TR') and filename.endswith('.csv'):
            session_files.append(path)
    
    # Sort by session number
    def extract_session_num(path):
        filename = os.path.basename(path)
        try:
            # Extract TR number (e.g., TR1 -> 1, TR2 -> 2)
            # Look for pattern TR followed by digits
            import re
            match = re.search(r'TR(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0
    
    session_files.sort(key=extract_session_num)
    print(f"[DEBUG] Found {len(session_files)} session files for player {player_id}: {[os.path.basename(f) for f in session_files]}")
    return session_files


def load_player_dataset_latest_sessions(player_id: str, data_root: str = DATA_ROOT, 
                                      num_sessions: int = SESSIONS_TO_USE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate the latest N session CSVs for a player.
    Uses rolling window approach: if 4 sessions exist, uses latest 3.
    Applies feature engineering if enabled in configuration.
    """
    session_files = get_session_files(player_id, data_root)
    
    if not session_files:
        raise FileNotFoundError(f"No session datasets found for player {player_id}")
    
    # Use latest N sessions
    latest_sessions = session_files[-num_sessions:] if len(session_files) >= num_sessions else session_files
    
    print(f"[INFO] Loading {len(latest_sessions)} sessions for player {player_id}: {[os.path.basename(f) for f in latest_sessions]}")
    
    # Load data using pandas for better handling
    dataframes: List[pd.DataFrame] = []
    for path in latest_sessions:
        try:
            # Try to load with pandas first (better for mixed data types)
            df = pd.read_csv(path)
            
            # Remove non-numeric columns except the target (last column)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df = df[numeric_cols]
                dataframes.append(df)
            else:
                print(f"[WARN] No numeric columns found in {path}")
                
        except Exception as e:
            print(f"[WARN] Failed to load {path} with pandas: {e}")
            # Fallback to numpy
            try:
                arr = np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                try:
                    arr = np.loadtxt(path, delimiter=",")
                    if arr.size > 0 and not np.issubdtype(arr.dtype, np.number):
                        arr = arr[1:]  # Skip header row
                except Exception:
                    print(f"[ERROR] Could not load {path}")
                    continue
            
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            
            # Convert numpy array to DataFrame
            df = pd.DataFrame(arr)
            dataframes.append(df)

    if not dataframes:
        raise ValueError(f"No valid data loaded for player {player_id}")
    
    # Concatenate all dataframes
    combined_data = pd.concat(dataframes, ignore_index=True)
    
    # Apply feature engineering
    combined_data = engineer_features(combined_data)
    
    # Handle NaN values created by rolling operations
    combined_data = combined_data.fillna(method='bfill').fillna(method='ffill')
    
    # Convert to numpy arrays
    all_data = combined_data.values
    
    # Last column is the target, rest are features
    X = all_data[:, :-1]
    y = all_data[:, -1].astype(int)
    
    print(f"[INFO] Final dataset shape: {X.shape}, target shape: {y.shape}")
    return X, y


def load_player_dataset(player_id: str, data_root: str = DATA_ROOT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy function - now uses latest sessions approach.
    Load and concatenate the latest N session CSVs in data_root/player_id.
    """
    # Convert player_id to directory name if needed
    if not player_id.startswith('player_'):
        player_dir_name = f"player_{player_id}"
    else:
        player_dir_name = player_id
    
    return load_player_dataset_latest_sessions(player_dir_name, data_root)


def get_player_model_path(player_id: str, models_root: str = MODELS_ROOT) -> str:
    os.makedirs(models_root, exist_ok=True)
    return os.path.join(models_root, f"Player{player_id}_rf.pkl")


def train_and_save_player_model(player_id: str) -> None:
    # Convert player_id to directory name if needed
    if not player_id.startswith('player_'):
        player_dir_name = f"player_{player_id}"
    else:
        player_dir_name = player_id
    
    # Check if we have enough sessions before training
    session_files = get_session_files(player_dir_name)
    if len(session_files) < MIN_SESSIONS_REQUIRED:
        print(f"[WARN] Cannot train player {player_id}: insufficient sessions ({len(session_files)} < {MIN_SESSIONS_REQUIRED})")
        return
    
    X, y = load_player_dataset_latest_sessions(player_dir_name)
    print(f"[INFO] Training model for player {player_id} on {len(X)} samples from latest sessions...")
    
    # Backup previous models before updating (if enabled in config)
    if CONFIG['backup']['enabled'] and CONFIG['backup']['backup_on_update']:
        backup_previous_pkl_model(player_id)  # Backup PKL model
        backup_previous_hb_model(player_id)   # Backup HB model
    
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(X, y)
    model_path = get_player_model_path(player_id)
    joblib.dump(model, model_path)
    print(f"[INFO] Saved new player {player_id} PKL model to {model_path} (replaced old model)")

    # Also export to per-player Hummingbird tensor format
    try:
        hb_out = get_player_hb_path(player_id)
        save_model_as_hb(model, X.shape[1], hb_out)
        print(f"[INFO] Saved new player {player_id} HB model to {hb_out} (replaced old model)")
    except Exception as e:
        print(f"[WARN] Failed HB export for player {player_id}: {e}")


def evaluate_existing_player_model(player_id: str) -> Tuple[float, int]:
    X, y = load_player_dataset(player_id)
    model_path = get_player_model_path(player_id)
    if not os.path.exists(model_path):
        return 0.0, len(X)
    model = joblib.load(model_path)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    return acc, len(X)


# -----------------------------
# Watcher for auto update
# -----------------------------
def scan_player_folder_mtime(player_id: str, data_root: str = DATA_ROOT) -> float:
    player_dir = os.path.join(data_root, player_id)
    latest_mtime = 0.0
    for path in glob.glob(os.path.join(player_dir, DATA_FILE_GLOB)):
        try:
            mtime = os.path.getmtime(path)
            latest_mtime = max(latest_mtime, mtime)
        except FileNotFoundError:
            continue
    return latest_mtime


def run_auto_update_loop():
    print(f"[INFO] Auto-update watching '{DATA_ROOT}' every {POLL_INTERVAL_SECS}s ...")
    last_seen_mtime: Dict[str, float] = {}
    while True:
        now = time.time()
        player_ids = list_player_ids()
        for player_id in player_ids:
            latest_mtime = scan_player_folder_mtime(player_id)
            prev_mtime = last_seen_mtime.get(player_id, 0.0)

            # Debounce: only act if file is older than debounce window
            if latest_mtime > prev_mtime and (now - latest_mtime) > DEBOUNCE_SECS:
                print(f"[INFO] Detected new/updated data for player {player_id}")
                try:
                    # Optional: evaluate existing model and decide retrain
                    acc, samples = evaluate_existing_player_model(player_id)
                    print(f"[INFO] Existing model acc={acc:.3f} samples={samples}")
                    if my_condition(acc, samples) or not os.path.exists(get_player_model_path(player_id)):
                        print(f"[INFO] Retraining model for player {player_id} ...")
                        train_and_save_player_model(player_id)
                    else:
                        print(f"[INFO] Skipping retrain for player {player_id} (condition not met)")
                except Exception as e:
                    print(f"[ERROR] Failed to update model for player {player_id}: {e}")
                finally:
                    last_seen_mtime[player_id] = latest_mtime

        # Cleanup players removed from disk
        last_seen_mtime = {pid: last_seen_mtime.get(pid, 0.0) for pid in player_ids}
        time.sleep(POLL_INTERVAL_SECS)

# -----------------------------
# Condition Function
# -----------------------------
def my_condition(acc, samples):
    """
    Retraining condition based on configuration:
    - retrain if accuracy < threshold
    - retrain if dataset > threshold
    """
    accuracy_threshold = CONFIG['retraining']['accuracy_threshold']
    samples_threshold = CONFIG['retraining']['min_samples_threshold']
    force_retrain = CONFIG['retraining']['force_retrain_on_new_data']
    
    return acc < accuracy_threshold or samples > samples_threshold or force_retrain

# -----------------------------
# Conditional Training Pipeline
# -----------------------------
def conditional_training(X, y, condition_fn=my_condition):
    retrain = False

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"[INFO] Existing model accuracy: {acc:.3f}")
        if condition_fn(acc, len(X)):
            print("[INFO] Condition met → retraining...")
            retrain = True
    else:
        print("[INFO] No existing model → training new...")
        retrain = True

    if retrain:
        model = train_rf(X, y)
    else:
        model = joblib.load(MODEL_PATH)

    return model

# -----------------------------
# Convert to Hummingbird + ONNX
# -----------------------------
def convert_to_hb_onnx(model, num_features):
    # Convert to Hummingbird PyTorch
    hb_model = convert(model, backend="pytorch", device="cuda")
    hb_model.save(HB_PATH)
    print(f"[INFO] Saved Hummingbird model at {HB_PATH}")

    # Export to ONNX
    torch_model = hb_model.model
    dummy_input = torch.randn(1, num_features)
    torch.onnx.export(
        torch_model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}}
    )
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"[INFO] Exported ONNX model at {ONNX_PATH}")

# -----------------------------
# Per-player HB conversion utils
# -----------------------------
def get_player_hb_path(player_id: str, root: str = HB_TENSORS_ROOT) -> str:
    os.makedirs(root, exist_ok=True)
    # Save format consistent with consumer expectations: hb_{index}_model.zip
    return os.path.join(root, f"hb_{player_id}_model.zip")


def backup_previous_hb_model(player_id: str) -> None:
    """
    Move current HB model to previous folder before updating.
    """
    current_hb_path = get_player_hb_path(player_id, HB_TENSORS_ROOT)
    if os.path.exists(current_hb_path):
        os.makedirs(HB_TENSORS_PREVIOUS, exist_ok=True)
        backup_hb_path = get_player_hb_path(player_id, HB_TENSORS_PREVIOUS)
        
        # Remove existing backup if it exists
        if os.path.exists(backup_hb_path):
            os.remove(backup_hb_path)
        
        # Move current to backup
        os.rename(current_hb_path, backup_hb_path)
        print(f"[INFO] Backed up previous HB model for player {player_id} to {backup_hb_path}")

def backup_previous_pkl_model(player_id: str) -> None:
    """
    Backup current PKL model before updating.
    """
    current_pkl_path = get_player_model_path(player_id)
    if os.path.exists(current_pkl_path):
        # Create backup directory
        backup_dir = os.path.join(MODELS_ROOT, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_pkl_path = os.path.join(backup_dir, f"{player_id}_backup_{timestamp}.pkl")
        
        # Copy current to backup
        import shutil
        shutil.copy2(current_pkl_path, backup_pkl_path)
        print(f"[INFO] Backed up previous PKL model for player {player_id} to {backup_pkl_path}")

def setup_logging():
    """Setup logging based on configuration."""
    log_config = CONFIG['logging']
    
    # Configure logging level
    log_level = getattr(logging, log_config['level'].upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if log_config['console_output']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_config['log_to_file']:
        file_handler = logging.FileHandler(log_config['log_file'])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    print(f"[INFO] Logging configured - Level: {log_config['level']}, File: {log_config['log_to_file']}")

def detect_device() -> str:
    """Detect available device with Jetson optimizations."""
    if not CONFIG['jetson']['device_detection']:
        return "cpu"
    
    try:
        if torch.cuda.is_available():
            # Jetson-specific GPU memory management
            gpu_memory_fraction = CONFIG['jetson']['gpu_memory_fraction']
            if gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            return "cuda"
        else:
            return "cpu"
    except Exception as e:
        print(f"[WARN] GPU detection failed: {e}, using CPU")
        return "cpu"

def save_model_as_hb(model, num_features: int, out_path: str) -> None:
    device = detect_device()
    hb_model = convert(model, backend="pytorch", device=device)
    hb_model.save(out_path)
    # Quick sanity check predict on dummy tensor to ensure the artifact is valid
    try:
        dummy = np.zeros((1, num_features), dtype=np.float32)
        _ = hb_model.predict(dummy)
    except Exception:
        pass

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Setup logging first
    setup_logging()
    
    # Print configuration summary
    print(f"[INFO] Jetson Nano ML Training")
    print(f"[INFO] Training params: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, n_jobs={N_JOBS}")
    print(f"[INFO] Sessions to use: {SESSIONS_TO_USE}, Poll interval: {POLL_INTERVAL_SECS}s")
    print(f"[INFO] Device detection: {CONFIG['jetson']['device_detection']}")
    
    # Check if live prediction is running before starting training
    print("\n" + "="*60)
    print("🔍 CHECKING FOR LIVE PREDICTION")
    print("="*60)
    
    if not check_prediction_before_training():
        print("[ERROR] Training cannot proceed due to live prediction conflict.")
        print("[INFO] Please stop the live prediction script and try again.")
        sys.exit(1)
    
    # Create training lockfile to indicate training is in progress
    create_training_lockfile()
    
    # Setup cleanup handler
    def cleanup_handler(signum, frame):
        print("\n[INFO] Training interrupted. Cleaning up...")
        remove_training_lockfile()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    print("\n" + "="*60)
    print("🚀 STARTING ML MODEL TRAINING")
    print("="*60)
    
    # Startup scan: check each of 30 players for new/updated datasets and train as needed
    def get_player_session_info(idx: int) -> Tuple[str, List[str]]:
        """
        Return (player_id_str, session_files) for a player.
        The data layout is athlete_training_data/player_{idx}/TR*_*.csv
        We'll use the numeric index as the player_id for saving (1..30).
        """
        player_dir_name = f"player_{idx}"
        player_dir = os.path.join(DATA_ROOT, player_dir_name)
        session_files = get_session_files(player_dir_name)  # Use directory name, not just index
        return str(idx), session_files

    def get_latest_session_mtime(session_files: List[str]) -> float:
        """Get the modification time of the latest session file."""
        if not session_files:
            return 0.0
        latest_file = max(session_files, key=os.path.getmtime)
        try:
            return os.path.getmtime(latest_file)
        except Exception:
            return 0.0

    def model_artifacts_mtime(player_id_str: str) -> float:
        pkl_path = get_player_model_path(player_id_str)
        hb_path = get_player_hb_path(player_id_str)
        mtimes = []
        for p in (pkl_path, hb_path):
            if os.path.exists(p):
                try:
                    mtimes.append(os.path.getmtime(p))
                except Exception:
                    pass
        return max(mtimes) if mtimes else 0.0

    retrained_count = 0
    for idx in range(1, 31):
        player_id_str, session_files = get_player_session_info(idx)
        
        if not session_files:
            print(f"[WARN] No session files found for player {player_id_str}")
            continue

        # Check if we have enough sessions for training
        if len(session_files) < MIN_SESSIONS_REQUIRED:
            print(f"[WARN] Insufficient sessions for player {player_id_str}: {len(session_files)} (need at least {MIN_SESSIONS_REQUIRED})")
            continue

        try:
            # Get modification time of latest session
            data_mtime = get_latest_session_mtime(session_files)
            artifacts_mtime = model_artifacts_mtime(player_id_str)
            
            # Check if we have more sessions than before (new data added)
            existing_model_exists = artifacts_mtime > 0.0
            
            # Force retrain if:
            # 1. No existing model exists
            # 2. New data is newer than existing model
            # 3. Force retrain is enabled in config
            should_train = (
                not existing_model_exists or 
                data_mtime > artifacts_mtime or 
                CONFIG['retraining']['force_retrain_on_new_data']
            )

            if not should_train:
                print(f"[INFO] Up-to-date: player {player_id_str} (dataset older than artifacts)")
                continue

            # Train using latest sessions approach
            print(f"[INFO] Training player {player_id_str} with latest {min(SESSIONS_TO_USE, len(session_files))} sessions...")
            
            # Backup previous models before training (if enabled)
            if CONFIG['backup']['enabled'] and CONFIG['backup']['backup_on_update']:
                backup_previous_pkl_model(player_id_str)  # Backup PKL model
                backup_previous_hb_model(player_id_str)   # Backup HB model
            
            # Load data from latest sessions
            X, y = load_player_dataset_latest_sessions(f"player_{idx}")
            
            # Train model with configurable parameters
            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                min_samples_split=MIN_SAMPLES_SPLIT,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )
            model.fit(X, y)

            # Save new PKL model (replaces old one)
            pkl_out = get_player_model_path(player_id_str)
            joblib.dump(model, pkl_out)
            print(f"[INFO] Saved new PKL model: {pkl_out} (replaced old model)")

            # Save HB
            hb_out = get_player_hb_path(player_id_str)
            save_model_as_hb(model, X.shape[1], hb_out)
            print(f"[INFO] Saved HB model: {hb_out}")

            retrained_count += 1
        except Exception as e:
            print(f"[ERROR] Training failed for player {player_id_str}: {e}")

    print(f"[INFO] Startup scan complete. Retrained {retrained_count} player models.")
    
    # Cleanup training lockfile
    remove_training_lockfile()
    
    print("\n" + "="*60)
    print("✅ ML MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
