# Jetson Nano ML Training System - Complete Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Complete Data Flow Analysis](#complete-data-flow-analysis)
3. [Training Data Generation Flow](#training-data-generation-flow)
4. [Live Prediction Flow](#live-prediction-flow)
5. [Key Integration Points](#key-integration-points)
6. [Conflict Prevention System](#conflict-prevention-system)
7. [File Structure](#file-structure)
8. [Configuration Management](#configuration-management)
9. [Installation & Setup](#installation--setup)
10. [Usage Examples](#usage-examples)
11. [Troubleshooting](#troubleshooting)

---

## System Overview

The Jetson Nano ML Training System is a comprehensive athlete monitoring solution that provides:

- **Multi-player sensor data simulation** (1-30 athletes)
- **Real-time prediction engine** for live performance monitoring
- **Automated ML model training** with conflict prevention
- **Health metrics calculation** (heart rate, stress, VO2 Max, TRIMP)
- **Session management** with automatic data saving

### Key Components

| Component | Purpose | File |
|-----------|---------|------|
| **Data Publisher** | Generates realistic sensor data | `publisher.py` |
| **Prediction Engine** | Real-time ML predictions | `test_30_players.py` |
| **Training System** | Automated ML model training | `sup_ml_rf_training.py` |
| **Configuration** | Unified system settings | `jetson_config.yaml` |
| **Deployment Helper** | Setup and management | `jetson_deploy.py` |

---

## Complete Data Flow Analysis

### Flow 1: Training Data Generation & ML Training

```
🎯 TRAINING SCENARIO
================

1. PUBLISHER (Training Data Generation)
   ├── Command: python publisher.py <num_players>
   ├── Generates: Realistic sensor data (10 Hz sampling rate)
   ├── MQTT Topics: player/{device_id}/sensor/data
   ├── Data Format: 
   │   ├── device_id, timestamp, athlete_id
   │   ├── age, weight, height, gender
   │   ├── acc_x/y/z (accelerometer)
   │   ├── gyro_x/y/z (gyroscope)
   │   ├── mag_x/y/z (magnetometer)
   │   └── heart_rate_bpm
   ├── Storage: In-memory collection during session
   └── On Stop (Ctrl+C): Saves to CSV files

2. DATA SAVING (Automatic on Publisher Stop)
   ├── Directory: athlete_training_data/player_{id}/
   ├── Filename Pattern: TR{seq}_A{athlete_id}_D{device_id}_{timestamp}.csv
   ├── CSV Format: timestamp,athlete_id,age,weight,height,gender,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate
   ├── Sequence Management: Auto-increments (TR1, TR2, TR3...)
   └── Status: Ready for ML training

3. ML TRAINING (Triggered by new data)
   ├── Command: python sup_ml_rf_training.py
   ├── Conflict Check: Ensures no live prediction running
   ├── Player Scan: All 30 players for new/updated TR files
   ├── Decision Logic:
   │   ├── ✅ Train if: No model exists OR new data OR force_retrain=true
   │   └── ❌ Skip if: Model newer than data OR insufficient sessions (<3)
   ├── Data Processing:
   │   ├── Load: Latest 3 TR sessions per player
   │   ├── Feature Engineering: Rolling stats, FFT, jerk calculations
   │   └── Preprocessing: Handle NaN values, normalize data
   ├── Model Training:
   │   ├── Algorithm: RandomForest Regressor
   │   ├── Parameters: 100 trees, max_depth=8, min_samples_split=5
   │   └── Target: Heart rate prediction
   ├── Model Saving:
   │   ├── PKL Format: athlete_models_pkl/Player{id}_rf.pkl
   │   └── Hummingbird Format: athlete_models_tensors_updated/hb_{id}_model.zip
   └── Result: Updated models ready for prediction
```

### Flow 2: Live Prediction During Actual Play

```
🎮 LIVE PREDICTION SCENARIO
=========================

1. PUBLISHER (Real Sensor Data)
   ├── Source: Real athlete sensors OR simulated data
   ├── Command: python publisher.py <num_players>
   ├── MQTT Topics: player/{device_id}/sensor/data
   ├── Data Rate: 10 Hz per player
   ├── Multi-player: Supports 1-30 athletes simultaneously
   └── Real-time streaming: Continuous data flow

2. PREDICTION ENGINE (test_30_players.py)
   ├── Startup:
   │   ├── Command: python test_30_players.py
   │   ├── Lockfile: Creates .prediction_running.lock
   │   └── MQTT Connection: localhost:1883
   ├── Model Loading:
   │   ├── Discovery: Scans athlete_models_tensors_updated/
   │   ├── Registry: Maps device_id to specific model
   │   ├── Fallback: Default model if specific not found
   │   └── GPU Support: CUDA acceleration if available
   ├── Real-time Processing Pipeline:
   │   ├── Data Ingestion: MQTT → sensor_data structure
   │   ├── Multi-device Context: Separate state per player
   │   ├── Sensor Fusion: Madgwick filter for orientation
   │   ├── Motion Analysis: Velocity, distance, acceleration
   │   ├── ML Prediction: Heart rate using player-specific model
   │   ├── Health Metrics:
   │   │   ├── Stress Level: Based on HR, HRV, activity
   │   │   ├── VO2 Max: Estimated from age, gender, HR, HRV
   │   │   ├── TRIMP: Training impulse calculation
   │   │   ├── Energy Expenditure: MET-based calculation
   │   │   ├── G-impact Detection: High-g event monitoring
   │   │   └── Fitness/Hydration: Dynamic decay modeling
   │   ├── Output Generation: Real-time metrics JSON
   │   └── MQTT Publish: {device_id}/predictions
   ├── Session Management:
   │   ├── Multi-device: Handles up to 30 players simultaneously
   │   ├── Per-device Context: Separate processing state
   │   ├── Session Logging: logs/A{id}_D{device}_{timestamp}.log
   │   ├── Auto-summary: Generates report on idle/stop
   │   └── Memory Monitoring: Real-time usage tracking
   └── Outputs:
       ├── Real-time JSON: A{id}_{name}/A{id}_D{device}_realtime_output.json
       ├── Session Logs: Detailed activity logs
       └── G-impact Logs: High-impact event records
```

---

## Key Integration Points

| Component | Training Flow | Prediction Flow |
|-----------|---------------|-----------------|
| **Publisher** | Generates training data for model development | Provides real sensor data for live monitoring |
| **MQTT Topics** | `player/{device}/sensor/data` | `player/{device}/sensor/data` |
| **Data Format** | Identical structure, stored for training | Identical structure, processed in real-time |
| **Conflict Prevention** | Checks for running prediction before starting | Creates lockfile to block training during prediction |
| **Models** | Creates and updates ML models | Consumes existing trained models |
| **Output** | TR*.csv training files | Real-time predictions and health metrics |
| **Configuration** | `jetson_config.yaml` controls all parameters | Same configuration file |
| **Logging** | Training progress and results | Session activities and health events |

---

## Conflict Prevention System

### Problem Solved
Prevents simultaneous training and prediction that could cause:
- Model file corruption during updates
- Prediction instability during training
- Resource contention and memory issues
- Inconsistent model states

### Implementation

```
TRAINING ATTEMPT FLOW
├── Startup Check: Is test_30_players.py running?
├── Detection Methods:
│   ├── 1. Process Detection: Scans running processes for script names
│   └── 2. Lockfile Check: Looks for .prediction_running.lock
├── If Conflict Detected:
│   ├── Warning: "Live prediction is currently running!"
│   ├── User Options: Wait for prediction to stop OR Cancel training
│   ├── Wait Logic: Polls every 5 seconds with 5-minute timeout
│   └── Timeout: Exits with error if prediction doesn't stop
└── If Clear: Creates .training_running.lock and proceeds

PREDICTION STARTUP FLOW
├── Lockfile Creation: .prediction_running.lock with current PID
├── Signal Handlers: Clean up lockfile on Ctrl+C or termination
├── Training Block: Prevents training from starting during prediction
└── Cleanup: Removes lockfile on normal or interrupted exit
```

### Configuration Options

```yaml
prediction_check:
  enabled: true                    # Enable/disable conflict checking
  prediction_script_names:         # Scripts to monitor for conflicts
    - 'test_30_players.py'
    - 'test_deployment1.py'
  lockfile_path: '.prediction_running.lock'
  check_interval_secs: 5           # How often to check (seconds)
  max_wait_time_secs: 300          # Maximum wait time (seconds)
  force_training: false            # Override conflict prevention (DANGEROUS!)
```

---

## File Structure

```
jetson_nano_test/                   # Project root directory
├── Core Scripts
│   ├── publisher.py                # Multi-player data generation
│   ├── test_30_players.py         # Live prediction engine (multi-device)
│   ├── test_deployment1.py        # Live prediction engine (single-device)
│   ├── sup_ml_rf_training.py      # ML model training system
│   └── jetson_deploy.py           # Deployment helper script
├── Configuration
│   ├── jetson_config.yaml         # Unified system configuration
│   └── requirements.txt           # Python dependencies
├── Training Data
│   └── athlete_training_data/     # Training data storage
│       ├── player_1/
│       │   ├── TR1_A1_D001_2025_09_15-12_31_50.csv
│       │   ├── TR2_A1_D001_2025_09_16-14_22_10.csv
│       │   └── TR3_A1_D001_2025_09_17-16_45_30.csv
│       ├── player_2/
│       │   ├── TR1_A2_D002_2025_09_15-12_31_50.csv
│       │   └── TR2_A2_D002_2025_09_16-14_22_10.csv
│       └── ... (up to player_30)
├── Models
│   ├── athlete_models_pkl/        # Scikit-learn models
│   │   ├── Player1_rf.pkl
│   │   ├── Player2_rf.pkl
│   │   └── ... (up to Player30_rf.pkl)
│   ├── athlete_models_tensors_updated/  # Hummingbird models (active)
│   │   ├── hb_1_model.zip
│   │   ├── hb_2_model.zip
│   │   └── ... (up to hb_30_model.zip)
│   └── athlete_models_tensors_previous/ # Backup models
├── Runtime Data
│   ├── logs/                      # Prediction session logs
│   │   ├── A1_D001_2025_09_17-14_30_15.log
│   │   ├── A2_D002_2025_09_17-14_30_15.log
│   │   └── ...
│   ├── A1_Device_001/            # Real-time output (Player 1)
│   │   ├── A1_D001_realtime_output.json
│   │   └── A1_D001_session_summary_*.json
│   ├── A2_Device_002/            # Real-time output (Player 2)
│   │   └── ...
│   └── ... (up to A30_Device_030)
├── System Files
│   ├── .prediction_running.lock   # Prediction active indicator
│   ├── .training_running.lock     # Training active indicator
│   └── jetson_training.log        # ML training log
└── Documentation
    ├── JETSON_NANO_ML_SYSTEM_DOCUMENTATION.docx
    ├── JETSON_NANO_ML_SYSTEM_DOCUMENTATION.md
    ├── PUBLISHER_USAGE.md
    └── README files
```

---

## Configuration Management

### Unified Configuration File: jetson_config.yaml

All system settings are centralized in a single configuration file:

#### Core Paths
```yaml
paths:
  data_root: 'athlete_training_data'           # Training data location
  models_root: 'athlete_models_pkl'            # Scikit-learn models
  hb_tensors_updated: 'athlete_models_tensors_updated'  # Active Hummingbird models
  hb_tensors_previous: 'athlete_models_tensors_previous' # Backup models
  data_file_pattern: '*.csv'                   # Training file pattern
```

#### Training Parameters
```yaml
training:
  n_estimators: 100              # RandomForest trees
  max_depth: 8                   # Maximum tree depth
  min_samples_split: 5           # Minimum samples for split
  min_samples_leaf: 10           # Minimum samples per leaf
  random_state: 42               # Reproducibility seed
  n_jobs: 2                      # Parallel processing threads
  sessions_to_use: 3             # Latest N sessions for training
  min_sessions_required: 3       # Minimum sessions needed
```

#### Feature Engineering
```yaml
feature_engineering:
  enabled: true                  # Enable advanced features
  rolling_window: 10             # Rolling statistics window
  sampling_frequency: 10         # Data sampling rate (Hz)
  features:
    resultant: true              # Resultant acceleration
    rolling_stats: true          # Rolling mean/std
    jerk: true                   # Jerk calculations
    fft_features: true           # Frequency domain features
```

#### Jetson Optimization
```yaml
jetson:
  device_detection: true         # Auto-detect GPU/CPU
  memory_limit_mb: 2048          # Memory usage limit
  batch_size: 1000              # Processing batch size
  use_mixed_precision: false     # Mixed precision training
  gpu_memory_fraction: 0.7       # GPU memory allocation
```

#### Conflict Prevention
```yaml
prediction_check:
  enabled: true                  # Enable conflict prevention
  prediction_script_names:       # Scripts to monitor
    - 'test_30_players.py'
    - 'test_deployment1.py'
  lockfile_path: '.prediction_running.lock'
  check_interval_secs: 5         # Check frequency
  max_wait_time_secs: 300        # Maximum wait time
  force_training: false          # Override conflicts
```

#### Logging Configuration
```yaml
logging:
  level: 'INFO'                  # Log level (DEBUG/INFO/WARN/ERROR)
  log_to_file: true             # Enable file logging
  log_file: 'jetson_training.log'  # Training log file
  console_output: true           # Console output
  detailed_progress: true        # Detailed progress info
  logs_dir: './logs'            # Prediction session logs directory
```

---

## Installation & Setup

### System Requirements
- **Hardware**: NVIDIA Jetson Nano (or compatible system)
- **OS**: Ubuntu 18.04+ or Jetson Linux
- **Python**: 3.7+
- **Memory**: Minimum 4GB RAM
- **Storage**: 16GB+ available space

### Required Python Packages
```
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
joblib>=1.0.0
torch>=1.8.0
hummingbird-ml>=0.4.0
pyyaml>=5.4.0
scipy>=1.6.0
paho-mqtt>=1.5.0
ahrs>=0.3.0
psutil>=5.8.0
python-dotenv>=0.19.0
```

### Installation Steps

1. **Clone/Setup Project Directory**
   ```bash
   # Navigate to your project directory
   cd /path/to/jetson_nano_test
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python jetson_deploy.py check
   ```

4. **Setup Directories**
   ```bash
   python jetson_deploy.py setup
   ```

5. **Validate Configuration**
   ```bash
   python jetson_deploy.py validate
   ```

6. **Start MQTT Broker** (if not running)
   ```bash
   sudo apt install mosquitto mosquitto-clients
   sudo systemctl start mosquitto
   sudo systemctl enable mosquitto
   ```

---

## Usage Examples

### Training Data Generation

#### Generate data for all 30 players:
```bash
python publisher.py 30
# Let it run for desired duration (e.g., 5 minutes)
# Press Ctrl+C to stop and save data
```

#### Generate data for 5 random players:
```bash
python publisher.py 5
# Shows which players are selected
# Data saved to athlete_training_data/player_X/
```

#### Generate data for 1 specific player:
```bash
python publisher.py 1
# Randomly selects one player
# Creates TR files with proper sequence numbering
```

### ML Model Training

#### Automatic training (recommended):
```bash
python sup_ml_rf_training.py
# Automatically scans all players
# Trains models where new data exists
# Shows progress and results
```

#### Check training status:
```bash
python jetson_deploy.py status
# Shows system configuration
# Lists available models
# Displays directory status
```

### Live Prediction

#### Multi-player prediction:
```bash
# Terminal 1: Start prediction engine
python test_30_players.py

# Terminal 2: Generate live data
python publisher.py 10
```

#### Single-player prediction:
```bash
# Terminal 1: Start single-device prediction
python test_deployment1.py
# (Requires athlete ID input)

# Terminal 2: Generate data
python publisher.py 1
```

### System Management

#### Full system check:
```bash
python jetson_deploy.py check      # Check dependencies
python jetson_deploy.py validate   # Validate configuration
python jetson_deploy.py status     # Show system status
```

#### Run training with helper:
```bash
python jetson_deploy.py run
# Performs checks and runs training
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. MQTT Connection Failed
**Problem**: Cannot connect to MQTT broker
**Solutions**:
```bash
# Check if mosquitto is running
sudo systemctl status mosquitto

# Start mosquitto if not running
sudo systemctl start mosquitto

# Check port availability
netstat -an | grep 1883
```

#### 2. Training Won't Start
**Problem**: "Live prediction is currently running!"
**Solutions**:
```bash
# Check for running prediction processes
ps aux | grep python | grep test_30_players

# Remove stale lockfile if no process running
rm .prediction_running.lock

# Force training (use with caution)
# Edit jetson_config.yaml: force_training: true
```

#### 3. No Models Found
**Problem**: "No models found under athlete_models_tensors_updated/"
**Solutions**:
```bash
# Check if training has been run
ls athlete_models_tensors_updated/

# Run training to generate models
python sup_ml_rf_training.py

# Check training data exists
ls athlete_training_data/player_*/TR*.csv
```

#### 4. Memory Issues
**Problem**: Out of memory during training/prediction
**Solutions**:
```bash
# Check memory usage
free -h

# Reduce batch size in jetson_config.yaml
# batch_size: 500  (instead of 1000)

# Reduce GPU memory fraction
# gpu_memory_fraction: 0.5  (instead of 0.7)

# Use CPU instead of GPU
# USE_CUDA=0 python test_30_players.py
```

#### 5. Permission Errors
**Problem**: Cannot write files or create directories
**Solutions**:
```bash
# Check permissions
ls -la

# Fix permissions
chmod 755 .
chmod 644 *.py

# Create directories manually if needed
mkdir -p athlete_training_data logs
```

#### 6. Model Loading Errors
**Problem**: "Failed to load model" errors
**Solutions**:
```bash
# Check model file integrity
ls -la athlete_models_tensors_updated/

# Retrain models if corrupted
rm athlete_models_tensors_updated/*
python sup_ml_rf_training.py

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

#### 7. Data Format Issues
**Problem**: Training data not loading correctly
**Solutions**:
```bash
# Check CSV file format
head -n 5 athlete_training_data/player_1/TR1*.csv

# Verify columns match expected format
# Expected: timestamp,athlete_id,age,weight,height,gender,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate

# Regenerate data if format is wrong
rm athlete_training_data/player_*/TR*.csv
python publisher.py 5  # Generate new data
```

### Log Analysis

#### Check training logs:
```bash
tail -f jetson_training.log
```

#### Check prediction logs:
```bash
tail -f logs/A*_D*_*.log
```

#### Check system logs:
```bash
journalctl -u mosquitto -f  # MQTT broker logs
dmesg | tail -20            # System messages
```

### Performance Optimization

#### For Jetson Nano:
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Check GPU usage
tegrastats
```

#### For training optimization:
- Reduce `n_estimators` for faster training
- Increase `n_jobs` for parallel processing
- Use `sessions_to_use: 2` for less data per model
- Disable feature engineering for basic models

#### For prediction optimization:
- Use `USE_CUDA=1` for GPU acceleration
- Reduce model count for memory savings
- Adjust `data_points_per_second` in publisher

---

## System Specifications

### Data Specifications
- **Sampling Rate**: 10 Hz per player
- **Data Fields**: 13 fields per sample
- **File Format**: CSV with headers
- **Naming Convention**: TR{seq}_A{athlete_id}_D{device_id}_{timestamp}.csv
- **Storage**: ~36,000 samples per 1-hour session per player

### Model Specifications
- **Algorithm**: RandomForest Regressor
- **Default Trees**: 100
- **Max Depth**: 8
- **Features**: Sensor data + engineered features + athlete profile
- **Target**: Heart rate (BPM)
- **Formats**: PKL (scikit-learn) + ZIP (Hummingbird)

### Performance Specifications
- **Training**: Handles 30 players with 3 sessions each
- **Prediction**: Real-time processing for up to 30 players
- **Latency**: <100ms prediction time per player
- **Memory**: <2GB RAM usage (configurable)
- **Storage**: ~1MB per training session per player

### Network Specifications
- **Protocol**: MQTT
- **Broker**: Mosquitto on localhost:1883
- **Topics**: `player/{device_id}/sensor/data` (input), `{device_id}/predictions` (output)
- **Message Format**: JSON
- **Rate**: 10 messages/second per player

---

*This documentation covers the complete Jetson Nano ML Training System for athlete monitoring and performance prediction. For additional support or feature requests, refer to the individual script documentation and configuration files.*
