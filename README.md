# Jetson Nano ML Training System

A comprehensive machine learning system for athlete performance analysis, designed specifically for deployment on NVIDIA Jetson Nano 4GB with dynamic model loading and memory optimization.

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation & Setup](#installation--setup)
5. [Quick Start Guide](#quick-start-guide)
6. [Dynamic Model Loading System](#dynamic-model-loading-system)
7. [Data Publisher Usage](#data-publisher-usage)
8. [ML Training System](#ml-training-system)
9. [Live Prediction System](#live-prediction-system)
10. [Jetson Nano Migration Guide](#jetson-nano-migration-guide)
11. [Configuration Management](#configuration-management)
12. [Troubleshooting](#troubleshooting)
13. [Performance Optimization](#performance-optimization)
14. [Project Structure](#project-structure)
15. [Documentation](#documentation)

---

## 🚀 System Overview

The Jetson Nano ML Training System is a comprehensive athlete monitoring solution that provides:

- **Multi-player sensor data simulation** (1-30 athletes)
- **Real-time prediction engine** for live performance monitoring
- **Automated ML model training** with conflict prevention
- **Dynamic model loading** for memory optimization
- **Health metrics calculation** (heart rate, stress, VO2 Max, TRIMP)
- **Session management** with automatic data saving

### Key Components

| Component | Purpose | File |
|-----------|---------|------|
| **Data Publisher** | Generates realistic sensor data | `publisher.py` |
| **Prediction Engine** | Real-time ML predictions | `main.py` |
| **Training System** | Automated ML model training | `sup_ml_rf_training.py` |
| **Dynamic Model Loader** | Memory-efficient model management | `dynamic_model_loader.py` |
| **Configuration** | Unified system settings | `jetson_orin_32gb_config.yaml` |
| **Deployment Helper** | Setup and management | `jetson_deploy.py` |

---

## ✨ Features

### Core Features
- **Real-time ML Training**: Train Random Forest models for up to 30 athletes simultaneously
- **Dynamic Model Loading**: Load models on-demand based on player/device IDs (83% memory reduction)
- **GPU Acceleration**: CUDA support with automatic fallback to CPU
- **Memory Optimization**: Designed for Jetson Nano 4GB constraints
- **MQTT Communication**: Real-time data streaming and model updates
- **Conflict Prevention**: Prevents training/prediction conflicts
- **Automatic Retraining**: Models update when new data arrives

### Advanced Features
- **Multi-mode Data Generation**: Training sessions and game simulations
- **Health Metrics**: Stress, VO2 Max, TRIMP, energy expenditure
- **G-impact Detection**: High-g event monitoring for injury prevention
- **Session Management**: Automatic data saving and session summaries
- **Performance Monitoring**: Real-time resource usage tracking

---

## 📋 System Requirements

### Hardware
- **NVIDIA Jetson Nano 4GB** (recommended)
- **microSD Card**: 32GB+ Class 10
- **Power Supply**: 5V/4A official adapter
- **Cooling**: Active cooling fan recommended

### Software
- **JetPack 4.6.6** (CUDA 10.2.300)
- **Python 3.10**
- **Ubuntu 18.04**

### Memory Requirements
- **Training**: ~1.5-2GB RAM peak usage
- **Prediction**: ~800MB-1.2GB RAM with dynamic loading
- **Storage**: ~1GB for models + data

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd jetson_nano_test
```

### 2. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Setup System
```bash
python3 jetson_deploy.py check
python3 jetson_deploy.py setup
```

### 4. Configure MQTT (Optional)
```bash
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

### 5. Verify Installation
```bash
python3 jetson_deploy.py validate
python3 jetson_deploy.py status
```

---

## 🛠️ Deployment Helper

The `jetson_deploy.py` script provides comprehensive deployment assistance and works with both configuration files:

### Available Commands

```bash
# Check system requirements and dependencies
python jetson_deploy.py check

# Setup directories and validate configuration
python jetson_deploy.py setup

# Run the training system
python jetson_deploy.py run

# Show system status (works with both config files)
python jetson_deploy.py status

# Validate configuration file structure
python jetson_deploy.py validate
```

### Key Features

- **Configuration**: Uses `jetson_orin_32gb_config.yaml` for optimized Jetson Orin 32GB settings (CUDA enabled)
- **Package validation**: Checks for required dependencies (numpy, pandas, scikit-learn, etc.)
- **Directory setup**: Creates necessary folders for data and models
- **Configuration validation**: Ensures config file structure is correct
- **System status**: Shows current configuration and directory status
- **Dynamic model loading info**: Displays cache settings and memory optimization

### Example Output

```bash
$ python jetson_deploy.py status

=== Jetson Orin ML Training System Status (jetson_orin_32gb_config.yaml) ===
Environment: Jetson Nano 4GB Optimized
Version: 4GB Optimized Configuration
Training Parameters:
  - n_estimators: 60
  - max_depth: 6
  - n_jobs: 2
  - sessions_to_use: 2
Dynamic Model Loading:
  - cache_size: 3
  - device: cpu
  - models_directory: athlete_models_tensors_updated
  - enable_memory_monitoring: True
Directories:
  - athlete_training_data: ✓
  - athlete_models_tensors_updated: ✓
```

---

## 🎯 Quick Start Guide

### Generate Training Data
```bash
# Generate data for 5 players
python3 publisher.py 5 --mode training
```

### Train Models
```bash
# Train ML models for all players
python3 sup_ml_rf_training.py
```

### Test Predictions
```bash
# Test real-time predictions
python3 test_30_players.py
```

### Complete Workflow
```bash
# Terminal 1: Start prediction engine
python3 test_30_players.py

# Terminal 2: Generate live data
python3 publisher.py 5 --mode game --duration 30
```

---

## 🧠 Dynamic Model Loading System

### Overview
The Dynamic Model Loading System optimizes memory usage on resource-constrained devices by loading ML models on-demand based on player/device IDs instead of loading all models at startup.

### Memory Savings
| Aspect | Before (Static Loading) | After (Dynamic Loading) | Savings |
|--------|------------------------|------------------------|---------|
| **Memory Usage** | ~1.5GB (30 models) | ~250MB (5 models cache) | **83% reduction** |
| **Startup Time** | Slow (loads all models) | Fast (loads default only) | **Significant improvement** |
| **Memory Efficiency** | Poor (unused models) | Excellent (active models only) | **Dramatic improvement** |

### How It Works
1. **On-Demand Loading**: Models loaded only when specific player/device ID is encountered
2. **LRU Cache**: Configurable cache size with automatic eviction of least recently used models
3. **Memory Monitoring**: Real-time memory usage tracking and reporting
4. **CUDA/CPU Support**: Automatic device detection and fallback

### Configuration
```yaml
# In jetson_orin_32gb_config.yaml
model_loading:
  cache_size: 3                    # Optimized for 4GB Jetson Nano
  device: 'cpu'                    # Force CPU for memory efficiency
  models_directory: 'athlete_models_tensors_updated'
  enable_memory_monitoring: true
```

### Usage
```python
from dynamic_model_loader import DynamicModelLoader

# Create dynamic model loader
loader = DynamicModelLoader(
    models_dir="athlete_models_tensors_updated",
    cache_size=3,  # Keep 3 models in memory
    device="cpu",
    enable_memory_monitoring=True
)

# Get model for specific player
model = loader.get_model(player_id=1)
```

### Performance Characteristics
- **Cache Hit Rate**: >80% for excellent performance
- **Loading Time**: ~2-5 seconds per model (first load)
- **Cache Hit**: ~0.001 seconds (in-memory)
- **Memory Usage**: ~250MB with cache size 3

---

## 📊 Data Publisher Usage

### Multi-Mode Data Publisher
The enhanced `publisher.py` script supports both training data generation and game simulation with flexible player selection.

### Command Line Syntax
```bash
python publisher.py <num_players> [--mode training|game] [--duration minutes] [--players list]
```

### Usage Examples

#### Random Player Selection
```bash
# Training Mode
python publisher.py 30 --mode training                    # All 30 players
python publisher.py 5 --mode training                     # 5 random players  
python publisher.py 1 --mode training                     # 1 random player

# Game Mode
python publisher.py 11 --mode game --duration 90          # 11 players, 90-min game
python publisher.py 5 --mode game --duration 45           # 5 players, 45-min game
```

#### Specific Player Selection
```bash
# Training Mode with Specific Players
python publisher.py --players [1,3,7] --mode training     # Players 1, 3, 7 for training
python publisher.py --players [2,9,28,12,5] --mode training  # Specific players

# Game Mode with Specific Players
python publisher.py --players [2,9,28,12,5] --mode game   # Specific players for game
python publisher.py --players [11,22] --mode game --duration 45  # 45-min game
```

### Data Modes

#### 🏃 Training Mode
- **Patterns**: Structured phases (warm-up → active → cool-down)
- **Duration**: Flexible 10-minute cycles
- **Intensity**: Predictable progression (70-145 bpm)
- **Storage**: `athlete_training_data/player_X/`
- **File prefix**: TR (TR1, TR2, TR3...)

#### ⚽ Game Mode
- **Patterns**: Dynamic game events (sprints, tackles, shots, passes)
- **Duration**: Game-specific (45-minute halves, customizable)
- **Intensity**: Variable based on position and events (90-180 bpm)
- **Storage**: `athlete_game_data/player_X/`
- **File prefix**: GM (GM1, GM2, GM3...)

### Data Formats

#### Training Mode CSV Format
```
timestamp,athlete_id,age,weight,height,gender,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate,session_phase,intensity_level,mode
```

#### Game Mode CSV Format
```
timestamp,athlete_id,age,weight,height,gender,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate,session_phase,intensity_level,mode,position,playing_style,fitness_level,fatigue_rate
```

---

## 🤖 ML Training System

### Overview
The ML training system automatically trains Random Forest models for each athlete based on their sensor data.

### Training Process
1. **Data Collection**: Publisher generates training data
2. **Conflict Check**: Ensures no live prediction running
3. **Player Scan**: Scans all 30 players for new/updated data
4. **Training Decision**: Trains if new data exists or model missing
5. **Feature Engineering**: Rolling stats, FFT, jerk calculations
6. **Model Training**: RandomForest Regressor with optimized parameters
7. **Model Saving**: Saves in both PKL and Hummingbird formats

### Training Parameters (Jetson Optimized)
```yaml
training:
  n_estimators: 60             # Reduced for faster training
  max_depth: 6                 # Reduced complexity for memory efficiency
  min_samples_split: 8         # Increased for simpler trees
  min_samples_leaf: 15         # Increased for better generalization
  n_jobs: 2                    # Use 2 cores, leave 2 for system
  sessions_to_use: 2           # Use 2 sessions instead of 3 (less memory)
```

### Usage
```bash
# Automatic training (recommended)
python sup_ml_rf_training.py

# Check training status
python jetson_deploy.py status
```

### Performance
- **Single Player Model**: ~30-60 seconds
- **30 Player Models**: ~15-30 minutes
- **Memory Usage**: ~1.5-2GB peak
- **GPU Acceleration**: 2-3x faster than CPU-only

---

## 🎮 Live Prediction System

### Overview
The live prediction system provides real-time performance monitoring for up to 30 athletes simultaneously.

### Features
- **Multi-device Support**: Handles up to 30 players simultaneously
- **Real-time Processing**: <50ms prediction time per player
- **Health Metrics**: Stress, VO2 Max, TRIMP, energy expenditure
- **G-impact Detection**: High-g event monitoring
- **Session Management**: Automatic data saving and summaries

### Usage
```bash
# Multi-player prediction
python test_30_players.py

# Single-player prediction
python test_deployment1.py
```

### Output Files
- **Real-time JSON**: `prediction_outputs/A{id}_{name}/A{id}_D{device}_realtime_output.json`
- **Session Logs**: `logs/A{id}_D{device}_{timestamp}.log`
- **G-impact Logs**: High-impact event records

### Performance
- **Real-time Prediction**: <50ms per player
- **Multi-player Support**: Up to 30 players simultaneously
- **Memory Usage**: ~800MB-1.2GB during prediction
- **Throughput**: 10 Hz per player sustained

---

## 🚀 Jetson Nano Migration Guide

### ✅ Compatibility Analysis

| Component | Requirement | Jetson Nano 4GB | Status |
|-----------|-------------|------------------|---------|
| **RAM** | ~2GB for training, ~1GB for prediction | 4GB total | ✅ **Compatible** |
| **GPU** | CUDA support (optional) | 128-core Maxwell GPU | ✅ **Supported** |
| **Storage** | ~1GB for models + data | microSD 32GB+ | ✅ **Sufficient** |
| **CPU** | Multi-core for parallel processing | Quad-core ARM A57 | ✅ **Adequate** |
| **Python** | 3.7+ | Ubuntu 18.04 with Python 3.6+ | ✅ **Compatible** |

### Migration Steps

#### 1. Prepare Jetson Nano
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-dev build-essential
sudo apt install -y mosquitto mosquitto-clients

# Enable maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### 2. Transfer Project Files
```bash
# Copy project directory
cp -r /path/to/jetson_nano_test /home/jetson/

# Set permissions
cd /home/jetson/jetson_nano_test
chmod +x *.py
```

#### 3. Install Dependencies
```bash
# Install Python packages
pip3 install -r requirements.txt

# Verify installation
python3 jetson_deploy.py check
python3 jetson_deploy.py validate
python3 jetson_deploy.py setup
```

#### 4. Optimize for Jetson Nano 4GB
Use the provided `jetson_orin_32gb_config.yaml` with optimized settings:
- Reduced memory limits
- Smaller batch sizes
- Conservative GPU usage
- Dynamic model loading enabled

### Expected Performance on Jetson Nano 4GB
- **Training**: 30 models in ~20-40 minutes
- **Prediction**: Real-time for 10-30 players simultaneously
- **Memory**: ~1.5GB peak usage during training
- **Temperature**: Monitor to prevent throttling

---

## ⚙️ Configuration Management

### Unified Configuration File: `jetson_orin_32gb_config.yaml`

All system settings are centralized in a single configuration file optimized for Jetson Nano 4GB:

#### Core Paths
```yaml
paths:
  data_root: 'athlete_training_data'
  models_root: 'athlete_models_pkl'
  hb_tensors_updated: 'athlete_models_tensors_updated'
  hb_tensors_previous: 'athlete_models_tensors_previous'
```

#### Training Parameters (4GB Optimized)
```yaml
training:
  n_estimators: 60             # Reduced for faster training
  max_depth: 6                 # Reduced complexity
  min_samples_split: 8         # Increased for simpler trees
  min_samples_leaf: 15         # Increased for generalization
  n_jobs: 2                    # Use 2 cores, leave 2 for system
  sessions_to_use: 2           # Use 2 sessions (less memory)
```

#### Dynamic Model Loading
```yaml
model_loading:
  cache_size: 3                    # Optimized for 4GB Jetson Nano
  device: 'cpu'                    # Force CPU for memory efficiency
  models_directory: 'athlete_models_tensors_updated'
  enable_memory_monitoring: true
```

#### Jetson Optimization
```yaml
jetson:
  device_detection: true
  memory_limit_mb: 1800        # Conservative limit for 4GB system
  batch_size: 500              # Smaller batches for memory efficiency
  gpu_memory_fraction: 0.5     # Conservative GPU memory usage
```

#### Conflict Prevention
```yaml
prediction_check:
  enabled: true
  prediction_script_names:
    - 'test_30_players.py'
    - 'test_deployment1.py'
  check_interval_secs: 5
  max_wait_time_secs: 300
  force_training: false
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. ML Training Won't Start
**Problem**: "ml_training: Stopped (exit code: 1)"
**Root Cause**: Live prediction scripts are running (safety feature)
**Solutions**:
```bash
# Stop prediction processes
pkill -f "test_30_players.py"
pkill -f "test_deployment1.py"

# Then run training
python sup_ml_rf_training.py

# Or force training (use with caution)
export FORCE_TRAINING=true
python sup_ml_rf_training.py
```

#### 2. MQTT Connection Failed
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

#### 3. Memory Issues
**Problem**: Out of memory during training/prediction
**Solutions**:
```bash
# Check memory usage
free -h

# Reduce batch size in jetson_orin_32gb_config.yaml
# batch_size: 500  (instead of 1000)

# Use dynamic model loading (already enabled)
# cache_size: 3  (reduces memory usage by 83%)
```

#### 4. No Models Found
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

#### 5. Dynamic Model Loading Issues
**Problem**: Models not loading correctly
**Solutions**:
```bash
# Test dynamic loading system
python test_dynamic_loading.py

# Check cache configuration
grep -A 10 "model_loading:" jetson_orin_32gb_config.yaml

# Monitor memory usage
python -c "from dynamic_model_loader import DynamicModelLoader; loader = DynamicModelLoader(); print(loader.get_cache_info())"
```

### Debug Commands
```bash
# Check system status
python3 jetson_deploy.py check
python3 jetson_deploy.py validate
python3 jetson_deploy.py status

# Test data generation
python3 publisher.py 3 --mode training

# Monitor training
tail -f jetson_training.log

# Check prediction logs
tail -f logs/A*_D*_*.log
```

---

## ⚡ Performance Optimization

### Jetson Nano Specific Optimizations

#### Memory Management
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature (important for sustained performance)
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Monitor GPU usage
tegrastats
```

#### Dynamic Model Loading Optimization
- **Cache Size**: Start with 3-5 models, adjust based on available memory
- **Preloading**: Disable for memory-constrained systems
- **Memory Monitoring**: Enable for real-time tracking
- **Device Selection**: Use CPU for memory efficiency

#### Training Optimization
- **Reduce `n_estimators`** for faster training
- **Increase `n_jobs`** for parallel processing (max 2 on Jetson)
- **Use `sessions_to_use: 2`** for less data per model
- **Disable feature engineering** for basic models

#### Prediction Optimization
- **Use `USE_CUDA=1`** for GPU acceleration
- **Enable dynamic model loading** for memory savings
- **Adjust `data_points_per_second`** in publisher
- **Monitor cache hit rates** for performance tuning

### Performance Monitoring
```bash
# Monitor system resources
watch -n 1 'free -h && echo "---" && nvidia-smi'

# Monitor temperature
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Monitor GPU usage
tegrastats
```

---

## 📁 Project Structure

```
jetson_nano_test/                   # Project root directory
├── Core Scripts
│   ├── publisher.py                # Multi-mode data generation
│   ├── test_30_players.py         # Live prediction engine (multi-device)
│   ├── test_deployment1.py        # Live prediction engine (single-device)
│   ├── sup_ml_rf_training.py      # ML model training system
│   ├── dynamic_model_loader.py    # Dynamic model loading system
│   └── jetson_deploy.py           # Deployment helper script
├── Configuration
│   ├── jetson_orin_32gb_config.yaml # Unified system configuration (32GB optimized)
│   └── requirements.txt           # Python dependencies
├── Training Data
│   ├── athlete_training_data/     # Training data storage
│   │   ├── player_1/
│   │   │   ├── TR1_A1_D001_*.csv
│   │   │   ├── TR2_A1_D001_*.csv
│   │   │   └── TR3_A1_D001_*.csv
│   │   └── player_2/...
│   └── athlete_game_data/         # Game data storage
│       ├── player_1/
│       │   ├── GM1_A1_D001_*.csv
│       │   └── GM2_A1_D001_*.csv
│       └── player_2/...
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
│   │   ├── A1_D001_*.log
│   │   ├── A2_D002_*.log
│   │   └── ...
│   ├── prediction_outputs/        # Real-time prediction outputs
│   │   ├── A1_Device_001/
│   │   │   ├── A1_D001_realtime_output.json
│   │   │   └── A1_D001_session_summary_*.json
│   │   ├── A2_Device_002/
│   │   └── ... (up to A30_Device_030)
│   └── jetson_training.log        # ML training log
├── System Files
│   ├── .prediction_running.lock   # Prediction active indicator
│   ├── .training_running.lock     # Training active indicator
│   └── .prediction_running.lock   # Dynamic loading lockfile
└── Documentation
    ├── README.md                  # This comprehensive guide
    ├── DYNAMIC_MODEL_LOADING_README.md
    └── Other documentation files
```

---

## 📚 Documentation

### Complete Data Flow Analysis

#### Flow 1: Training Data Generation & ML Training
```
1. PUBLISHER (Training Data Generation)
   ├── Generates: Realistic sensor data (10 Hz sampling rate)
   ├── MQTT Topics: player/{device_id}/sensor/data
   ├── Storage: In-memory collection during session
   └── On Stop (Ctrl+C): Saves to CSV files

2. DATA SAVING (Automatic on Publisher Stop)
   ├── Directory: athlete_training_data/player_{id}/
   ├── Filename Pattern: TR{seq}_A{athlete_id}_D{device_id}_{timestamp}.csv
   └── Status: Ready for ML training

3. ML TRAINING (Triggered by new data)
   ├── Conflict Check: Ensures no live prediction running
   ├── Player Scan: All 30 players for new/updated TR files
   ├── Feature Engineering: Rolling stats, FFT, jerk calculations
   ├── Model Training: RandomForest Regressor
   └── Model Saving: PKL + Hummingbird formats
```

#### Flow 2: Live Prediction During Actual Play
```
1. PUBLISHER (Real Sensor Data)
   ├── Source: Real athlete sensors OR simulated data
   ├── MQTT Topics: player/{device_id}/sensor/data
   └── Real-time streaming: Continuous data flow

2. PREDICTION ENGINE (test_30_players.py)
   ├── Dynamic Model Loading: Loads models on-demand
   ├── Multi-device Context: Separate state per player
   ├── Sensor Fusion: Madgwick filter for orientation
   ├── ML Prediction: Heart rate using player-specific model
   ├── Health Metrics: Stress, VO2 Max, TRIMP, energy expenditure
   └── Output Generation: Real-time metrics JSON
```

### Key Integration Points

| Component | Training Flow | Prediction Flow |
|-----------|---------------|-----------------|
| **Publisher** | Generates training data for model development | Provides real sensor data for live monitoring |
| **MQTT Topics** | `player/{device}/sensor/data` | `player/{device}/sensor/data` |
| **Data Format** | Identical structure, stored for training | Identical structure, processed in real-time |
| **Conflict Prevention** | Checks for running prediction before starting | Creates lockfile to block training during prediction |
| **Models** | Creates and updates ML models | Consumes existing trained models via dynamic loading |
| **Output** | TR*.csv training files | Real-time predictions and health metrics |

### Conflict Prevention System

The system prevents simultaneous training and prediction that could cause:
- Model file corruption during updates
- Prediction instability during training
- Resource contention and memory issues
- Inconsistent model states

**Implementation**:
- Process detection using `psutil`
- Lockfile mechanism (`.prediction_running.lock`)
- User interaction with timeout
- Configuration override options

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Jetson Nano
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA for Jetson Nano platform
- Scikit-learn for ML algorithms
- Hummingbird ML for model optimization
- MQTT community for communication protocols

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the configuration settings
3. Check system logs
4. Create an issue in the repository

---

**Ready for Jetson Nano deployment with dynamic model loading and memory optimization!** 🚀

The system now provides:
- ✅ **83% memory reduction** with dynamic model loading
- ✅ **Jetson Orin 32GB optimized** configuration
- ✅ **Real-time performance monitoring** for up to 30 athletes
- ✅ **Automated ML training** with conflict prevention
- ✅ **Comprehensive health metrics** and session management
- ✅ **Multi-mode data generation** for training and game scenarios
