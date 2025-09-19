# Jetson Nano ML Training System

A comprehensive machine learning system for athlete performance analysis, designed specifically for deployment on NVIDIA Jetson Nano 4GB.

## 🚀 Features

- **Real-time ML Training**: Train Random Forest models for up to 30 athletes simultaneously
- **GPU Acceleration**: CUDA support with automatic fallback to CPU
- **Memory Optimization**: Designed for Jetson Nano 4GB constraints
- **MQTT Communication**: Real-time data streaming and model updates
- **Conflict Prevention**: Prevents training/prediction conflicts
- **Automatic Retraining**: Models update when new data arrives

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

## 🛠️ Installation

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
```

## 🎯 Quick Start

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

## 📁 Project Structure

```
jetson_nano_test/
├── athlete_training_data/     # Training data (CSV files)
├── athlete_game_data/         # Game data (CSV files)
├── athlete_models_pkl/        # Trained models (PKL files)
├── athlete_models_tensors_updated/  # Hummingbird models
├── prediction_outputs/        # Prediction results (JSON)
├── logs/                      # System logs
├── jetson_config.yaml        # Main configuration
├── jetson_nano_4gb_config.yaml  # 4GB optimized config
├── requirements.txt           # Python dependencies
├── publisher.py              # Data generation
├── sup_ml_rf_training.py    # ML training system
├── test_30_players.py       # Prediction testing
└── jetson_deploy.py         # Deployment utilities
```

## ⚙️ Configuration

### Main Configuration (`jetson_config.yaml`)
```yaml
jetson:
  memory_limit_mb: 1800        # Memory limit for 4GB system
  batch_size: 500              # Batch size for processing
  gpu_memory_fraction: 0.5     # GPU memory allocation

training:
  n_estimators: 80             # Number of trees
  max_depth: 6                 # Tree depth
  n_jobs: 2                    # Parallel jobs
  sessions_to_use: 2           # Training sessions
```

### 4GB Optimized Configuration (`jetson_nano_4gb_config.yaml`)
- Reduced memory limits
- Smaller batch sizes
- Conservative GPU usage
- Optimized for 4GB constraints

## 📊 Performance

### Training Performance
- **Single Player Model**: ~30-60 seconds
- **30 Player Models**: ~15-30 minutes
- **Memory Usage**: ~1.5-2GB peak
- **GPU Acceleration**: 2-3x faster than CPU-only

### Prediction Performance
- **Real-time Prediction**: <50ms per player
- **Multi-player Support**: Up to 30 players simultaneously
- **Memory Usage**: ~800MB-1.2GB during prediction
- **Throughput**: 10 Hz per player sustained

## 🔧 Key Components

### ML Training System (`sup_ml_rf_training.py`)
- Random Forest model training
- Feature engineering
- Model validation
- Automatic retraining
- Conflict prevention

### Data Publisher (`publisher.py`)
- Simulates athlete sensor data
- Generates training and game data
- MQTT communication
- Configurable data patterns

### Prediction System (`test_30_players.py`)
- Real-time prediction
- Multi-player support
- Performance monitoring
- Output generation

### Deployment Utilities (`jetson_deploy.py`)
- System validation
- Dependency checking
- Performance monitoring
- Setup automation

## 📈 Monitoring

### System Monitoring
```bash
# Monitor system resources
watch -n 1 'free -h && echo "---" && nvidia-smi'

# Monitor temperature
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Monitor GPU usage
tegrastats
```

### Log Files
- `jetson_training.log`: Training system logs
- `logs/`: Detailed system logs
- Console output with detailed progress

## 🚨 Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce `memory_limit_mb` in config
2. **GPU Issues**: Check CUDA installation
3. **MQTT Issues**: Verify broker installation
4. **Performance Issues**: Monitor temperature and throttling

### Debug Commands
```bash
# Check system status
python3 jetson_deploy.py check

# Validate configuration
python3 jetson_deploy.py validate

# Test data generation
python3 publisher.py 3 --mode training

# Monitor training
tail -f jetson_training.log
```

## 📚 Documentation

- `JETSON_DEPLOYMENT.md`: Detailed deployment guide
- `JETSON_NANO_MIGRATION_GUIDE.md`: Migration instructions
- `ML_TRAINING_TROUBLESHOOTING.md`: Troubleshooting guide
- `PUBLISHER_USAGE.md`: Data generation guide

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
1. Check the troubleshooting documentation
2. Review the migration guide
3. Check system logs
4. Create an issue in the repository

---

**Ready for Jetson Nano deployment!** 🚀
