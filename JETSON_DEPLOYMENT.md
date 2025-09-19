# Jetson Nano ML Training System Deployment Guide

This guide helps you deploy the ML training system on Jetson Nano with configurable settings.

## Quick Start

1. **Check Requirements**
   ```bash
   python jetson_deploy.py check
   ```

2. **Setup System**
   ```bash
   python jetson_deploy.py setup
   ```

3. **Run Training**
   ```bash
   python jetson_deploy.py run
   ```

## Configuration

All settings are controlled by `jetson_config.yaml`. No code changes needed!

### Key Configuration Sections

#### Training Parameters (Jetson Optimized)
```yaml
training:
  n_estimators: 100          # Reduced from 300 for Jetson
  max_depth: 8               # Reduced from 12 for Jetson
  min_samples_split: 5       # Reduced from 10 for Jetson
  min_samples_leaf: 10       # Reduced from 20 for Jetson
  n_jobs: 2                  # Limited for Jetson (was -1)
  sessions_to_use: 3         # Number of latest sessions
```

#### Jetson Nano Optimizations
```yaml
jetson:
  device_detection: true     # Auto-detect CUDA/CPU
  memory_limit_mb: 2048      # Memory limit for training
  gpu_memory_fraction: 0.7   # GPU memory fraction to use
  batch_size: 1000           # Batch size for processing
```

#### Monitoring Settings
```yaml
monitoring:
  poll_interval_secs: 10     # Check for updates every 10s
  debounce_secs: 3           # Wait 3s after file change
  auto_update_enabled: true  # Enable auto-updates
```

#### Retraining Logic
```yaml
retraining:
  accuracy_threshold: 0.85   # Retrain if accuracy < 85%
  min_samples_threshold: 500 # Retrain if samples > 500
  force_retrain_on_new_data: true  # Always retrain on new data
```

## Directory Structure

```
jetson_nano_test/
├── jetson_config.yaml              # Configuration file
├── jetson_deploy.py                # Deployment helper
├── sup_ml_rf_training.py           # Main training script
├── athlete_training_data/          # Training data
│   ├── player_1/
│   │   ├── TR1_A1_D001_*.csv
│   │   ├── TR2_A1_D001_*.csv
│   │   └── TR3_A1_D001_*.csv
│   └── player_2/...
├── athlete_models_pkl/             # PKL models
├── athlete_models_tensors_updated/ # Current HB models
└── athlete_models_tensors_previous/ # Backup HB models
```

## How It Works

### Rolling Window Training
- **Initial**: Uses first 3 sessions (TR1, TR2, TR3)
- **Updates**: When TR4 arrives, retrains with latest 3 (TR2, TR3, TR4)
- **Backup**: Previous models saved to `athlete_models_tensors_previous`

### Training Triggers
Training starts when:
- No model exists for player
- New session data is added
- Model accuracy drops below threshold
- Dataset grows beyond threshold

### Jetson Optimizations
- Reduced model complexity for limited resources
- GPU memory management
- Optimized batch processing
- Limited parallel processing

## Commands

### Check System
```bash
python jetson_deploy.py check      # Check requirements
python jetson_deploy.py validate   # Validate config
python jetson_deploy.py status     # Show system status
```

### Setup and Run
```bash
python jetson_deploy.py setup      # Create directories
python jetson_deploy.py run        # Start training
```

### Manual Training
```bash
python sup_ml_rf_training.py       # Run training directly
```

## Customization

### Adjust for Different Hardware
Edit `jetson_config.yaml`:

**For More Powerful Hardware:**
```yaml
training:
  n_estimators: 200
  max_depth: 12
  n_jobs: 4
jetson:
  memory_limit_mb: 4096
```

**For Less Powerful Hardware:**
```yaml
training:
  n_estimators: 50
  max_depth: 6
  n_jobs: 1
jetson:
  memory_limit_mb: 1024
```

### Change Training Logic
```yaml
retraining:
  accuracy_threshold: 0.9    # Higher accuracy requirement
  min_samples_threshold: 1000 # Larger dataset threshold
  force_retrain_on_new_data: false  # Don't always retrain
```

### Adjust Monitoring
```yaml
monitoring:
  poll_interval_secs: 30     # Check less frequently
  debounce_secs: 5           # Longer debounce
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `memory_limit_mb`
   - Lower `n_estimators` and `max_depth`
   - Set `n_jobs: 1`

2. **Slow Training**
   - Increase `n_jobs` (if memory allows)
   - Reduce `sessions_to_use`
   - Lower `batch_size`

3. **GPU Issues**
   - Set `device_detection: false` to force CPU
   - Reduce `gpu_memory_fraction`

### Logs
Check `jetson_training.log` for detailed logs.

### Status Check
```bash
python jetson_deploy.py status
```

## Performance Tips

1. **Jetson Nano Specific:**
   - Use `n_jobs: 2` maximum
   - Keep `n_estimators` under 150
   - Monitor memory usage

2. **Data Management:**
   - Use SSD storage for better I/O
   - Keep session files organized
   - Regular cleanup of old backups

3. **Monitoring:**
   - Adjust poll intervals based on data arrival frequency
   - Use appropriate debounce times
   - Monitor system resources

## Support

For issues or questions:
1. Check logs in `jetson_training.log`
2. Run `python jetson_deploy.py status`
3. Validate configuration with `python jetson_deploy.py validate`



# `Prediction Conflict Prevention System`

This document describes the prediction conflict prevention system implemented to ensure ML model training doesn't interfere with live prediction processes.

## Overview

The system prevents ML model training from starting when live prediction scripts are running. This prevents:
- Model file conflicts during training
- Prediction instability during model updates
- Resource contention between training and prediction
- Data corruption from concurrent model access

## How It Works

### 1. Process Detection
The system uses `psutil` to scan running processes and detect if prediction scripts are active:
- Scans for `test_30_players.py` and `test_deployment1.py` processes
- Checks command line arguments to identify the correct scripts
- Works across different Python interpreters and execution methods

### 2. Lockfile Mechanism
As a backup to process detection, the system uses lockfiles:
- Prediction scripts create `.prediction_running.lock` when they start
- Training scripts check for this lockfile before starting
- Stale lockfiles (from crashed processes) are automatically cleaned up
- Training creates `.training_running.lock` to indicate training is in progress

### 3. User Interaction
When a conflict is detected, the training script:
- Displays a clear warning message
- Asks the user if they want to wait for prediction to stop
- Provides a timeout mechanism (default: 5 minutes)
- Allows manual override through configuration

## Configuration

Edit `jetson_config.yaml` to customize the behavior:

```yaml
prediction_check:
  enabled: true                    # Enable/disable conflict checking
  prediction_script_names:         # Scripts to monitor
    - 'test_30_players.py'
    - 'test_deployment1.py'
  lockfile_path: '.prediction_running.lock'
  check_interval_secs: 5           # How often to check (seconds)
  max_wait_time_secs: 300          # Max wait time (seconds)
  force_training: false            # Force training (DANGEROUS!)
```

## Usage

### Normal Operation

1. **Starting Prediction**: 
   ```bash
   python test_30_players.py
   ```
   - Creates lockfile automatically
   - Displays lockfile creation message

2. **Starting Training**:
   ```bash
   python sup_ml_rf_training.py
   ```
   - Checks for running prediction automatically
   - Shows conflict detection results
   - Prompts user for action if conflict detected

### Manual Override

If you need to force training (not recommended):

```yaml
# In jetson_config.yaml
prediction_check:
  force_training: true
```

Or disable checking entirely:
```yaml
prediction_check:
  enabled: false
```

## Testing

Run the test suite to verify the system works:

```bash
python test_prediction_conflict.py
```

This will:
- Test lockfile creation and cleanup
- Test process detection
- Simulate prediction/training conflicts
- Verify proper conflict prevention

## File Locations

- **Configuration**: `jetson_config.yaml`
- **Prediction Lockfile**: `.prediction_running.lock`
- **Training Lockfile**: `.training_running.lock`
- **Test Script**: `test_prediction_conflict.py`

## Error Handling

### Common Scenarios

1. **Stale Lockfile**: Automatically detected and removed
2. **Process Crash**: Lockfiles cleaned up on next run
3. **Permission Issues**: Warnings logged, system continues
4. **Missing Config**: Uses sensible defaults

### Troubleshooting

**Training won't start**:
- Check if prediction scripts are running: `ps aux | grep python`
- Look for lockfiles: `ls -la .*.lock`
- Check configuration: `prediction_check.enabled` in config

**False positives**:
- Adjust `prediction_script_names` in configuration
- Check for other Python scripts with similar names

**System not working**:
- Run test suite: `python test_prediction_conflict.py`
- Check logs in `jetson_training.log`
- Verify `psutil` is installed: `pip install psutil`

## Implementation Details

### Modified Files

1. **sup_ml_rf_training.py**:
   - Added prediction checking functions
   - Modified main execution to check before training
   - Added lockfile management
   - Added signal handlers for cleanup

2. **test_30_players.py**:
   - Added lockfile creation on startup
   - Added signal handlers for cleanup
   - Added lockfile removal on exit

3. **test_deployment1.py**:
   - Same modifications as test_30_players.py

4. **jetson_config.yaml**:
   - Added prediction_check configuration section

### Key Functions

- `is_prediction_script_running()`: Main detection function
- `check_prediction_before_training()`: User interaction and decision
- `wait_for_prediction_to_stop()`: Wait mechanism with timeout
- `create_prediction_lockfile()`: Lockfile creation
- `remove_prediction_lockfile()`: Lockfile cleanup

## Safety Features

- **Automatic cleanup**: Lockfiles removed on normal exit
- **Signal handling**: Lockfiles removed on Ctrl+C or kill
- **Stale detection**: Old lockfiles automatically cleaned
- **Timeout protection**: Won't wait indefinitely
- **User control**: Always asks before waiting
- **Configuration override**: Can be disabled if needed

## Best Practices

1. **Always stop prediction before training** for best results
2. **Monitor logs** for conflict detection messages
3. **Use test suite** to verify system works in your environment
4. **Keep configuration updated** if you add new prediction scripts
5. **Don't use force_training** unless absolutely necessary

## Future Enhancements

Potential improvements:
- Network-based coordination for distributed systems
- More sophisticated process detection
- Integration with job schedulers
- Automatic training queuing when prediction stops
- Real-time status dashboard
