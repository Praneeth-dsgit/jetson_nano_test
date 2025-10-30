#!/usr/bin/env python3
"""
Jetson Orin Deployment Helper Script
This script helps deploy and manage the ML training system on Jetson Orin.
Uses jetson_orin_32gb_config.yaml for optimized Jetson Orin 32GB settings.

Commands:
- check: Validate required packages are installed
- setup: Create necessary directories and validate configuration
- run: Run the ML training system
- status: Show system status and configuration
- validate: Validate configuration file structure
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'joblib', 'torch', 'hummingbird-ml', 'pyyaml', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[ERROR] Missing required packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("[INFO] All required packages are installed")
    return True

def create_directories():
    """Create necessary directories for the system."""
    config_path = "config/jetson_orin_32gb_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file {config_path} not found!")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    directories = [
        config['paths']['data_root'],
        config['paths']['models_root'],
        config['paths']['hb_tensors_updated'],
        config['paths']['hb_tensors_previous']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Created directory: {directory}")
    
    return True

def validate_config():
    """Validate the configuration file."""
    config_path = "config/jetson_orin_32gb_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file {config_path} not found!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections (updated for jetson_orin_32gb_config.yaml)
        required_sections = ['paths', 'training', 'monitoring', 'jetson', 'backup', 'retraining', 'model_loading']
        for section in required_sections:
            if section not in config:
                print(f"[ERROR] Missing configuration section: {section}")
                return False
        
        print(f"[INFO] Configuration file {config_path} is valid")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        return False

def run_training():
    """Run the ML training system."""
    print("[INFO] Starting ML training system...")
    try:
        subprocess.run([sys.executable, "sup_ml_rf_training.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user")
        return True
    return True

def show_status():
    """Show system status."""
    config_path = "config/jetson_orin_32gb_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file {config_path} not found!")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n=== Jetson Orin ML Training System Status ({config_path}) ===")
    
    # Show deployment info if available
    if 'deployment' in config:
        print(f"Environment: {config['deployment']['environment']}")
        print(f"Version: {config['deployment']['version']}")
    else:
        print("Environment: Jetson Orin 32GB Optimized")
        print("Version: 32GB Optimized Configuration")
    
    print(f"Training Parameters:")
    print(f"  - n_estimators: {config['training']['n_estimators']}")
    print(f"  - max_depth: {config['training']['max_depth']}")
    print(f"  - n_jobs: {config['training']['n_jobs']}")
    print(f"  - sessions_to_use: {config['training']['sessions_to_use']}")
    print(f"Monitoring:")
    print(f"  - poll_interval: {config['monitoring']['poll_interval_secs']}s")
    print(f"  - auto_update: {config['monitoring']['auto_update_enabled']}")
    print(f"Jetson Optimizations:")
    print(f"  - device_detection: {config['jetson']['device_detection']}")
    print(f"  - memory_limit: {config['jetson']['memory_limit_mb']}MB")
    print(f"  - gpu_memory_fraction: {config['jetson']['gpu_memory_fraction']}")
    
    # Show feature engineering if available
    if 'feature_engineering' in config:
        print(f"Feature Engineering:")
        print(f"  - enabled: {config['feature_engineering']['enabled']}")
        print(f"  - rolling_window: {config['feature_engineering']['rolling_window']}")
        print(f"  - sampling_frequency: {config['feature_engineering']['sampling_frequency']}")
        print(f"  - features: {list(config['feature_engineering']['features'].keys())}")
    
    # Show dynamic model loading configuration if available
    if 'model_loading' in config:
        print(f"Dynamic Model Loading:")
        print(f"  - cache_size: {config['model_loading']['cache_size']}")
        print(f"  - device: {config['model_loading']['device']}")
        print(f"  - models_directory: {config['model_loading']['models_directory']}")
        print(f"  - enable_memory_monitoring: {config['model_loading']['enable_memory_monitoring']}")
    
    # Check directories
    print(f"\nDirectories:")
    for key, path in config['paths'].items():
        if key.endswith('_root') or key.endswith('_updated') or key.endswith('_previous'):
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  - {path}: {exists}")

def main():
    parser = argparse.ArgumentParser(description="Jetson Orin ML Training Deployment Helper")
    parser.add_argument("command", choices=["check", "setup", "run", "status", "validate"], 
                       help="Command to execute")
    
    args = parser.parse_args()
    
    if args.command == "check":
        check_requirements()
    elif args.command == "validate":
        validate_config()
    elif args.command == "setup":
        if check_requirements() and validate_config():
            create_directories()
            print("[INFO] Setup completed successfully!")
        else:
            print("[ERROR] Setup failed!")
            sys.exit(1)
    elif args.command == "run":
        if check_requirements() and validate_config():
            run_training()
        else:
            print("[ERROR] Cannot run training - check requirements and config")
            sys.exit(1)
    elif args.command == "status":
        show_status()

if __name__ == "__main__":
    main()
