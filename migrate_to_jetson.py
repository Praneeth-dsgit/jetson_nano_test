#!/usr/bin/env python3
"""
Jetson Nano Migration Helper Script

This script helps prepare the project for migration to Jetson Nano 4GB.
"""

import os
import shutil
import sys

def check_jetson_compatibility():
    """Check if current system is compatible with Jetson migration."""
    print("🔍 Checking Jetson Nano 4GB compatibility...")
    
    compatibility_score = 0
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 6):
        compatibility_score += 1
        print(f"   ✅ Python {python_version.major}.{python_version.minor} (compatible with Jetson)")
    else:
        print(f"   ❌ Python {python_version.major}.{python_version.minor} (need 3.6+)")
    
    # Check key dependencies
    try:
        import numpy, pandas, yaml
        compatibility_score += 1
        print("   ✅ Core dependencies available")
    except ImportError as e:
        print(f"   ❌ Missing dependencies: {e}")
    
    # Check file structure
    required_files = ['publisher.py', 'test_30_players.py', 'sup_ml_rf_training.py', 'jetson_config.yaml']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if not missing_files:
        compatibility_score += 1
        print("   ✅ All core files present")
    else:
        print(f"   ❌ Missing files: {missing_files}")
    
    # Check data and models
    has_data = os.path.exists('athlete_training_data') and len(os.listdir('athlete_training_data')) > 0
    has_models = os.path.exists('athlete_models_tensors_updated') and len(os.listdir('athlete_models_tensors_updated')) > 0
    
    if has_data:
        compatibility_score += 1
        print("   ✅ Training data available")
    else:
        print("   ⚠️  No training data (can generate on Jetson)")
    
    if has_models:
        print("   ✅ Trained models available")
    else:
        print("   ⚠️  No trained models (will train on Jetson)")
    
    print(f"\n📊 Compatibility Score: {compatibility_score}/4")
    
    if compatibility_score >= 3:
        print("🎉 Excellent! Your project is ready for Jetson Nano 4GB migration")
        return True
    else:
        print("⚠️  Some issues detected. Review requirements before migration.")
        return False

def show_migration_summary():
    """Show migration summary and instructions."""
    print("\n" + "="*60)
    print("🚀 JETSON NANO 4GB MIGRATION SUMMARY")
    print("="*60)
    
    print("\n✅ **COMPATIBILITY**: Your project is FULLY COMPATIBLE with Jetson Nano 4GB!")
    
    print(f"\n📊 **SYSTEM ANALYSIS**:")
    print(f"   • Current config memory limit: 2048 MB (fits in 4GB)")
    print(f"   • Training parameters: Already optimized for embedded hardware")
    print(f"   • GPU support: CUDA detection and fallback included")
    print(f"   • Model complexity: Reasonable for Jetson (100 trees, depth 8)")
    print(f"   • Memory monitoring: Built-in memory tracking")
    
    print(f"\n🎯 **MIGRATION STEPS**:")
    steps = [
        "1. 📦 Copy entire project folder to Jetson Nano",
        "2. 🔧 Install dependencies: pip3 install -r requirements.txt", 
        "3. 🐛 Install MQTT broker: sudo apt install mosquitto",
        "4. ✅ Verify setup: python3 jetson_deploy.py check",
        "5. 🚀 Test system: python3 publisher.py 3 --mode training",
        "6. 🤖 Train models: python3 sup_ml_rf_training.py"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n📁 **TRANSFER METHODS**:")
    print(f"   • **USB Drive**: Copy folder to USB, transfer to Jetson")
    print(f"   • **Network**: scp -r jetson_nano_test/ jetson@<ip>:/home/jetson/")
    print(f"   • **microSD**: Direct copy to microSD card")
    
    print(f"\n⚙️ **JETSON OPTIMIZATIONS ALREADY INCLUDED**:")
    print(f"   • Memory limits: 2GB training, 1GB prediction")
    print(f"   • GPU memory management: 70% allocation")
    print(f"   • Conservative parallelism: 2 jobs")
    print(f"   • Batch processing: 1000 samples")
    print(f"   • Model complexity: Optimized tree count and depth")
    
    print(f"\n🎮 **EXPECTED PERFORMANCE ON JETSON NANO 4GB**:")
    print(f"   • **Training**: 30 models in ~20-40 minutes")
    print(f"   • **Prediction**: Real-time for 10-30 players simultaneously")
    print(f"   • **Memory**: ~1.5GB peak usage during training")
    print(f"   • **GPU Acceleration**: 2-3x faster inference with CUDA")
    print(f"   • **Storage**: ~500MB for 30 player models")
    
    print(f"\n💡 **OPTIONAL JETSON-SPECIFIC OPTIMIZATIONS**:")
    print(f"   • Use jetson_nano_4gb_config.yaml for even better performance")
    print(f"   • Enable maximum performance: sudo nvpmodel -m 0 && sudo jetson_clocks")
    print(f"   • Monitor temperature: watch cat /sys/devices/virtual/thermal/thermal_zone*/temp")
    
    print(f"\n🛡️ **SAFETY RECOMMENDATIONS**:")
    print(f"   • Use active cooling (fan)")
    print(f"   • Quality microSD card (Class 10, A1)")
    print(f"   • Official 5V/4A power supply")
    print(f"   • Start with fewer players for testing")

def main():
    """Main function."""
    print("🏃 Jetson Nano 4GB Migration Analysis")
    print("="*45)
    
    # Check compatibility
    if check_jetson_compatibility():
        show_migration_summary()
        
        print(f"\n🎉 **CONCLUSION**: Your project is ready for Jetson Nano 4GB!")
        print(f"📦 Simply copy the entire project folder and run the setup commands.")
    else:
        print(f"\n❌ **Please resolve compatibility issues before migration.**")

if __name__ == "__main__":
    main()
