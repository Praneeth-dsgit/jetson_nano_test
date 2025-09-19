# Jetson Nano 4GB Migration Guide

## ✅ **YES, your project can be migrated to Jetson Nano 4GB!**

Your ML training system is well-designed for Jetson Nano deployment. Here's a comprehensive migration guide.

---

## 🔍 **Compatibility Analysis**

### **System Requirements vs Jetson Nano 4GB**

| Component | Requirement | Jetson Nano 4GB | Status |
|-----------|-------------|------------------|---------|
| **RAM** | ~2GB for training, ~1GB for prediction | 4GB total | ✅ **Compatible** |
| **GPU** | CUDA support (optional) | 128-core Maxwell GPU | ✅ **Supported** |
| **Storage** | ~1GB for models + data | microSD 32GB+ | ✅ **Sufficient** |
| **CPU** | Multi-core for parallel processing | Quad-core ARM A57 | ✅ **Adequate** |
| **Python** | 3.7+ | Ubuntu 18.04 with Python 3.6+ | ✅ **Compatible** |

### **Current Configuration Analysis**
```yaml
# Current settings (already Jetson-optimized!)
jetson:
  memory_limit_mb: 2048      # ✅ 2GB limit fits in 4GB system
  batch_size: 1000           # ✅ Reasonable for Jetson
  gpu_memory_fraction: 0.7   # ✅ 70% GPU memory usage
  
training:
  n_estimators: 100          # ✅ Optimized for Jetson (not 300+)
  max_depth: 8               # ✅ Reasonable depth
  n_jobs: 2                  # ✅ Conservative parallelism
```

---

## 📦 **Migration Steps**

### **Step 1: Prepare Jetson Nano**

#### **1.1 Flash JetPack**
```bash
# Use NVIDIA SDK Manager or balenaEtcher
# Recommended: JetPack 4.6.1 or newer
# Includes: Ubuntu 18.04, CUDA, cuDNN, TensorRT
```

#### **1.2 Initial Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-dev build-essential
sudo apt install -y mosquitto mosquitto-clients
sudo apt install -y git curl wget

# Enable maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

### **Step 2: Transfer Project Files**

#### **2.1 Copy Project Directory**
```bash
# Option A: USB/microSD transfer
cp -r /path/to/jetson_nano_test /home/jetson/

# Option B: Git clone (if using version control)
git clone <your-repo> /home/jetson/jetson_nano_test

# Option C: SCP transfer
scp -r jetson_nano_test/ jetson@<jetson-ip>:/home/jetson/
```

#### **2.2 Set Permissions**
```bash
cd /home/jetson/jetson_nano_test
chmod +x *.py
chmod 644 *.yaml *.md *.txt
```

### **Step 3: Install Dependencies**

#### **3.1 Python Dependencies**
```bash
cd /home/jetson/jetson_nano_test

# Install pip packages
pip3 install -r requirements.txt

# If torch installation fails, use Jetson-specific wheel:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **3.2 Verify Installation**
```bash
python3 jetson_deploy.py check
python3 jetson_deploy.py validate
python3 jetson_deploy.py setup
```

### **Step 4: Optimize for Jetson Nano 4GB**

#### **4.1 Update Configuration**
Edit `jetson_config.yaml` for optimal Jetson performance:

```yaml
# Jetson Nano 4GB Optimized Settings
jetson:
  device_detection: true
  memory_limit_mb: 1800        # Reduced for 4GB system (leave ~2GB for OS)
  batch_size: 500              # Smaller batches for memory efficiency  
  use_mixed_precision: true    # Enable for memory savings
  gpu_memory_fraction: 0.6     # Conservative GPU memory usage

training:
  n_estimators: 80             # Slightly reduced for faster training
  max_depth: 6                 # Reduced depth for memory efficiency
  min_samples_split: 8         # Increased for simpler trees
  min_samples_leaf: 15         # Increased for generalization
  n_jobs: 2                    # Use 2 cores (leave 2 for system)
  sessions_to_use: 2           # Use 2 sessions instead of 3 (less memory)

feature_engineering:
  enabled: true
  rolling_window: 8            # Reduced window size
  sampling_frequency: 10       # Keep at 10Hz
```

#### **4.2 Set Environment Variables**
```bash
# Add to ~/.bashrc
echo 'export USE_CUDA=1' >> ~/.bashrc
echo 'export PYTHONPATH=/home/jetson/jetson_nano_test:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 🚀 **Deployment Instructions**

### **Quick Deployment**
```bash
# 1. Navigate to project
cd /home/jetson/jetson_nano_test

# 2. Check system
python3 jetson_deploy.py check

# 3. Setup directories  
python3 jetson_deploy.py setup

# 4. Generate some training data
python3 publisher.py 5 --mode training

# 5. Train models
python3 sup_ml_rf_training.py

# 6. Test prediction
python3 test_30_players.py
```

### **Performance Testing**
```bash
# Monitor system resources during operation
# Terminal 1: Start monitoring
watch -n 1 'free -h && echo "---" && nvidia-smi'

# Terminal 2: Run training
python3 sup_ml_rf_training.py

# Terminal 3: Check temperature
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

---

## ⚡ **Jetson Nano Optimizations**

### **Memory Management**
```python
# The system already includes:
- Memory monitoring (print_detailed_memory_usage)
- GPU memory fraction control (0.6-0.7)
- Batch size optimization (500-1000)
- Process memory limits (1800-2048 MB)
```

### **GPU Acceleration**
```python
# Automatic CUDA detection:
if torch.cuda.is_available():
    device = "cuda"
    # GPU acceleration for Hummingbird models
else:
    device = "cpu"
    # CPU fallback
```

### **Model Optimization**
```yaml
# Jetson-optimized model parameters:
training:
  n_estimators: 80           # Faster training, good accuracy
  max_depth: 6               # Prevents overfitting, saves memory
  min_samples_leaf: 15       # Better generalization
```

---

## 📊 **Expected Performance on Jetson Nano 4GB**

### **Training Performance**
- **Single Player Model**: ~30-60 seconds
- **30 Player Models**: ~15-30 minutes  
- **Memory Usage**: ~1.5-2GB during training
- **GPU Acceleration**: 2-3x faster than CPU-only

### **Prediction Performance**
- **Real-time Prediction**: <50ms per player
- **Multi-player Support**: Up to 30 players simultaneously
- **Memory Usage**: ~800MB-1.2GB during prediction
- **Throughput**: 10 Hz per player sustained

### **Resource Usage**
- **CPU**: 60-80% during training, 30-50% during prediction
- **GPU**: 40-60% utilization with CUDA
- **RAM**: 1.5-2GB peak usage
- **Storage**: ~500MB for 30 player models

---

## 🛠️ **Jetson-Specific Optimizations Already Included**

### **1. Memory Management**
```python
# Automatic memory monitoring
print_detailed_memory_usage()

# GPU memory management  
torch.cuda.set_per_process_memory_fraction(0.6)

# Process memory limits
memory_limit_mb: 1800
```

### **2. Model Optimization**
```python
# Hummingbird conversion for GPU acceleration
hb_model = convert(model, backend="pytorch", device="cuda")

# Fallback to CPU if GPU fails
try:
    m.to("cuda")
except:
    m.to("cpu")
```

### **3. Batch Processing**
```python
# Configurable batch sizes
batch_size: 500  # Optimized for Jetson

# Adaptive input handling
def _safe_prepare_input(base_features, required_dim):
    # Handles variable input sizes efficiently
```

### **4. Resource Monitoring**
```python
# Built-in monitoring
- CPU usage tracking
- Memory usage alerts  
- GPU utilization monitoring
- Temperature monitoring (tegrastats)
```

---

## 🔧 **Migration Checklist**

### **Pre-Migration**
- [ ] Backup current project
- [ ] Test project on current system
- [ ] Verify all dependencies in requirements.txt
- [ ] Check MQTT broker configuration

### **During Migration**
- [ ] Flash JetPack to Jetson Nano
- [ ] Transfer project files
- [ ] Install Python dependencies
- [ ] Configure MQTT broker
- [ ] Update jetson_config.yaml for 4GB optimization

### **Post-Migration Testing**
- [ ] Run dependency check: `python3 jetson_deploy.py check`
- [ ] Validate configuration: `python3 jetson_deploy.py validate`
- [ ] Test data generation: `python3 publisher.py 3 --mode training`
- [ ] Test ML training: `python3 sup_ml_rf_training.py`
- [ ] Test prediction: `python3 test_30_players.py`
- [ ] Monitor resource usage during operation

---

## 🎯 **Recommended Jetson Nano 4GB Configuration**

```yaml
# Optimized jetson_config.yaml for 4GB Jetson Nano
jetson:
  device_detection: true
  memory_limit_mb: 1800        # Leave ~2GB for OS and other processes
  batch_size: 500              # Smaller batches for memory efficiency
  use_mixed_precision: true    # Enable for memory savings
  gpu_memory_fraction: 0.5     # Conservative GPU memory (128MB GPU has ~2GB shared)

training:
  n_estimators: 60             # Faster training, still good accuracy
  max_depth: 6                 # Reduced complexity
  min_samples_split: 10        # More conservative splits
  min_samples_leaf: 20         # Better generalization
  n_jobs: 2                    # Use 2 cores, leave 2 for system
  sessions_to_use: 2           # Use 2 sessions (less memory)

feature_engineering:
  rolling_window: 8            # Smaller window for memory efficiency
  sampling_frequency: 10       # Keep 10Hz sampling
```

---

## 🚨 **Potential Issues & Solutions**

### **Memory Limitations**
**Issue**: 4GB shared between CPU and GPU
**Solutions**:
- Reduce `memory_limit_mb` to 1500-1800
- Use `sessions_to_use: 2` instead of 3
- Enable `use_mixed_precision: true`
- Reduce `batch_size` to 300-500

### **Storage Limitations**
**Issue**: microSD card storage
**Solutions**:
- Use Class 10 microSD (32GB+)
- Enable model backup rotation
- Periodic cleanup of old logs
- Compress training data if needed

### **Performance Optimization**
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperature (important for sustained performance)
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Monitor GPU usage
tegrastats
```

---

## 📁 **File Transfer Methods**

### **Method 1: Direct Copy (microSD)**
1. Remove microSD from Jetson
2. Copy project folder to microSD on PC
3. Insert microSD back into Jetson

### **Method 2: Network Transfer**
```bash
# From your PC:
scp -r jetson_nano_test/ jetson@<jetson-ip>:/home/jetson/

# Or using rsync:
rsync -av jetson_nano_test/ jetson@<jetson-ip>:/home/jetson/jetson_nano_test/
```

### **Method 3: USB Transfer**
1. Copy project to USB drive
2. Connect USB to Jetson
3. Copy from USB to Jetson home directory

---

## 🎉 **Expected Results**

After migration, you'll have:

✅ **Full ML training system** running on Jetson Nano
✅ **Real-time prediction** for up to 30 athletes  
✅ **GPU acceleration** for faster inference
✅ **Automatic model updates** when new data arrives
✅ **Conflict prevention** between training and prediction
✅ **Organized data storage** with proper file structure
✅ **Memory-efficient operation** within 4GB constraints

### **Performance Expectations**
- **Training**: 30 models in ~20-40 minutes
- **Prediction**: Real-time for 10-30 players simultaneously
- **Memory**: ~1.5GB peak usage during training
- **Temperature**: Monitor to prevent throttling

---

## 🛡️ **Safety Recommendations**

1. **Temperature Monitoring**: Use cooling fan, monitor thermal throttling
2. **Power Supply**: Use official 5V/4A power adapter
3. **Storage**: Use high-quality microSD card (Class 10, A1)
4. **Backup**: Regular backup of trained models
5. **Testing**: Gradual load testing (start with fewer players)

---

**Your system is ready for Jetson Nano deployment! The configuration is already optimized for embedded hardware.** 🚀
