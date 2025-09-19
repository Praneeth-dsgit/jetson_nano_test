# 🔧 ML Training Troubleshooting Guide

## ❌ Common Issue: "ml_training: Stopped (exit code: 1)"

### 🎯 **Root Cause**
The ML training process is failing with exit code 1 because it detects that live prediction scripts are running. This is a **safety feature** to prevent conflicts between training and prediction processes.

### 🔍 **Why This Happens**
1. **Live prediction is running** (`test_30_players.py` or `test_deployment1.py`)
2. **ML training is blocked** to prevent model conflicts
3. **Process exits with code 1** to indicate the conflict

### ✅ **Solutions**

#### **Option 1: Use the Frontend (Recommended)**
1. **Open the Streamlit frontend**: `streamlit run streamlit_app.py`
2. **Check Process Status**: Look at the left panel to see running processes
3. **If prediction is running**: You'll see a warning and two options:
   - **🚀 Force Training**: Override the safety check and train anyway
   - **⏳ Wait & Train**: Stop prediction processes first, then train

#### **Option 2: Stop Prediction Processes First**
```bash
# Stop any running prediction processes
# Check what's running
ps aux | grep python

# Kill prediction processes (replace PID with actual process ID)
kill <PID_of_test_30_players.py>
kill <PID_of_test_deployment1.py>

# Then run ML training
python sup_ml_rf_training.py
```

#### **Option 3: Force Training (Advanced)**
```bash
# Set environment variable to force training
export FORCE_TRAINING=true
python sup_ml_rf_training.py

# Or on Windows
set FORCE_TRAINING=true
python sup_ml_rf_training.py
```

### 🛡️ **Safety Features Explained**

#### **Why the Conflict Check Exists**
- **Model Stability**: Prevents training from corrupting models while they're being used
- **Data Integrity**: Ensures prediction data isn't affected by training
- **System Reliability**: Avoids crashes and unexpected behavior

#### **When to Use Force Training**
- **Development/Testing**: When you need to test training while prediction runs
- **Emergency Updates**: When models need urgent updates
- **Controlled Environment**: When you're sure it's safe

### 📊 **Process Status Indicators**

#### **In the Frontend**
- ✅ **Green**: Process running successfully
- ❌ **Red**: Process stopped/failed
- ⚠️ **Yellow**: Warning about conflicts

#### **Exit Codes**
- **Exit Code 0**: Success
- **Exit Code 1**: Conflict detected (prediction running)
- **Exit Code 2**: Other error (check logs)

### 🔄 **Recommended Workflow**

#### **Normal Training Session**
1. **Stop all prediction processes**
2. **Run ML training**: `python sup_ml_rf_training.py`
3. **Start prediction processes** after training completes

#### **Using the Frontend**
1. **Launch frontend**: `streamlit run streamlit_app.py`
2. **Check process status** in left panel
3. **Use appropriate training option** based on current state
4. **Monitor progress** in real-time

### 🚨 **Emergency Recovery**

#### **If Training Gets Stuck**
```bash
# Find and kill training process
ps aux | grep sup_ml_rf_training.py
kill <PID>

# Remove lockfile if it exists
rm -f .training_running.lock
```

#### **If Prediction Gets Stuck**
```bash
# Find and kill prediction processes
ps aux | grep test_30_players.py
ps aux | grep test_deployment1.py
kill <PID>

# Remove prediction lockfile
rm -f .prediction_running.lock
```

### 📝 **Log Files**
Check these files for detailed error information:
- `logs/` directory for training logs
- Console output for real-time status
- `jetson_training.log` for ML training logs

### 🎯 **Quick Fixes**

#### **Most Common Solution**
```bash
# Stop everything
pkill -f "test_30_players.py"
pkill -f "test_deployment1.py"
pkill -f "sup_ml_rf_training.py"

# Wait a moment
sleep 2

# Start training
python sup_ml_rf_training.py
```

#### **Using the Frontend**
1. Open `streamlit run streamlit_app.py`
2. Look for "🛑 Stop Processes" section
3. Click "Stop" for any running processes
4. Click "🔄 Update System" to start training

### 💡 **Pro Tips**

1. **Always check process status** before starting training
2. **Use the frontend** for easier process management
3. **Monitor logs** for detailed error information
4. **Keep backups** of working models before training
5. **Test in development** before production updates

---

**Need Help?** Check the console output and log files for specific error messages. The frontend provides the easiest way to manage these processes safely.
