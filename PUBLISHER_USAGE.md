# Multi-Mode Data Publisher - Complete Usage Guide

This guide explains how to use the enhanced `publisher.py` script for both training data generation and game simulation.

## Overview

The multi-mode publisher generates realistic sensor data with two distinct modes and supports flexible player selection. Key features:

- **🏃 Training Mode**: Structured training sessions with predictable patterns
- **⚽ Game Mode**: Dynamic game scenarios with realistic match conditions  
- **🎯 Specific Player Selection**: Choose exact players like [2,9,28,12,5]
- **🎲 Random Selection**: Generate data for 1-30 random players
- **📁 Organized Storage**: Separate folders for training vs game data
- **📊 Realistic Patterns**: Position-based activity, fatigue modeling, game events

## Command Line Syntax

```bash
python publisher.py <num_players> [--mode training|game] [--duration minutes] [--players list]
```

## Usage Examples

### Random Player Selection

#### Training Mode (saves to `athlete_training_data/`):
```bash
python publisher.py 30 --mode training                    # All 30 players
python publisher.py 5 --mode training                     # 5 random players  
python publisher.py 1 --mode training                     # 1 random player
```

#### Game Mode (saves to `athlete_game_data/`):
```bash
python publisher.py 11 --mode game --duration 90          # 11 players, 90-min game
python publisher.py 5 --mode game --duration 45           # 5 players, 45-min game
python publisher.py 22 --mode game                        # 22 players, default 90-min
```

### Specific Player Selection (NEW!)

#### Training Mode with Specific Players:
```bash
python publisher.py --players [1,3,7] --mode training     # Players 1, 3, 7 for training
python publisher.py --players [2,9,28,12,5] --mode training  # Players 2,9,28,12,5 for training
python publisher.py --players 1,5,10,15,20 --mode training   # Alternative format (no brackets)
```

#### Game Mode with Specific Players:
```bash
python publisher.py --players [2,9,28,12,5] --mode game   # Specific players for game
python publisher.py --players [11,22] --mode game --duration 45  # 45-min game for players 11,22
python publisher.py --players 7,14,21,28 --mode game      # Alternative format
```

## How It Works

### 1. Player Selection Methods

#### **Random Selection**:
- **All players**: `python publisher.py 30 --mode training`
- **Random subset**: `python publisher.py 5 --mode game`
- **Single random**: `python publisher.py 1 --mode training`

#### **Specific Selection** (NEW!):
- **Exact players**: `python publisher.py --players [2,9,28,12,5] --mode game`
- **Alternative format**: `python publisher.py --players 2,9,28,12,5 --mode training`
- **Validation**: Ensures all player IDs are between 1-30
- **Duplicate removal**: Automatically handles duplicates

### 2. Mode-Specific Data Generation

#### **🏃 Training Mode**:
- **Patterns**: Structured phases (warm-up → active → cool-down)
- **Duration**: Flexible 10-minute cycles
- **Intensity**: Predictable progression (70-145 bpm)
- **Characteristics**: Moderate acceleration variation, consistent patterns
- **File prefix**: TR (TR1, TR2, TR3...)
- **Storage**: `athlete_training_data/player_X/`

#### **⚽ Game Mode**:
- **Patterns**: Dynamic game events (sprints, tackles, shots, passes)
- **Duration**: Game-specific (45-minute halves, customizable)
- **Intensity**: Variable based on position and events (90-180 bpm)
- **Characteristics**: High acceleration variation, sudden spikes
- **Extra attributes**: Position, playing style, fitness level, fatigue rate
- **File prefix**: GM (GM1, GM2, GM3...)
- **Storage**: `athlete_game_data/player_X/`

### 3. Enhanced Data Features

#### **Player Attributes (All Modes)**:
- **Basic**: Age (18-35), Weight (60-100kg), Height (160-200cm), Gender
- **Sensor data**: Accelerometer, gyroscope, magnetometer, heart rate
- **Session info**: Phase, intensity level, mode

#### **Game Mode Additional Attributes**:
- **Position**: Forward, Midfielder, Defender, Goalkeeper
- **Playing Style**: Aggressive, Moderate, Conservative
- **Fitness Level**: 70-100% current fitness
- **Fatigue Rate**: Individual fatigue characteristics
- **Game Events**: Sprints, tackles, shots logged

### 4. Organized Data Storage
- **Real-time collection**: Data stored in memory during session
- **Automatic saving**: Triggered when you press Ctrl+C
- **Mode-based folders**: Training vs Game data separation
- **Proper sequencing**: Auto-increments sequence numbers
- **Enhanced format**: Additional fields for game analysis

## File Structure

### Training Mode Files (athlete_training_data/):
```
athlete_training_data/
├── player_1/
│   ├── TR1_A1_D001_2025_09_17-14_30_15.csv
│   ├── TR2_A1_D001_2025_09_17-15_45_20.csv
│   └── TR3_A1_D001_2025_09_17-16_30_25.csv
├── player_2/
│   ├── TR1_A2_D002_2025_09_17-14_30_15.csv
│   └── TR2_A2_D002_2025_09_17-15_45_20.csv
```

### Game Mode Files (athlete_game_data/):
```
athlete_game_data/
├── player_1/
│   ├── GM1_A1_D001_2025_09_17-18_30_15.csv
│   ├── GM2_A1_D001_2025_09_17-20_15_30.csv
│   └── GM3_A1_D001_2025_09_17-22_45_10.csv
├── player_2/
│   ├── GM1_A2_D002_2025_09_17-18_30_15.csv
│   └── GM2_A2_D002_2025_09_17-20_15_30.csv
└── ...
```

### Real-time Outputs (prediction_outputs/):
```
prediction_outputs/
├── A1_Device_001/
│   ├── A1_D001_realtime_output.json
│   └── A1_D001_session_summary_*.json
├── A2_Device_002/
│   └── ...
└── ...
```

### Filename Formats

#### **Training Files**:
`TR<sequence>_A<athlete_id>_D<device_id>_<timestamp>.csv`
- **TR<sequence>**: Training session number (TR1, TR2, TR3...)
- **A<athlete_id>**: Athlete ID (A1, A2, A3...)
- **D<device_id>**: Device ID (D001, D002, D003...)
- **<timestamp>**: Session start time (YYYY_MM_DD-HH_MM_SS)

#### **Game Files**:
`GM<sequence>_A<athlete_id>_D<device_id>_<timestamp>.csv`
- **GM<sequence>**: Game session number (GM1, GM2, GM3...)
- **A<athlete_id>**: Athlete ID (A1, A2, A3...)
- **D<device_id>**: Device ID (D001, D002, D003...)
- **<timestamp>**: Session start time (YYYY_MM_DD-HH_MM_SS)

## Data Formats

### Training Mode CSV Format:
```
timestamp,athlete_id,age,weight,height,gender,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate,session_phase,intensity_level,mode
```

### Game Mode CSV Format:
```
timestamp,athlete_id,age,weight,height,gender,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate,session_phase,intensity_level,mode,position,playing_style,fitness_level,fatigue_rate
```

### Field Descriptions

#### **Common Fields (Both Modes)**:
- **timestamp**: Data point timestamp (YYYY-MM-DD HH:MM:SS)
- **athlete_id**: Unique athlete identifier (1-30)
- **age**: Athlete age (18-35 years)
- **weight**: Athlete weight (60-100 kg)
- **height**: Athlete height (160-200 cm)
- **gender**: Male/Female (string format)
- **acc_x/y/z**: Accelerometer readings (m/s²)
- **gyro_x/y/z**: Gyroscope readings (rad/s)
- **heart_rate**: Heart rate (bpm)
- **session_phase**: Current phase (warmup/active/cooldown or first_half/halftime/second_half)
- **intensity_level**: Low/Medium/High
- **mode**: training/game

#### **Game Mode Additional Fields**:
- **position**: Forward/Midfielder/Defender/Goalkeeper
- **playing_style**: Aggressive/Moderate/Conservative
- **fitness_level**: Current fitness percentage (0.7-1.0)
- **fatigue_rate**: Individual fatigue characteristics (0.8-1.2)

## Usage Workflow

### Training Data Generation Workflow

#### Step 1: Start Training Data Collection
```bash
# Random selection
python publisher.py 5 --mode training

# Specific selection  
python publisher.py --players [1,5,10,15,20] --mode training
```

You'll see output like:
```
🏃 Training Session Publisher Started
📊 Generating data for 5 players: [1, 5, 10, 15, 20]
📡 Publishing to MQTT and collecting training data...
💾 Data will be saved to: athlete_training_data/

   Player 1: Age=28, Weight=75.2kg, Height=182.1cm, Gender=Male
   Player 5: Age=24, Weight=68.9kg, Height=175.8cm, Gender=Female
   ...
```

#### Step 2: Let It Collect Data
The script will:
- Generate structured training patterns (warm-up → active → cool-down)
- Publish data to MQTT topics at 10 Hz
- Store data in memory with progress indicators
- Show progress every 100 data points per player

#### Step 3: Stop and Save
Press **Ctrl+C** when you want to stop. You'll see:
```
🛑 Stopping data collection...
💾 Saving training data...
   ✅ Player 1: 1250 data points saved to TR2_A1_D001_2025_09_17-14_30_15.csv
   ✅ Player 5: 1250 data points saved to TR1_A5_D005_2025_09_17-14_30_15.csv
   ...

📊 TRAINING SESSION SUMMARY:
   Mode: Training 🏃
   Duration: 125.0 seconds
   Players: 5
   Total data points: 6250
   Data saved to: athlete_training_data/
```

### Game Data Generation Workflow

#### Step 1: Start Game Data Collection
```bash
# Random game players
python publisher.py 11 --mode game --duration 90

# Specific game players
python publisher.py --players [2,9,28,12,5] --mode game --duration 45
```

You'll see output like:
```
⚽ Game Session Publisher Started (90 min)
📊 Generating data for 5 players: [2, 9, 28, 12, 5]
📡 Publishing to MQTT and collecting game data...
💾 Data will be saved to: athlete_game_data/

   Player 2: Age=26, Weight=78.1kg, Height=180.5cm, Gender=Male
      Position: Midfielder, Style: Aggressive
   Player 9: Age=23, Weight=71.3kg, Height=175.2cm, Gender=Female  
      Position: Forward, Style: Moderate
   ...
```

#### Step 2: Dynamic Game Simulation
The script generates:
- **Dynamic events**: Sprints, tackles, shots, passes, dribbles
- **Position-based activity**: Different patterns for each position
- **Fatigue modeling**: Players tire over game duration
- **Variable intensity**: Based on game events and player style

#### Step 3: Stop and Save Game Data
```
🛑 Stopping data collection...
💾 Saving game data...
   ✅ Player 2: 2700 data points saved to GM1_A2_D002_2025_09_17-18_30_15.csv
      🎮 Game events: 12 significant actions
   ✅ Player 9: 2700 data points saved to GM1_A9_D009_2025_09_17-18_30_15.csv
      🎮 Game events: 8 significant actions

📊 GAME SESSION SUMMARY:
   Mode: Game ⚽
   Duration: 270.0 seconds
   Players: 5
   Game events: 45 significant actions recorded
   Event breakdown: {'sprint': 12, 'tackle': 8, 'shot': 6, 'pass': 19}
```

### Verification
Check the generated files:
```bash
# Training data
ls athlete_training_data/player_*/TR*.csv

# Game data  
ls athlete_game_data/player_*/GM*.csv

# Real-time outputs (if prediction was running)
ls prediction_outputs/A*_Device_*/
```

## Integration with ML Training

The generated data is immediately compatible with your ML training pipeline:

### Training Data Integration
1. **Training files**: Saved to `athlete_training_data/` with TR prefix
2. **Format compatibility**: Matches existing training data format
3. **Sequence management**: Auto-increments TR numbers (TR1, TR2, TR3...)
4. **ML Training**: Run `python sup_ml_rf_training.py` after data generation
5. **Feature engineering**: Enhanced format supports advanced features

### Game Data Analysis
1. **Game files**: Saved to `athlete_game_data/` with GM prefix
2. **Enhanced format**: Additional fields for game analysis
3. **Event tracking**: Significant game events recorded
4. **Performance analysis**: Position-based patterns and fatigue modeling

### Real-time Integration
1. **MQTT Publishing**: Live data streaming during generation
2. **Prediction compatibility**: Same format as live prediction input
3. **Multi-device support**: Handles up to 30 players simultaneously
4. **Organized outputs**: Real-time results saved to `prediction_outputs/`

## Advanced Configuration

### Mode-Specific Customization

#### **Training Mode Customization**:
Edit `_generate_training_data()` method to:
- Adjust training phase durations
- Modify intensity progressions  
- Change warm-up/cool-down patterns
- Customize heart rate ranges

#### **Game Mode Customization**:
Edit `_generate_game_data()` method to:
- Add new game events (corners, free kicks, etc.)
- Modify position-based activity patterns
- Adjust fatigue modeling
- Change event frequency and intensity

### System Configuration
- **Sampling Rate**: Modify `self.data_points_per_second = 10`
- **Game Events**: Edit event types and intensities in `_get_game_event_intensity()`
- **Position Patterns**: Adjust multipliers in `_get_position_multiplier()`
- **Player Profiles**: Modify age, weight, height ranges in `_initialize_player_profiles()`

## Troubleshooting

### Common Issues

#### **"No module named 'paho'"**
```bash
pip install paho-mqtt
```

#### **"Invalid player list format"**
```bash
# Correct formats:
python publisher.py --players [2,9,28,12,5] --mode game
python publisher.py --players 2,9,28,12,5 --mode training

# Incorrect formats:
python publisher.py --players [2 9 28] --mode game  # Missing commas
python publisher.py --players 2-9-28 --mode game    # Wrong separator
```

#### **"Must specify either num_players or --players list"**
```bash
# Must use one of these:
python publisher.py 5 --mode training                    # Random selection
python publisher.py --players [1,2,3] --mode training    # Specific selection

# Not both:
python publisher.py 5 --players [1,2,3] --mode training  # ERROR
```

#### **MQTT connection failed**
```bash
# Check MQTT broker
sudo systemctl status mosquitto

# Start if not running
sudo systemctl start mosquitto

# Install if missing
sudo apt install mosquitto mosquitto-clients
```

#### **Permission denied when saving files**
```bash
# Check permissions
ls -la

# Fix permissions
chmod 755 .
mkdir -p athlete_training_data athlete_game_data prediction_outputs
```

#### **Data not being saved**
- Make sure to press **Ctrl+C** to trigger save
- Check for error messages in the output
- Verify disk space: `df -h`

### Verification

#### **Check Generated Data**:
```bash
# Training data
ls -la athlete_training_data/player_*/TR*.csv

# Game data
ls -la athlete_game_data/player_*/GM*.csv

# Check file content
head -n 5 athlete_training_data/player_1/TR*.csv
```

#### **Verify Data Format**:
```bash
# Training format (16 fields)
head -n 1 athlete_training_data/player_1/TR1*.csv

# Game format (20 fields)  
head -n 1 athlete_game_data/player_1/GM1*.csv
```

## Best Practices

### Training Data Generation
1. **Duration**: Collect data for at least 60 seconds for meaningful datasets
2. **Multiple Sessions**: Generate multiple TR sessions per player (3+ recommended)
3. **Player Variety**: Use different player combinations for diverse datasets
4. **Consistent Profiles**: Players maintain same attributes during session
5. **Verification**: Always verify generated data format before ML training

### Game Data Generation  
1. **Realistic Duration**: Use 45-90 minute games for authentic patterns
2. **Position Diversity**: Include players from different positions
3. **Event Tracking**: Monitor significant game events in output
4. **Fatigue Modeling**: Longer games show realistic fatigue patterns
5. **Team Composition**: Use 11 or 22 players for realistic team scenarios

### System Integration
1. **MQTT Compatibility**: Ensure MQTT broker is running before starting
2. **Prediction Integration**: Data compatible with live prediction system
3. **Storage Management**: Monitor disk space for large datasets
4. **Backup Strategy**: Regular backup of training and game data

## Advanced Usage

### Batch Training Data Generation
Create comprehensive training datasets:
```bash
# Generate multiple training sessions for different player groups
python publisher.py --players [1,2,3,4,5] --mode training      # Session 1
python publisher.py --players [6,7,8,9,10] --mode training     # Session 2  
python publisher.py --players [1,3,5,7,9] --mode training      # Session 3 (mixed)
```

### Game Scenario Simulation
Simulate different game scenarios:
```bash
# Full team game
python publisher.py --players [1,2,3,4,5,6,7,8,9,10,11] --mode game --duration 90

# Half-time simulation
python publisher.py --players [1,3,7,9,11] --mode game --duration 45

# Training match
python publisher.py --players [2,4,6,8,10,12,14] --mode game --duration 30
```

### Integration with Live Prediction
Test complete system workflow:
```bash
# Terminal 1: Start live prediction
python test_30_players.py

# Terminal 2: Generate live data (specific players)
python publisher.py --players [1,5,10] --mode game --duration 20

# Terminal 3: Monitor outputs
watch -n 1 ls -la prediction_outputs/A*_Device_*/
```

### Performance Testing
Test system limits:
```bash
# Test maximum players
python publisher.py 30 --mode training

# Test specific high-activity players
python publisher.py --players [1,5,10,15,20,25,30] --mode game --duration 60

# Test rapid session generation
python publisher.py --players [1,2,3] --mode training  # Run 30s, stop
python publisher.py --players [1,2,3] --mode training  # Run 30s, stop  
python publisher.py --players [1,2,3] --mode training  # Run 30s, stop
```

## Summary

The enhanced multi-mode publisher provides:

✅ **Flexible Player Selection**: Random or specific player lists
✅ **Dual Mode Support**: Training sessions and game simulations  
✅ **Organized Storage**: Separate folders for different data types
✅ **Enhanced Data Format**: Additional fields for comprehensive analysis
✅ **Real-time Integration**: MQTT publishing for live prediction
✅ **Conflict Prevention**: Works with ML training conflict prevention
✅ **Jetson Compatibility**: Optimized for embedded deployment

This complete solution supports both ML model training and real-time athletic performance monitoring! 🎉

