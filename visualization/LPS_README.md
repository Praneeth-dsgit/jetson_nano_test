# Live Position System (LPS) Visualization

A real-time visualization tool for tracking athlete positions on a football field using MQTT data streams.

## Features

- **Real-time Position Tracking**: Live visualization of up to 30 devices/athletes
- **FIFA Standard Field**: Accurate football field dimensions and markings
- **MQTT Integration**: Seamless integration with existing athlete monitoring system
- **G-Impact Events**: Visualization of collision and impact events
- **Configurable**: YAML-based configuration for easy customization
- **Athlete Profiles**: Integration with existing athlete data for better labeling

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Default Configuration

```bash
python lps_visualization.py
```

### 3. Run with Custom Settings

```bash
# Enable debug logging
python lps_visualization.py --debug

# Track specific number of devices
python lps_visualization.py --devices 15

# Set session duration (in minutes)
python lps_visualization.py --duration 90

# Set session mode
python lps_visualization.py --mode training
```

## Configuration

The system uses `config/lps_config.yaml` for configuration. Key settings include:

### MQTT Settings
```yaml
mqtt:
  broker: "localhost"
  port: 1883
  timeout: 60
```

### Device Configuration
```yaml
devices:
  count: 30  # Number of devices to track
  topic_patterns:
    lps_specific: "lps/data/{device_id}"
    legacy: "player/{device_id}/sensor/data"
```

### Field Settings
```yaml
field:
  length: 105.0  # meters
  width: 60.0    # meters
  background_color: "#228B22"  # Forest green
```

## MQTT Data Format

### LPS-Specific Topics
```
Topic: lps/data/{device_id}
Message: {
  "device_id": "001",
  "x": 25.5,
  "y": 30.2,
  "timestamp": 1640995200.0,
  "accuracy": 1.2
}
```

### Legacy Sensor Data Topics
```
Topic: player/{device_id}/sensor/data
Message: {
  "device_id": "001",
  "athlete_id": "A1",
  "x": 25.5,
  "y": 30.2,
  "acc_x": 0.1,
  "acc_y": -0.2,
  "acc_z": 9.8,
  "heart_rate_bpm": 150,
  "mode": "game"
}
```

## G-Impact Events

G-impact events are loaded from JSON log files matching the pattern `*/*_g_impact_log.json`:

```json
[
  {
    "device_id": "001",
    "athlete_id": "A1",
    "name": "Player One",
    "x": 25.5,
    "y": 30.2,
    "g_force": 8.5,
    "timestamp": 1640995200.0,
    "event_type": "collision"
  }
]
```

## Integration with Existing System

The LPS visualization integrates seamlessly with your existing athlete monitoring system:

1. **Same MQTT Broker**: Uses the same MQTT broker as your main system
2. **Compatible Topics**: Subscribes to both LPS-specific and legacy sensor data topics
3. **Athlete Profiles**: Automatically loads athlete information from `data/athlete_training_data/`
4. **Device Mapping**: Maps device IDs to athlete IDs for better labeling

## Testing

### Sample Data Publisher

The LPS visualization automatically starts when you run the main publisher:

```bash
python communication/publisher.py
```

This will:
- Start the data publisher for athlete monitoring
- Automatically launch the LPS visualization with matching parameters
- Begin real-time position tracking with session duration and device count
- Display session information (mode, duration, active devices) on the field

### Manual Testing

1. Start your MQTT broker
2. Run the LPS visualization: `python lps_visualization.py`
3. Publish test data to topics like `lps/data/001` or `player/001/sensor/data`

## Command Line Options

```bash
python lps_visualization.py [OPTIONS]

Options:
  -c, --config PATH     Configuration file path (default: config/lps_config.yaml)
  --debug               Enable debug logging
  -d, --devices COUNT   Number of devices to track
  --duration MINUTES    Session duration in minutes (0 = unlimited)
  --mode MODE           Session mode (training or game)
  -h, --help            Show help message
```

## Troubleshooting

### Common Issues

1. **No positions showing**: Check MQTT connection and topic subscriptions
2. **Connection failed**: Verify MQTT broker is running and accessible
3. **Performance issues**: Reduce update interval or device count in config
4. **Missing G-impact events**: Check file paths and JSON format

### Debug Mode

Enable debug logging to see detailed information:

```bash
python lps_visualization.py --debug
```

### Log Files

Configure log file output in `config/lps_config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "lps_visualization.log"
```

## Performance Optimization

For better performance on resource-constrained systems:

1. **Reduce device count**: Set `devices.count` to match your actual needs
2. **Increase update interval**: Set `visualization.update_interval` to 500ms or higher
3. **Disable blitting**: Set `performance.enable_blit: false`
4. **Limit position history**: Reduce `performance.position_history_size`

## Field Customization

The football field can be customized in the configuration:

```yaml
field:
  length: 105.0              # Field length in meters
  width: 60.0                # Field width in meters
  center_circle_radius: 9.15 # Center circle radius
  penalty_area:
    length: 16.5             # Penalty area length
    width: 40.3              # Penalty area width
  goal_width: 7.32           # Goal width
  visualization:
    background_color: "#228B22"  # Field color
    line_color: "white"          # Line color
    goal_color: "yellow"         # Goal color
```

## License

This tool is part of the athlete monitoring system and follows the same licensing terms.
