"""
API documentation and usage guide.
"""

# FastAPI Web API Setup Guide
# ============================

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify FastAPI installation:
   ```bash
   python -c "import fastapi; print(fastapi.__version__)"
   ```

## Running the API

### Basic Usage

```bash
python run_api.py
```

### Advanced Options

```bash
# Custom host and port
python run_api.py --host 0.0.0.0 --port 8080

# Enable auto-reload (development mode)
python run_api.py --reload

# Run on specific interface
python run_api.py --host 192.168.1.100 --port 8000
```

## API Endpoints

### Predictions

- `GET /api/predictions/realtime/{device_id}` - Get latest prediction for a device
- `GET /api/predictions/realtime/all` - Get all active predictions
- `GET /api/predictions/history/{device_id}` - Get historical session summaries
- `WS /api/predictions/ws/{device_id}` - WebSocket for real-time updates
- `WS /api/predictions/ws/all` - WebSocket for all devices

### Players

- `GET /api/players/list` - List all active players
- `GET /api/players/{player_id}` - Get player details
- `GET /api/players/{player_id}/metrics` - Get player health metrics

### Training

- `POST /api/training/start` - Start ML model training
- `GET /api/training/status` - Get training status
- `POST /api/training/stop` - Stop running training

### System

- `GET /api/system/status` - Get system status
- `GET /api/system/health` - Get detailed system health
- `GET /api/system/processes` - Get running processes

## Web UI

Access the web dashboard at:
- Main UI: `http://localhost:8000/`
- API Docs: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Integration with Prediction Engine

The API reads from:
- `data/prediction_outputs/A{athlete_id}_{name}/A{athlete_id}_D{device_id}_realtime_output.json` - Real-time predictions
- `data/prediction_outputs/A{athlete_id}_{name}/*_session_summary_*.json` - Session summaries

Make sure the prediction engine (`core/main.py`) is running to generate these files.

## WebSocket Usage

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/api/predictions/ws/001');

ws.onmessage = (event) => {
    const prediction = JSON.parse(event.data);
    console.log('Heart Rate:', prediction.heart_rate);
    console.log('Stress:', prediction.stress_percent);
};
```

### Python Example

```python
import asyncio
import websockets
import json

async def listen_predictions():
    uri = "ws://localhost:8000/api/predictions/ws/001"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            prediction = json.loads(data)
            print(f"Heart Rate: {prediction['heart_rate']}")

asyncio.run(listen_predictions())
```

## Troubleshooting

### API not connecting to prediction data

1. Verify prediction engine is running
2. Check that `data/prediction_outputs/` directory exists
3. Verify file paths in `api/services/prediction_service.py`

### WebSocket not updating

1. Check browser console for errors
2. Verify WebSocket URL is correct
3. Ensure device_id matches active devices

### Training endpoint not working

1. Verify training script exists: `training/sup_ml_rf_training.py`
2. Check lockfile permissions
3. Review training logs

## Production Deployment

For production deployment:

1. Use a production ASGI server:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. Configure reverse proxy (nginx):
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:8000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

3. Enable HTTPS with SSL certificates

4. Configure CORS properly in `api/main.py`:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

5. Add authentication middleware if needed

