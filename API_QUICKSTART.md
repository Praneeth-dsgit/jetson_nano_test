# FastAPI Web API - Quick Start Guide

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   python run_api.py
   ```

3. **Access the web UI:**
   - Open: `http://localhost:8000/`
   - API Docs: `http://localhost:8000/api/docs`

## 📁 Project Structure

```
api/
├── __init__.py
├── main.py                 # FastAPI application
├── routers/               # API route handlers
│   ├── predictions.py     # Prediction endpoints
│   ├── players.py         # Player management
│   ├── training.py        # Training control
│   └── system.py         # System status
├── models/                # Pydantic schemas
│   └── schemas.py
├── services/              # Business logic
│   ├── prediction_service.py
│   └── data_service.py
└── README.md

webui/
└── templates/
    └── index.html        # Web dashboard

run_api.py                 # API server launcher
```

## 🔧 Usage

### Basic API Call Example

```python
import requests

# Get system status
response = requests.get("http://localhost:8000/api/system/status")
print(response.json())

# Get all players
response = requests.get("http://localhost:8000/api/players/list")
print(response.json())

# Get real-time prediction for device 001
response = requests.get("http://localhost:8000/api/predictions/realtime/001")
print(response.json())
```

### WebSocket Example

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8000/api/predictions/ws/001');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Heart Rate:', data.heart_rate);
    console.log('Stress:', data.stress_percent);
};
```

## 📊 Features

- ✅ Real-time prediction data access
- ✅ Player management and monitoring
- ✅ Training control via API
- ✅ System status monitoring
- ✅ Web dashboard UI
- ✅ WebSocket support for live updates
- ✅ RESTful API with OpenAPI documentation

## 🔗 Integration

The API reads from your existing prediction outputs:
- `data/prediction_outputs/A{id}_{name}/A{id}_D{device}_realtime_output.json`

Make sure your prediction engine (`core/main.py`) is running to generate this data.

## 📖 Full Documentation

See `api/README.md` for detailed documentation.

