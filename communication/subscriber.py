#!/usr/bin/env python3
"""
MQTT Subscriber for Raw Sensor Data

Subscribes to "rawData/+" topics from MQTT broker and processes sensor data
from PM001 to PM030 devices.

Topic structure:
    rawData/+  (where level1=rawData, level2=+ wildcard)
    Examples: rawData/PM001, rawData/PM002, ..., rawData/PM030
    
Payload format (full sensor data):
    SH004|PM004|acc_x|acc_y|acc_z|gyro_x|gyro_y|gyro_z|magno|pressure|temperature|btyVolt|
    
    Where:
    - SH004: Subhost ID attached to the player module
    - PM004: Player Module ID (PM001 to PM030)
    - Sensor values: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, magno, pressure, temperature, btyVolt
    
Short control message format:
    SH008|PM001:TM003:0

Usage:
    python subscriber.py
    
Examples:
    python subscriber.py
"""

import paho.mqtt.client as mqtt
import json
import logging
import sys
import signal
import time
from datetime import datetime

logger = logging.getLogger(__name__)
from typing import Dict, Optional, Set, Tuple, List
import numpy as np
from dotenv import load_dotenv
import os
import math

# Load environment variables
load_dotenv()


def transform_lps_to_global(
    lps_x: float,
    lps_y: float,
    lps_origin: Tuple[float, float],
    lps_rotation: float = 0.0,
    scale: float = 1.0
) -> Tuple[float, float]:
    """
    Transform local LPS coordinates to global pitch coordinates using affine transformation.
    
    This function maps positions from a local LPS coordinate system (e.g., a small test area)
    to a global coordinate system (e.g., full football pitch).
    
    Transformation steps:
    1. Scale (if needed)
    2. Rotate (if LPS system is rotated relative to pitch)
    3. Translate (move origin to global position)
    
    Args:
        lps_x: X coordinate in local LPS system (meters)
        lps_y: Y coordinate in local LPS system (meters)
        lps_origin: (x, y) position of LPS origin in global coordinates (meters)
        lps_rotation: Rotation angle in degrees (positive = counterclockwise)
        scale: Scale factor (1.0 = no scaling, >1.0 = expand, <1.0 = shrink)
    
    Returns:
        Tuple of (global_x, global_y) in global coordinate system
    
    Example:
        # LPS system at center of pitch, no rotation, 1:1 scale
        global_x, global_y = transform_lps_to_global(
            lps_x=3.5, lps_y=1.75,
            lps_origin=(52.5, 30.0),  # Center of 105×60m pitch
            lps_rotation=0.0,
            scale=1.0
        )
    """
    import math
    
    # Convert rotation to radians
    theta_rad = math.radians(lps_rotation)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    
    # Apply scaling
    scaled_x = lps_x * scale
    scaled_y = lps_y * scale
    
    # Apply rotation
    rotated_x = scaled_x * cos_theta - scaled_y * sin_theta
    rotated_y = scaled_x * sin_theta + scaled_y * cos_theta
    
    # Apply translation
    global_x = rotated_x + lps_origin[0]
    global_y = rotated_y + lps_origin[1]
    
    return global_x, global_y


def get_lps_anchor_positions(field_length: Optional[float] = None, field_width: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Get LPS anchor positions from environment variables or use defaults.
    
    Order matches visualization: A1=top, A2=right, A3=bottom, A4=left.
    Environment variables (comma-separated x,y pairs):
    - LPS_ANCHOR_1: x1,y1 (default: field_length/2, field_width)   # Top mid
    - LPS_ANCHOR_2: x2,y2 (default: field_length, field_width/2)   # Right mid
    - LPS_ANCHOR_3: x3,y3 (default: field_length/2, 0)             # Bottom mid
    - LPS_ANCHOR_4: x4,y4 (default: 0, field_width/2)              # Left mid
    
    Args:
        field_length: Field length in meters (for default calculation)
        field_width: Field width in meters (for default calculation)
    
    Returns:
        List of 4 (x, y) tuples: [A1=top, A2=right, A3=bottom, A4=left]
    """
    if field_length is None:
        field_length = float(os.getenv("LPS_FIELD_LENGTH", "105.0"))
    if field_width is None:
        field_width = float(os.getenv("LPS_FIELD_WIDTH", "60.0"))
    
    anchors = []
    for i in range(1, 5):
        env_var = f"LPS_ANCHOR_{i}"
        anchor_str = os.getenv(env_var)
        
        if anchor_str:
            try:
                parts = anchor_str.split(',')
                if len(parts) == 2:
                    x, y = float(parts[0].strip()), float(parts[1].strip())
                    anchors.append((x, y))
                else:
                    raise ValueError(f"Invalid format for {env_var}: expected 'x,y'")
            except (ValueError, AttributeError) as e:
                print(f"⚠️  Error parsing {env_var}: {e}, using default")
                # Fall through to default
        else:
            # Use defaults: A1=top, A2=right, A3=bottom, A4=left (matches visualization)
            if i == 1:
                anchors.append((field_length / 2, field_width))    # A1: Top midpoint
            elif i == 2:
                anchors.append((field_length, field_width / 2))    # A2: Right midpoint
            elif i == 3:
                anchors.append((field_length / 2, 0.0))            # A3: Bottom midpoint
            elif i == 4:
                anchors.append((0.0, field_width / 2))             # A4: Left midpoint
    
    return anchors


def get_lps_distance_order() -> List[int]:
    """
    Parse LPS_DISTANCE_ORDER env: which payload position (1–4) maps to our A1,A2,A3,A4.
    Default "1,2,3,4" = 1st→A1(top), 2nd→A2(right), 3rd→A3(bottom), 4th→A4(left).
    """
    s = os.getenv("LPS_DISTANCE_ORDER", "1,2,3,4").strip()
    try:
        order = [int(x.strip()) for x in s.split(",") if x.strip()]
        if len(order) == 4 and all(1 <= x <= 4 for x in order) and len(set(order)) == 4:
            return order
    except ValueError:
        pass
    return [1, 2, 3, 4]


def trilaterate_position(
    lps_a1: float,
    lps_a2: float,
    lps_a3: float,
    lps_a4: float,
    anchor_positions: Optional[List[Tuple[float, float]]] = None,
    field_length: Optional[float] = None,
    field_width: Optional[float] = None,
    validate_scale: bool = True,
    last_position: Optional[Tuple[float, float]] = None
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Convert LPS anchor distances to x, y coordinates using least-squares trilateration.
    
    Uses all available anchors (3-4) to minimize sensitivity to measurement noise.
    Implements a least-squares solution that is more robust than solving only 3 equations.
    
    Mathematical Approach:
    ----------------------
    For each anchor i at position (xi, yi) with measured distance di:
        (x - xi)^2 + (y - yi)^2 = di^2
    
    Expanding and rearranging:
        x^2 - 2*xi*x + xi^2 + y^2 - 2*yi*y + yi^2 = di^2
        2*xi*x + 2*yi*y = xi^2 + yi^2 - di^2 + (x^2 + y^2)
    
    Subtracting the first equation from others eliminates the (x^2 + y^2) term,
    giving a linear system: A * [x, y]^T = b
    
    We solve this using least squares: [x, y]^T = (A^T * A)^(-1) * A^T * b
    This minimizes the squared error across all anchor measurements.
    
    Args:
        lps_a1: Distance to A1 (top mid) in meters
        lps_a2: Distance to A2 (right mid) in meters
        lps_a3: Distance to A3 (bottom mid) in meters
        lps_a4: Distance to A4 (left mid) in meters
        anchor_positions: List of (x, y) tuples: [A1=top, A2=right, A3=bottom, A4=left].
                         If None, defaults to midpoints of field sides (legacy mode).
        field_length: Field length in meters (only used if anchor_positions is None)
        field_width: Field width in meters (only used if anchor_positions is None)
        validate_scale: If True, validate that distances are consistent with anchor spacing
        last_position: Optional (x, y) tuple of last known position. Used when exactly 1 anchor
                      is missing - returns last position instead of (0,0).
    
    Returns:
        Tuple of (x, y, confidence) where:
        - x, y: Estimated position coordinates in local LPS coordinate system (meters)
        - confidence: Quality metric (0.0-1.0) indicating reliability:
          * 1.0 = perfect fit (all distances match exactly)
          * >0.7 = good estimate (low residual error)
          * 0.5-0.7 = acceptable (moderate error)
          * <0.5 = poor estimate (high error, may be unreliable)
        Returns (None, None, 0.0) if:
          - Fewer than 3 valid anchors available
          - Scale mismatch detected (if validate_scale=True)
          - Matrix is singular
    
    Scale Mismatch Detection:
    ------------------------
    Detects when measured distances are inconsistent with anchor spacing:
    - Calculates maximum possible distance between anchors
    - Compares measured distances to expected range
    - Raises warning if distances suggest wrong coordinate system
    
    Example Usage:
    -------------
    # Local LPS coordinate system (7m × 3.5m): A1=top, A2=right, A3=bottom, A4=left
    anchors = [
        (3.5, 3.5),   # A1: Top midpoint
        (7.0, 1.75),  # A2: Right midpoint
        (3.5, 0.0),   # A3: Bottom midpoint
        (0.0, 1.75)   # A4: Left midpoint
    ]
    # lps_a1=dist to top, lps_a2=right, lps_a3=bottom, lps_a4=left
    x, y, conf = trilaterate_position(5.66, 1.57, 4.32, 4.37, anchor_positions=anchors)
    """
    # Determine anchor positions (A1=top, A2=right, A3=bottom, A4=left; matches visualization)
    if anchor_positions is None:
        # Legacy mode: derive from field dimensions
        if field_length is None:
            field_length = float(os.getenv("LPS_FIELD_LENGTH", "105.0"))
        if field_width is None:
            field_width = float(os.getenv("LPS_FIELD_WIDTH", "60.0"))
        
        anchors = np.array([
            [field_length / 2, field_width],     # A1: Top midpoint
            [field_length, field_width / 2],     # A2: Right midpoint
            [field_length / 2, 0],               # A3: Bottom midpoint
            [0, field_width / 2]                 # A4: Left midpoint
        ])
    else:
        # Use explicit anchor positions
        if len(anchor_positions) != 4:
            raise ValueError(f"anchor_positions must contain exactly 4 positions, got {len(anchor_positions)}")
        anchors = np.array(anchor_positions)
    
    distances = np.array([lps_a1, lps_a2, lps_a3, lps_a4])
    
    # Check how many anchor distances are valid (> 0)
    valid_mask = distances > 0
    num_valid = np.sum(valid_mask)
    num_invalid = 4 - num_valid
    
    # Handle missing anchor data based on count:
    # - 0 invalid (all 4 valid): proceed with full trilateration
    # - 1 invalid (3 valid): use last known position if available, else try with 3 anchors
    # - 2+ invalid (<=2 valid): return origin (0, 0)
    
    if num_invalid >= 2:
        # Two or more anchors missing - cannot reliably compute position
        invalid_anchors = [f"A{i+1}" for i, v in enumerate(valid_mask) if not v]
        print(f"⚠️  Multiple anchors missing: {', '.join(invalid_anchors)} = 0. Returning origin (0, 0).")
        return 0.0, 0.0, 0.0  # Return origin to indicate invalid data
    
    if num_invalid == 1:
        # Exactly one anchor missing - use last known position if available
        invalid_anchors = [f"A{i+1}" for i, v in enumerate(valid_mask) if not v]
        if last_position is not None and last_position[0] > 0 and last_position[1] > 0:
            print(f"⚠️  Missing anchor data: {', '.join(invalid_anchors)} = 0. Using last position ({last_position[0]:.2f}, {last_position[1]:.2f}).")
            return last_position[0], last_position[1], 0.5  # Return last position with reduced confidence
        else:
            # No last position available, try trilateration with 3 anchors (less accurate)
            print(f"⚠️  Missing anchor data: {', '.join(invalid_anchors)} = 0. Attempting trilateration with 3 anchors.")
    
    valid_anchors = anchors[valid_mask]
    valid_distances = distances[valid_mask]
    
    # Scale mismatch detection: validate that distances are consistent with anchor spacing
    if validate_scale and len(valid_anchors) >= 3:
        # Calculate maximum possible distance between any two anchors
        max_anchor_dist = 0.0
        for i in range(len(valid_anchors)):
            for j in range(i + 1, len(valid_anchors)):
                dist = np.linalg.norm(valid_anchors[i] - valid_anchors[j])
                max_anchor_dist = max(max_anchor_dist, dist)
        
        # Maximum possible distance from any point to an anchor (diagonal of bounding box)
        anchor_bounds = np.array([
            [np.min(valid_anchors[:, 0]), np.min(valid_anchors[:, 1])],
            [np.max(valid_anchors[:, 0]), np.max(valid_anchors[:, 1])]
        ])
        max_possible_distance = np.linalg.norm(anchor_bounds[1] - anchor_bounds[0])
        
        # Check if measured distances are reasonable
        max_measured = np.max(valid_distances)
        min_measured = np.min(valid_distances[valid_distances > 0])
        
        # Warning thresholds:
        # - If max measured distance > 2x max possible, likely scale mismatch
        # - If distances are much larger than anchor spacing, likely wrong coordinate system
        scale_ratio = max_measured / max_anchor_dist if max_anchor_dist > 0 else float('inf')
        
        if scale_ratio > 2.0:
            print(f"⚠️  SCALE MISMATCH DETECTED: Max measured distance ({max_measured:.2f}m) is "
                  f"{scale_ratio:.1f}x larger than max anchor spacing ({max_anchor_dist:.2f}m). "
                  f"This suggests wrong coordinate system or unit mismatch.")
            # Don't fail, but reduce confidence
            scale_warning = True
        elif scale_ratio > 1.5:
            print(f"⚠️  Possible scale issue: Max measured distance ({max_measured:.2f}m) is "
                  f"{scale_ratio:.1f}x larger than max anchor spacing ({max_anchor_dist:.2f}m).")
            scale_warning = False
        else:
            scale_warning = False
    else:
        scale_warning = False
        max_possible_distance = None
    
    # Get field dimensions for bounded optimization
    fl = field_length if field_length is not None else float(os.getenv("LPS_FIELD_LENGTH", "105.0"))
    fw = field_width if field_width is not None else float(os.getenv("LPS_FIELD_WIDTH", "60.0"))
    
    # DEBUG: Show valid anchors and distances (only when LPS_DEBUG_TRILAT=1 and log level is DEBUG)
    debug_trilat = os.getenv("LPS_DEBUG_TRILAT", "0") == "1"
    if debug_trilat:
        logger.debug("TRILAT: valid_anchors=%s, valid_distances=%s", valid_anchors.tolist(), valid_distances.tolist())
    
    # Use CONSTRAINED optimization - only search within field bounds
    # This ensures the position is always inside the field (semi-circles intersecting within bounds)
    from scipy.optimize import minimize
    
    def distance_error(pos):
        """Sum of squared errors between calculated and measured distances"""
        x, y = pos
        errors = 0.0
        for anchor, dist in zip(valid_anchors, valid_distances):
            calc_dist = np.sqrt((x - anchor[0])**2 + (y - anchor[1])**2)
            errors += (calc_dist - dist)**2
        return errors
    
    # Start from field center
    initial_guess = [fl / 2, fw / 2]
    
    # Constrain to field bounds (0 to field_length, 0 to field_width)
    bounds = [(0, fl), (0, fw)]
    
    # Optimize to find best position within bounds
    result = minimize(distance_error, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        x, y = result.x[0], result.x[1]
        if debug_trilat:
            logger.debug("TRILAT (bounded): solution=(%.3f, %.3f), error=%.3f", x, y, result.fun)
    else:
        # Fallback to unconstrained least-squares
        n = len(valid_anchors)
        p_ref = valid_anchors[0]
        d_ref = valid_distances[0]
        x_ref, y_ref = p_ref
        
        A = []
        b = []
        for i in range(1, n):
            xi, yi = valid_anchors[i]
            di = valid_distances[i]
            A.append([2 * (xi - x_ref), 2 * (yi - y_ref)])
            b.append((x_ref**2 + y_ref**2 - d_ref**2) - (xi**2 + yi**2 - di**2))
        
        A = np.array(A)
        b = np.array(b)
        
        try:
            position = np.linalg.lstsq(A, b, rcond=None)[0]
            x, y = position[0], position[1]
            if debug_trilat:
                logger.debug("TRILAT (fallback lstsq): solution=(%.3f, %.3f)", x, y)
        except np.linalg.LinAlgError:
            return None, None, 0.0
    
    # Calculate confidence metric based on residual error
    # Compute how well the solution matches the measured distances
    calculated_distances = np.sqrt(np.sum((valid_anchors - np.array([x, y]))**2, axis=1))
    residuals = np.abs(calculated_distances - valid_distances)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Normalize RMSE by average distance to get relative error
    avg_distance = np.mean(valid_distances)
    if avg_distance > 0:
        relative_error = rmse / avg_distance
        # Convert to confidence: lower error = higher confidence
        # Map: 0% error -> 1.0, 10% error -> 0.7, 20% error -> 0.4, 30%+ error -> 0.1
        confidence = max(0.0, min(1.0, 1.0 - (relative_error * 5)))
    else:
        confidence = 0.5  # Default if distances are very small
    
    # Reduce confidence if scale mismatch was detected
    if 'scale_warning' in locals() and scale_warning:
        confidence = min(confidence, 0.3)  # Cap confidence at 0.3 for scale mismatches
    
    # Check if position is within field bounds (should always be true with bounded optimization)
    in_bounds = (0 <= x <= fl) and (0 <= y <= fw)
    
    if not in_bounds:
        # Position is outside field - this shouldn't happen with bounded optimization
        # but keep as safety check
        confidence = 0.0
    
    # Return position (bounded optimization ensures it's within field)
    return float(x), float(y), float(confidence)


class RawDataSubscriber:
    """MQTT subscriber for raw sensor data from multiple devices."""
    
    def __init__(self, broker: str = "192.168.5.88", port: int = 1883, 
                 topic: str = "rawData/+", client_id: Optional[str] = None):
        """
        Initialize the MQTT subscriber.
        
        Args:
            broker: MQTT broker hostname or IP address
            port: MQTT broker port
            topic: Topic pattern to subscribe to (default: "rawData/+")
            client_id: Optional MQTT client ID (auto-generated if None)
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.running = True
        
        # Track active devices (1-30)
        self.active_devices: Set[str] = set()
        self.device_stats: Dict[str, Dict] = {}
        
        # Store last known positions for each device (used when 1 anchor is missing)
        self.last_positions: Dict[str, Tuple[float, float]] = {}
        
        # Initialize MQTT client
        if client_id is None:
            client_id = f"rawData_subscriber_{int(time.time())}"
        
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_subscribe = self._on_subscribe
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            print(f"✅ Connected to MQTT broker: {self.broker}:{self.port}")
            print(f"📡 Subscribing to topic: {self.topic}")
            client.subscribe(self.topic, qos=1)
        else:
            print(f"❌ Failed to connect to MQTT broker with result code {rc}")
            sys.exit(1)
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback."""
        if rc != 0:
            print(f"⚠️  Unexpected disconnection from MQTT broker (code: {rc})")
            print("🔄 Attempting to reconnect...")
        else:
            print("ℹ️  Disconnected from MQTT broker")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """MQTT subscription callback."""
        print(f"✅ Successfully subscribed to topic: {self.topic}")
        print(f"📊 Waiting for messages from devices 1-30...")
        print(f"⏹️  Press Ctrl+C to stop\n")
    
    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT messages."""
        try:
            # Decode message payload
            topic = msg.topic
            
            # Check if payload is empty
            if not msg.payload:
                print(f"⚠️  Empty payload received from {topic}")
                return
            
            try:
                payload_str = msg.payload.decode('utf-8')
            except UnicodeDecodeError as e:
                print(f"❌ Failed to decode payload from {topic}: {e}")
                print(f"   Payload (hex): {msg.payload.hex()[:100]}...")  # Show first 100 chars
                return
            
            # Check if payload is empty string
            if not payload_str or payload_str.strip() == '':
                print(f"⚠️  Empty string payload received from {topic}")
                return
            
            # Parse payload - try JSON first, then pipe-delimited format
            data = None
            try:
                data = json.loads(payload_str)
            except json.JSONDecodeError:
                # Try parsing as pipe-delimited format
                try:
                    data = self._parse_pipe_delimited(payload_str, topic)
                    if data is None:
                        print(f"⚠️  Could not parse payload from {topic}")
                        print(f"   Payload preview (first 200 chars): {payload_str[:200]}")
                        return
                except Exception as e:
                    print(f"❌ Failed to parse payload from {topic}: {e}")
                    print(f"   Payload preview (first 200 chars): {payload_str[:200]}")
                    print(f"   Payload length: {len(payload_str)} bytes")
                    import traceback
                    traceback.print_exc()
                    return
            
            # Extract device ID from topic (rawData/PM004) — each topic = one device; PM001 and PM020 are separate sources
            device_id = self._extract_device_id(topic)
            if not device_id:
                print(f"⚠️  Could not extract device ID from topic: {topic}")
                return
            
            # Validate device ID is in range 1-30
            if not self._is_valid_device_id(device_id):
                print(f"⚠️  Device ID out of range (1-30): {device_id}")
                return
            
            # Track active device
            self.active_devices.add(device_id)
            
            # Initialize device stats if needed
            if device_id not in self.device_stats:
                self.device_stats[device_id] = {
                    'message_count': 0,
                    'sensor_data_count': 0,
                    'control_message_count': 0,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'errors': 0
                }
            
            # Update device stats
            stats = self.device_stats[device_id]
            stats['message_count'] += 1
            stats['last_seen'] = datetime.now()
            
            # Track message type
            message_type = data.get('message_type', 'unknown')
            if message_type == 'sensor_data':
                stats['sensor_data_count'] += 1
            elif message_type == 'control':
                stats['control_message_count'] += 1
            
            # Debug: Print parsed data for first few messages to diagnose issues
            if stats['message_count'] <= 2:
                print(f"🔍 Debug - Device {device_id} ({message_type}) parsed fields: {sorted(data.keys())}")
                if message_type == 'sensor_data':
                    print(f"   Sample values - acc_x: {data.get('acc_x')}, pressure: {data.get('pressure')}, magno: {data.get('magno')}")
                elif message_type == 'control':
                    print(f"   Control value: {data.get('control_value')}")
            
            # Process the data
            self._process_data(device_id, data, topic, message_type)
            
        except Exception as e:
            print(f"❌ Error processing message from {msg.topic}: {e}")
            if device_id in self.device_stats:
                self.device_stats[device_id]['errors'] += 1
    
    def _parse_pipe_delimited(self, payload_str: str, topic: str) -> Optional[Dict]:
        """
        Parse pipe-delimited sensor data format.
        
        Handles two payload types:
        1. Full sensor data: SH004|PM004|1.12|-0.56|-9.25|0.00|0.00|0.00|75.9E|  917.40|  53.1| 0.00|
        2. Short control message: SH008|PM001:TM003:0 or SH004|PM004:TM001:0
        
        Returns:
            Dictionary with parsed sensor data, or None if parsing fails
        """
        try:
            # Remove trailing pipes and whitespace, but keep all parts (including empty ones for valid fields)
            payload_str = payload_str.strip().rstrip('|')
            parts = [p.strip() for p in payload_str.split('|')]  # Strip whitespace but keep all parts
            
            # Skip if too few parts
            if len(parts) < 2:
                print(f"⚠️  Too few parts in payload: {len(parts)}")
                print(f"   Payload: {payload_str[:100]}")
                return None
            
            # Extract device ID from second part (PM004 or PM004:TM001)
            device_part = parts[1].strip()
            # Handle both "PM004" and "PM004:TM001" formats
            if device_part.startswith('PM'):
                device_id_str = device_part[2:].split(':')[0]  # Get "004" from "PM004:TM001"
            else:
                device_id_str = device_part.split(':')[0]  # Get "004" from "004:TM001"
            
            # Convert to integer to normalize (remove leading zeros, e.g., "004" -> "4")
            try:
                device_id = str(int(device_id_str))
            except ValueError:
                device_id = device_id_str
            
            # Detect payload type: short control message (has colon in PM_ID part and only 2-3 parts)
            # Format: SH008|PM001:TM003:0
            is_short_message = ':' in device_part and len(parts) <= 3
            
            if is_short_message:
                # Short control/status message
                data = {
                    'device_id': device_id,
                    'subhost_id': parts[0].strip() if len(parts) > 0 else '',
                    'pm_id': parts[1].strip() if len(parts) > 1 else '',
                    'message_type': 'control',
                    'control_value': self._safe_float(parts[2]) if len(parts) > 2 else 0.0
                }
                return data
            
            # Full sensor data format:
            # Index 0: Subhost_ID
            # Index 1: PM_ID
            # Index 2: acc_x
            # Index 3: acc_y
            # Index 4: acc_z
            # Index 5: gyro_x
            # Index 6: gyro_y
            # Index 7: gyro_z
            # Index 8: magno
            # Index 9: pressure
            # Index 10: temperature
            # Index 11: btyVolt
            try:
                data = {
                    'device_id': device_id,  # Normalized (e.g., "4" not "004")
                    'subhost_id': parts[0].strip() if len(parts) > 0 else '',
                    'pm_id': parts[1].strip() if len(parts) > 1 else '',
                    'message_type': 'sensor_data',
                    'acc_x': self._safe_float(parts[2]) if len(parts) > 2 else 0.0,
                    'acc_y': self._safe_float(parts[3]) if len(parts) > 3 else 0.0,
                    'acc_z': self._safe_float(parts[4]) if len(parts) > 4 else 0.0,
                    'gyro_x': self._safe_float(parts[5]) if len(parts) > 5 else 0.0,
                    'gyro_y': self._safe_float(parts[6]) if len(parts) > 6 else 0.0,
                    'gyro_z': self._safe_float(parts[7]) if len(parts) > 7 else 0.0,
                }
                
                # Parse magno (index 8) - might be in format like "330.8NW" or "75.9E"
                if len(parts) > 8 and parts[8].strip():
                    magno_str = parts[8].strip()
                    # Extract numeric value from magno (remove direction letters like NW, SE, S, E, etc.)
                    magno_value = ''.join(c for c in magno_str if c.isdigit() or c == '.' or c == '-')
                    data['magno'] = float(magno_value) if magno_value else 0.0
                else:
                    data['magno'] = 0.0
                
                # Parse pressure (index 9)
                data['pressure'] = self._safe_float(parts[9]) if len(parts) > 9 else 0.0
                
                # Parse temperature (index 10)
                data['temperature'] = self._safe_float(parts[10]) if len(parts) > 10 else 0.0
                
                # Parse btyVolt (index 11)
                data['btyVolt'] = self._safe_float(parts[11]) if len(parts) > 11 else 0.0
                    
                # LPS fields: lps_a1=to A1(top), lps_a2=to A2(right), lps_a3=to A3(bottom), lps_a4=to A4(left)
                for i, field in enumerate(['lps_a1', 'lps_a2', 'lps_a3', 'lps_a4'], start=12):
                    data[field] = self._safe_float(parts[i]) if len(parts) > i else 0.0
                
                # Check for ECG - if present after LPS fields
                if len(parts) > 16 and parts[16].strip():
                    ecg_str = parts[16].strip()
                    if ecg_str:
                        data['ECG'] = self._safe_float(ecg_str)
                
                return data
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  Error parsing pipe-delimited data: {e}")
                print(f"   Parts count: {len(parts)}, Parts: {parts[:15]}")  # Show first 15 parts
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"⚠️  Exception in pipe-delimited parser: {e}")
            return None
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float, handling empty strings and whitespace."""
        if not value or not value.strip():
            return 0.0
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    
    def _extract_device_id(self, topic: str) -> Optional[str]:
        """
        Extract device ID from topic, preserving PM format.
        
        Topic structure: rawData/PM004
        - level1: rawData
        - level2: PM004 (or PM001 to PM030)
        
        Also handles formats like:
        - rawData/PM004:TM001 (with training module suffix)
        
        Returns: Device ID in PM format (e.g., "PM004", "PM021")
        """
        try:
            # Split topic by '/'
            parts = topic.split('/')
            
            # Handle 2-level structure: rawData/PM004
            if len(parts) >= 2:
                device_part = parts[1]  # Get PM004 from rawData/PM004
            else:
                return None
            
            # Handle formats like "PM001" or "PM004:TM001" or just "001" or "1"
            if device_part.startswith('PM'):
                # Keep PM format: PM004 -> PM004, PM004:TM001 -> PM004
                device_id = device_part.split(':')[0]  # Get part before colon if present
                return device_id  # Return "PM004" format
            else:
                # Handle formats like "004:TM001" or just "004" - add PM prefix
                device_id = device_part.split(':')[0]  # Get part before colon if present
                # Normalize numeric part and add PM prefix
                try:
                    device_num = int(device_id)
                    return f"PM{device_num:03d}"  # Return "PM004" format
                except ValueError:
                    # If not numeric, try to pad and add PM prefix
                    return f"PM{device_id.zfill(3)}"
        except Exception as e:
            print(f"⚠️  Error extracting device ID from topic {topic}: {e}")
            return None
    
    def _is_valid_device_id(self, device_id: str) -> bool:
        """Check if device ID is in valid range (1-30), handles PM format."""
        try:
            # Extract numeric part from PM format (e.g., "PM004" -> 4)
            if device_id.startswith('PM'):
                device_num = int(device_id[2:])
            else:
                device_num = int(device_id)
            return 1 <= device_num <= 30
        except (ValueError, TypeError):
            return False
    
    def _process_data(self, device_id: str, data: Dict, topic: str, message_type: str):
        """Process incoming data based on message type."""
        # Skip control messages as per requirement
        if message_type == 'control':
            if self.device_stats[device_id]['message_count'] % 100 == 0:
                print(f"📊 Device {device_id} (Control): "
                      f"PM_ID={data.get('pm_id', 'N/A')}, value={data.get('control_value', 'N/A')}, "
                      f"Messages: {self.device_stats[device_id]['message_count']}")
            return  # Skip control messages
        
        if message_type == 'sensor_data':
            # Process full sensor data by calling main.py processing functions
            try:
                # Import main.py processing function
                import sys
                import os
                # Add parent directory to path to import main.py
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                from core.main import process_raw_sensor_data
                
                # Calculate x, y position from LPS anchor data if available
                # Apply LPS_DISTANCE_ORDER: payload positions 1–4 may map to A1..A4 in a different order
                raw = [data.get('lps_a1', 0.0), data.get('lps_a2', 0.0), data.get('lps_a3', 0.0), data.get('lps_a4', 0.0)]
                order = get_lps_distance_order()
                lps_a1 = raw[order[0] - 1]
                lps_a2 = raw[order[1] - 1]
                lps_a3 = raw[order[2] - 1]
                lps_a4 = raw[order[3] - 1]
                
                # Log raw distances every 5th message for PM001 to verify they change when tag moves
                msg_count = self.device_stats.get(device_id, {}).get('message_count', 0)
                if device_id == "PM001" and msg_count % 5 == 0:
                    print(f"📡 PM001 RAW distances: [{raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}, {raw[3]:.2f}] "
                          f"→ order {order} → A1={lps_a1:.2f}, A2={lps_a2:.2f}, A3={lps_a3:.2f}, A4={lps_a4:.2f}")
                
                x, y, confidence = None, None, 0.0
                if lps_a1 > 0 or lps_a2 > 0 or lps_a3 > 0 or lps_a4 > 0:
                    # Get anchor positions from environment or use defaults
                    field_length = float(os.getenv("LPS_FIELD_LENGTH", "105.0"))
                    field_width = float(os.getenv("LPS_FIELD_WIDTH", "60.0"))
                    anchor_positions = get_lps_anchor_positions(field_length, field_width)
                    
                    # Debug: Log anchor positions for first few messages
                    msg_count = self.device_stats.get(device_id, {}).get('message_count', 0)
                    if msg_count < 3:
                        # Check if anchors are from env vars (explicit) or derived from field dimensions
                        explicit_anchors = any(os.getenv(f"LPS_ANCHOR_{i}") for i in range(1, 5))
                        anchor_source = "explicit (from env vars)" if explicit_anchors else f"derived from field dimensions ({field_length}m×{field_width}m)"
                        print(f"🔧 Device {device_id} - Using anchor positions: {anchor_positions}")
                        print(f"   Source: {anchor_source}")
                        print(f"   Distance order (LPS_DISTANCE_ORDER): {get_lps_distance_order()} → A1,A2,A3,A4")
                        print(f"   ⚠️  NOTE: If distances are small (2-5m) but anchors are far apart (>50m), set explicit anchors via LPS_ANCHOR_1-4. If tags are inside the field but pos is (0,0), try LPS_DISTANCE_ORDER=3,2,1,4.")
                    
                    # Debug: print inputs BEFORE calling trilaterate
                    if device_id == "PM001":
                        print(f"🔍 PM001 TRILATERATE INPUT: d1={lps_a1:.3f}, d2={lps_a2:.3f}, d3={lps_a3:.3f}, d4={lps_a4:.3f}")
                    
                    # Get last known position for this device (used when 1 anchor is missing)
                    last_pos = self.last_positions.get(device_id)
                    
                    x, y, confidence = trilaterate_position(
                        lps_a1, lps_a2, lps_a3, lps_a4,
                        anchor_positions=anchor_positions,
                        field_length=field_length,
                        field_width=field_width,
                        validate_scale=True,
                        last_position=last_pos
                    )
                    
                    # Store valid position as last known position (skip origin)
                    if x is not None and y is not None and (x > 0.1 or y > 0.1):
                        self.last_positions[device_id] = (x, y)
                    
                    # Debug: print output AFTER calling trilaterate
                    if device_id == "PM001":
                        print(f"🔍 PM001 TRILATERATE OUTPUT: x={x}, y={y}, conf={confidence}")
                    
                    # Always log for PM001 to debug the issue
                    if device_id == "PM001":
                        if x is not None and y is not None:
                            conf_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
                            print(f"📍 PM001 - LPS: anchors=({lps_a1:.2f}, {lps_a2:.2f}, {lps_a3:.2f}, {lps_a4:.2f}), "
                                  f"field={field_length}m×{field_width}m, pos=({x:.2f}, {y:.2f}), confidence={confidence:.2f} ({conf_label})")
                            if abs(x) < 0.6 and abs(y) < 0.6:
                                print(f"   ⚠️  WARNING: Position near (0,0) – raw solution was outside the field (see above).")
                                _lps_hint_scale = field_length > 20 or field_width > 15
                                if _lps_hint_scale:
                                    print(f"   💡 If this is a small test area: set --lps-length and --lps-width, or LPS_ANCHOR_1-4 for A1=top, A2=right, A3=bottom, A4=left.")
                                else:
                                    print(f"   💡 If tags are inside the field: hardware may use different order. Try LPS_DISTANCE_ORDER=3,2,1,4 (1st→A3, 2nd→A2, 3rd→A1, 4th→A4). See LPS_ANCHOR_SETUP.md.")
                        else:
                            print(f"❌ PM001 - Trilateration failed: anchors=({lps_a1:.2f}, {lps_a2:.2f}, {lps_a3:.2f}, {lps_a4:.2f})")
                            print(f"   💡 Check: ≥3 of lps_a1..a4 > 0; order d1=A1(top), d2=A2(right), d3=A3(bottom), d4=A4(left).")
                            if field_length > 20 or field_width > 15:
                                print(f"   💡 If using a small test area, set LPS_ANCHOR_1-4 or --lps-length/--lps-width.")
                    elif msg_count < 5:
                        # Log first few calculations for other devices
                        if x is not None and y is not None:
                            conf_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
                            print(f"📍 Device {device_id} - LPS: anchors=({lps_a1:.2f}, {lps_a2:.2f}, {lps_a3:.2f}, {lps_a4:.2f}), "
                                  f"field={field_length}m×{field_width}m, pos=({x:.2f}, {y:.2f}), confidence={confidence:.2f} ({conf_label})")
                        else:
                            print(f"❌ Device {device_id} - Trilateration failed: anchors=({lps_a1:.2f}, {lps_a2:.2f}, {lps_a3:.2f}, {lps_a4:.2f})")
                
                # Format data for main.py
                formatted_data = {
                    'device_id': device_id,
                    'athlete_id': device_id,  # Use device_id as athlete_id
                    'subhost_id': data.get('subhost_id', ''),
                    'pm_id': data.get('pm_id', ''),
                    'acc_x': data.get('acc_x', 0.0),
                    'acc_y': data.get('acc_y', 0.0),
                    'acc_z': data.get('acc_z', 0.0),
                    'gyro_x': data.get('gyro_x', 0.0),
                    'gyro_y': data.get('gyro_y', 0.0),
                    'gyro_z': data.get('gyro_z', 0.0),
                    'magno': data.get('magno', 0.0),  # Single value
                    'pressure': data.get('pressure', 0.0),
                    'temperature': data.get('temperature', 0.0),
                    'btyVolt': data.get('btyVolt', 0.0),
                    'mode': 'game',  # Default to game mode for sensor data
                }
                
                # Only include x, y if they are valid (not None)
                if x is not None and y is not None:
                    formatted_data['x'] = x
                    formatted_data['y'] = y
                # If x or y is None, don't include them (let latest_position keep previous value)
                
                # Call main.py processing function
                process_raw_sensor_data(formatted_data, device_id)
                
                # Log periodically
                if self.device_stats[device_id]['message_count'] % 100 == 0:
                    print(f"📊 Device {device_id} (Sensor Data): "
                          f"acc=({data.get('acc_x', 'N/A')}, {data.get('acc_y', 'N/A')}, {data.get('acc_z', 'N/A')}), "
                          f"pressure={data.get('pressure', 'N/A')}, temp={data.get('temperature', 'N/A')}, "
                          f"Messages: {self.device_stats[device_id]['message_count']}")
            except ImportError as e:
                print(f"⚠️  Could not import main.py processing functions: {e}")
                print(f"   Make sure core/main.py is accessible")
            except Exception as e:
                print(f"❌ Error processing sensor data in main.py: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Device {device_id}: Unknown message type: {message_type}")
    
    def _print_statistics(self):
        """Print statistics for all active devices."""
        if not self.active_devices:
            print("\n📊 No devices have sent data yet.")
            return
        
        print("\n" + "="*80)
        print("📊 DEVICE STATISTICS")
        print("="*80)
        
        # Helper function to extract numeric value from device ID for sorting (handles PM format)
        def get_device_sort_key(device_id: str) -> int:
            """Extract numeric value from device ID for sorting (handles PM format)."""
            try:
                if device_id.startswith('PM'):
                    return int(device_id[2:])  # Extract "001" from "PM001"
                else:
                    return int(device_id)
            except (ValueError, TypeError):
                return 0
        
        for device_id in sorted(self.active_devices, key=get_device_sort_key):
            stats = self.device_stats[device_id]
            duration = (stats['last_seen'] - stats['first_seen']).total_seconds()
            
            print(f"\n🔹 Device {device_id}:")
            print(f"   Total Messages: {stats['message_count']}")
            print(f"   Sensor Data: {stats['sensor_data_count']}")
            print(f"   Control Messages: {stats['control_message_count']}")
            print(f"   Errors: {stats['errors']}")
            print(f"   First Seen: {stats['first_seen'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Last Seen: {stats['last_seen'].strftime('%Y-%m-%d %H:%M:%S')}")
            if duration > 0:
                print(f"   Duration: {duration:.1f}s")
                print(f"   Message Rate: {stats['message_count']/duration:.2f} msg/s")
        
        print("\n" + "="*80)
        print(f"📈 Total Active Devices: {len(self.active_devices)}")
        print(f"📈 Total Messages Received: {sum(s['message_count'] for s in self.device_stats.values())}")
        print("="*80)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n🛑 Stopping subscriber...")
        self.running = False
        
        # Stop MQTT loop first to prevent new messages
        try:
            self.client.loop_stop()
        except:
            pass
        
        # Stop message queue processing thread
        try:
            from communication.mqtt_message_queue import message_queue
            if hasattr(message_queue, 'stop_processing'):
                message_queue.stop_processing()
        except:
            pass
        
        # Print statistics (with error handling to prevent shutdown issues)
        try:
            self._print_statistics()
        except Exception as e:
            print(f"⚠️  Error printing statistics: {e}")
        
        # Disconnect from MQTT
        try:
            print("\n🔄 Disconnecting from MQTT broker...")
            self.client.disconnect()
        except:
            pass
        
        # Give threads a moment to finish
        import time
        time.sleep(0.5)
        
        print("✅ Subscriber stopped successfully")
        sys.exit(0)
    
    def connect(self):
        """Connect to MQTT broker with retry logic."""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"🔄 Connecting to MQTT broker: {self.broker}:{self.port} (attempt {retry_count + 1}/{max_retries})...")
                self.client.connect(self.broker, self.port, 60)
                return
            except Exception as e:
                retry_count += 1
                print(f"❌ Connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    print(f"🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("❌ Failed to connect to MQTT broker after 5 attempts")
                    raise ConnectionError("Could not connect to MQTT broker")
    
    def run(self):
        """Start the subscriber and run the MQTT loop."""
        try:
            # Connect to broker
            self.connect()
            
            # Start MQTT network loop
            self.client.loop_start()
            
            # Wait for connection to establish
            time.sleep(2)
            
            # Keep running until interrupted
            print("✅ Subscriber is running. Waiting for messages...\n")
            
            # Import here to check publish client status
            try:
                from core.main import client as publish_client, MQTT_PUBLISH_BROKER, MQTT_PUBLISH_PORT
                from communication.mqtt_message_queue import message_queue
                publish_client_available = True
            except ImportError:
                publish_client_available = False
                publish_client = None
            
            last_status_check = 0
            status_check_interval = 30  # Check every 30 seconds
            
            while self.running:
                time.sleep(1)
                current_time = time.time()
                
                # Periodically print statistics and publish client status
                if current_time - last_status_check >= status_check_interval:
                    last_status_check = current_time
                    
                    # Print device statistics
                    if self.active_devices:
                        print(f"\n📊 Active devices: {len(self.active_devices)} | "
                              f"Total messages: {sum(s['message_count'] for s in self.device_stats.values())}")
                    
                    # Print publish client status
                    if publish_client_available and publish_client:
                        is_connected = publish_client.is_connected() if hasattr(publish_client, 'is_connected') else False
                        if is_connected:
                            try:
                                queue_stats = message_queue.get_queue_stats()
                                pending = queue_stats.get('status_counts', {}).get('pending', 0)
                                sent = queue_stats.get('status_counts', {}).get('sent', 0)
                                print(f"📤 Publish Client: ✅ Connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} | "
                                      f"Queue: {pending} pending, {sent} sent")
                            except:
                                print(f"📤 Publish Client: ✅ Connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}")
                        else:
                            try:
                                queue_stats = message_queue.get_queue_stats()
                                pending = queue_stats.get('status_counts', {}).get('pending', 0)
                                print(f"📤 Publish Client: ⚠️  NOT connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT} | "
                                      f"{pending} messages queued (waiting for connection)")
                            except:
                                print(f"📤 Publish Client: ⚠️  NOT connected to {MQTT_PUBLISH_BROKER}:{MQTT_PUBLISH_PORT}")
        
        except KeyboardInterrupt:
            self._signal_handler(None, None)
        except Exception as e:
            print(f"❌ Error in subscriber: {e}")
            self._signal_handler(None, None)


def main():
    """Main function using environment variables only."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MQTT Subscriber for Raw Sensor Data')
    parser.add_argument('--lps-length', type=float, default=None,
                       help='LPS field length in meters (overrides LPS_FIELD_LENGTH env var)')
    parser.add_argument('--lps-width', type=float, default=None,
                       help='LPS field width in meters (overrides LPS_FIELD_WIDTH env var)')
    
    args = parser.parse_args()
    
    # Set field dimensions from CLI args if provided, otherwise use env vars
    if args.lps_length is not None:
        os.environ["LPS_FIELD_LENGTH"] = str(args.lps_length)
        print(f"📏 LPS Field Length set to: {args.lps_length} m (from CLI)")
    if args.lps_width is not None:
        os.environ["LPS_FIELD_WIDTH"] = str(args.lps_width)
        print(f"📏 LPS Field Width set to: {args.lps_width} m (from CLI)")
    
    # Display current field dimensions being used
    field_length = float(os.getenv("LPS_FIELD_LENGTH", "105.0"))
    field_width = float(os.getenv("LPS_FIELD_WIDTH", "60.0"))
    print(f"📐 Using LPS field dimensions: {field_length}m × {field_width}m")
    
    broker = os.getenv("MQTT_BROKER", "192.168.5.88")
    port = int(os.getenv("MQTT_PORT", "1883"))
    topic = os.getenv("MQTT_TOPIC", "rawData/+")
    client_id = os.getenv("MQTT_CLIENT_ID") or None

    try:
        subscriber = RawDataSubscriber(
            broker=broker,
            port=port,
            topic=topic,
            client_id=client_id
        )
        subscriber.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
