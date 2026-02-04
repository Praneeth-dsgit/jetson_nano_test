#!/usr/bin/env python3
"""
Live Position System (LPS) Visualization Tool
Real-time tracking of athlete positions on a football field using MQTT data.

This tool integrates with the existing athlete monitoring system to provide
visual tracking of player positions and G-impact events.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import paho.mqtt.client as mqtt
import json
import glob
import os
import time
import logging
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import signal
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LPSConfig:
    """Configuration for LPS visualization loaded from YAML file"""
    
    def __init__(self, config_path: str = "lps_config.yaml"):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                config = {}
            
            # MQTT settings
            mqtt_config = config.get('mqtt', {})
            self.mqtt_broker = mqtt_config.get('broker', 'localhost')
            self.mqtt_port = mqtt_config.get('port', 1883)
            self.mqtt_timeout = mqtt_config.get('timeout', 60)
            self.mqtt_keepalive = mqtt_config.get('keepalive', 60)
            self.mqtt_reconnect_delay = mqtt_config.get('reconnect_delay', 5)
            
            # Device settings
            device_config = config.get('devices', {})
            self.num_devices = device_config.get('count', 30)
            self.position_timeout = device_config.get('position_timeout', 5.0)
            
            topic_patterns = device_config.get('topic_patterns', {})
            self.device_topic_pattern = topic_patterns.get('lps_specific', 'lps/{device_id}')
            self.legacy_topic_pattern = topic_patterns.get('legacy', 'player/{device_id}/sensor/data')
            
            # Field settings - check environment variables first, then config file, then defaults
            field_config = config.get('field', {})
            # Use same env vars as subscriber for consistency
            self.field_length = float(str(field_config.get('length', 17.5)))
            self.field_width = float(str(field_config.get('width', 10)))
            
            # FIFA standard dimensions for scaling reference
            FIFA_LENGTH = 105.0
            FIFA_WIDTH = 60.0
            FIFA_CENTER_CIRCLE_RADIUS = 9.15
            FIFA_PENALTY_LENGTH = 16.5
            FIFA_PENALTY_WIDTH = 40.3
            FIFA_GOAL_WIDTH = 7.32
            
            # Calculate scaling factors based on FIFA standard
            # Always scale from FIFA standard to maintain proper proportions
            length_scale = self.field_length / FIFA_LENGTH
            width_scale = self.field_width / FIFA_WIDTH
            avg_scale = (length_scale + width_scale) / 2
            
            # Scale inner elements proportionally from FIFA standard
            # Center circle uses average scale (circular element)
            self.center_circle_radius = FIFA_CENTER_CIRCLE_RADIUS * avg_scale
            
            # Penalty area scales with field dimensions (rectangular elements)
            self.penalty_length = FIFA_PENALTY_LENGTH * length_scale
            self.penalty_width = FIFA_PENALTY_WIDTH * width_scale
            
            # Goal width scales with field width
            self.goal_width = FIFA_GOAL_WIDTH * width_scale
            
            field_viz = field_config.get('visualization', {})
            self.field_bg_color = field_viz.get('background_color', '#228B22')
            self.field_line_color = field_viz.get('line_color', 'white')
            self.goal_color = field_viz.get('goal_color', 'yellow')
            
            # Visualization settings
            viz_config = config.get('visualization', {})
            self.figure_size = viz_config.get('figure_size', [16, 10])
            self.update_interval = viz_config.get('update_interval', 200)
            
            player_marker = viz_config.get('player_marker', {})
            self.player_color = player_marker.get('color', 'blue')
            self.player_size = player_marker.get('size', 8)
            self.player_label = player_marker.get('label', 'Players')
            
            player_labels = viz_config.get('player_labels', {})
            self.label_fontsize = player_labels.get('fontsize', 8)
            self.label_color = player_labels.get('color', 'blue')
            self.label_weight = player_labels.get('weight', 'bold')
            self.label_offset = player_labels.get('offset', [1, 1])
            self.tag_title_prefix = player_labels.get('tag_title_prefix', 'Tag: ')
            
            # Anchor titles (from anchors config if present)
            anchor_config = config.get('anchors', {})
            self.anchor_titles = anchor_config.get('titles', ['A1', 'A2', 'A3', 'A4'])
            
            # G-impact settings
            g_impact_config = config.get('g_impact', {})
            self.g_impact_log_pattern = g_impact_config.get('log_pattern', '*/*_g_impact_log.json')
            
            g_marker = g_impact_config.get('marker', {})
            self.g_impact_color = g_marker.get('color', 'red')
            self.g_impact_marker = g_marker.get('marker', 'D')
            self.g_impact_size = g_marker.get('size', 100)
            self.g_impact_label = g_marker.get('label', 'G-Impact Events')
            
            g_labels = g_impact_config.get('labels', {})
            self.g_label_fontsize = g_labels.get('fontsize', 8)
            self.g_label_color = g_labels.get('color', 'red')
            self.g_label_weight = g_labels.get('weight', 'bold')
            self.g_label_offset = g_labels.get('offset', [1, 1])
            self.g_show_fields = g_labels.get('show_fields', ['device_id', 'athlete_id', 'name'])
            
            # Logging settings
            logging_config = config.get('logging', {})
            self.log_level = logging_config.get('level', 'INFO')
            self.log_format = logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
            self.log_file = logging_config.get('file')
            
            # Performance settings
            perf_config = config.get('performance', {})
            self.max_fps = perf_config.get('max_fps', 10)
            self.position_history_size = perf_config.get('position_history_size', 100)
            self.enable_blit = perf_config.get('enable_blit', False)
            
            # Integration settings
            integration_config = config.get('integration', {})
            self.use_athlete_profiles = integration_config.get('use_athlete_profiles', True)
            self.athlete_data_path = integration_config.get('athlete_data_path', 'data/athlete_training_data')
            self.sync_with_main_system = integration_config.get('sync_with_main_system', True)
            
            position_fields = integration_config.get('position_fields', {})
            self.x_field = position_fields.get('x', 'x')
            self.y_field = position_fields.get('y', 'y')
            self.device_id_field = position_fields.get('device_id', 'device_id')
            self.athlete_id_field = position_fields.get('athlete_id', 'athlete_id')
            self.timestamp_field = position_fields.get('timestamp', 'timestamp')
            
            # Session settings
            session_config = config.get('session', {})
            self.default_duration_minutes = session_config.get('default_duration_minutes')
            self.default_mode = session_config.get('default_mode', 'game')
            self.auto_stop_on_duration = session_config.get('auto_stop_on_duration', True)
            
            # Status display settings
            status_display = session_config.get('status_display', {})
            self.status_enabled = status_display.get('enabled', True)
            self.status_position = status_display.get('position', [5, 55])
            self.status_fontsize = status_display.get('fontsize', 10)
            self.status_color = status_display.get('color', 'white')
            self.status_bg_color = status_display.get('background_color', 'black')
            self.status_bg_alpha = status_display.get('background_alpha', 0.7)
            self.show_active_devices = status_display.get('show_active_devices', True)
            self.show_elapsed_time = status_display.get('show_elapsed_time', True)
            self.show_remaining_time = status_display.get('show_remaining_time', True)
            self.show_session_mode = status_display.get('show_session_mode', True)
            
            # Initialize session settings (will be overridden by command line if provided)
            self.session_duration_minutes = None
            self.session_mode = self.default_mode
            
            # Heatmap settings
            heatmap_config = config.get('heatmap', {})
            self.heatmap_enabled = heatmap_config.get('enabled', False)
            self.heatmap_auto_save = heatmap_config.get('auto_save_on_exit', True)
            self.heatmap_bins = heatmap_config.get('bins', [30, 20])
            self.heatmap_colormap = heatmap_config.get('colormap', 'hot')
            self.heatmap_alpha = heatmap_config.get('alpha', 0.6)
            self.heatmap_save_data = heatmap_config.get('save_session_data', True)
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use defaults
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values"""
        self.mqtt_broker = "localhost"
        self.mqtt_port = 1883
        self.mqtt_timeout = 60
        self.mqtt_keepalive = 60
        self.mqtt_reconnect_delay = 5
        self.num_devices = 30
        self.position_timeout = 5.0
        self.device_topic_pattern = "lps/{device_id}"
        self.legacy_topic_pattern = "player/{device_id}/sensor/data"
        self.field_length = 105.0
        self.field_width = 60.0
        
        # FIFA standard dimensions for scaling reference
        FIFA_LENGTH = 105.0
        FIFA_WIDTH = 60.0
        FIFA_CENTER_CIRCLE_RADIUS = 9.15
        FIFA_PENALTY_LENGTH = 16.5
        FIFA_PENALTY_WIDTH = 40.3
        FIFA_GOAL_WIDTH = 7.32
        
        # Calculate scaling factors
        length_scale = self.field_length / FIFA_LENGTH
        width_scale = self.field_width / FIFA_WIDTH
        avg_scale = (length_scale + width_scale) / 2
        
        # Scale inner elements proportionally
        self.center_circle_radius = FIFA_CENTER_CIRCLE_RADIUS * avg_scale
        self.penalty_length = FIFA_PENALTY_LENGTH * length_scale
        self.penalty_width = FIFA_PENALTY_WIDTH * width_scale
        self.goal_width = FIFA_GOAL_WIDTH * width_scale
        self.field_bg_color = '#228B22'
        self.field_line_color = 'white'
        self.goal_color = 'yellow'
        self.figure_size = [16, 10]
        self.update_interval = 200
        self.player_color = 'blue'
        self.player_size = 8
        self.player_label = 'Players'
        self.label_fontsize = 8
        self.label_color = 'blue'
        self.label_weight = 'bold'
        self.label_offset = [1, 1]
        self.tag_title_prefix = 'Tag: '
        self.anchor_titles = ['A1', 'A2', 'A3', 'A4']
        self.g_impact_log_pattern = '*/*_g_impact_log.json'
        self.g_impact_color = 'red'
        self.g_impact_marker = 'D'
        self.g_impact_size = 100
        self.g_impact_label = 'G-Impact Events'
        self.g_label_fontsize = 8
        self.g_label_color = 'red'
        self.g_label_weight = 'bold'
        self.g_label_offset = [1, 1]
        self.g_show_fields = ['device_id', 'athlete_id', 'name']
        self.log_level = 'INFO'
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        self.log_file = None
        self.max_fps = 10
        self.position_history_size = 100
        self.enable_blit = False
        self.use_athlete_profiles = True
        self.athlete_data_path = 'data/athlete_training_data'
        self.sync_with_main_system = True
        self.x_field = 'x'
        self.y_field = 'y'
        self.device_id_field = 'device_id'
        self.athlete_id_field = 'athlete_id'
        self.timestamp_field = 'timestamp'
        
        # Session settings (these will be set by command line or config)
        self.session_duration_minutes = None  # None = unlimited
        self.session_mode = 'game'  # Default mode
        
        # Default session configuration
        self.default_duration_minutes = None
        self.default_mode = 'game'
        self.auto_stop_on_duration = True
        
        # Status display settings
        self.status_enabled = True
        self.status_position = [5, 55]
        self.status_fontsize = 10
        self.status_color = 'white'
        self.status_bg_color = 'black'
        self.status_bg_alpha = 0.7
        self.show_active_devices = True
        self.show_elapsed_time = True
        self.show_remaining_time = True
        self.show_session_mode = True
        
        # Heatmap defaults
        self.heatmap_enabled = False
        self.heatmap_auto_save = True
        self.heatmap_bins = [30, 20]
        self.heatmap_colormap = 'hot'
        self.heatmap_alpha = 0.6
        self.heatmap_save_data = True

class LPSVisualizer:
    """Live Position System Visualizer with MQTT integration"""
    
    def _normalize_device_id(self, device_id: str) -> str:
        """
        Normalize device ID to PM### format.
        Handles: "001", "1", "PM001", "PM1" -> "PM001"
        """
        if not device_id:
            return "PM000"
        
        device_id = str(device_id).strip()
        
        # Remove PM prefix if present to extract numeric part
        if device_id.upper().startswith("PM"):
            numeric_part = device_id[2:].lstrip('0') or "0"
        else:
            numeric_part = device_id.lstrip('0') or "0"
        
        try:
            # Convert to int and format as PM### 
            device_num = int(numeric_part)
            return f"PM{device_num:03d}"
        except ValueError:
            return "PM000"
    
    def __init__(self, config: LPSConfig):
        self.config = config
        # Use PM001, PM002, ... format for device IDs
        self.device_ids = [f"PM{i:03d}" for i in range(1, config.num_devices + 1)]
        
        # Store latest positions for each device {device_id: (x, y, timestamp)}
        self.device_positions: Dict[str, Tuple[float, float, float]] = {
            dev_id: (np.nan, np.nan, 0.0) for dev_id in self.device_ids
        }
        
        # Store G-impact events
        self.g_impact_events: List[Dict] = []
        
        # Heatmap data collection
        self.heatmap_data = {'x': [], 'y': []}  # Store all positions for heatmap
        self.heatmap_enabled = getattr(config, 'heatmap_enabled', False)
        
        # MQTT client and connection management
        self.client = None
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.last_connection_attempt = 0
        self.running = True
        
        # Athlete profiles for better labeling
        self.athlete_profiles: Dict[str, Dict] = {}
        
        # Session management
        self.session_start_time = time.time()
        self.session_duration_minutes = config.session_duration_minutes or config.default_duration_minutes
        self.session_mode = config.session_mode or config.default_mode
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize visualization
        self._setup_plot()
        self._load_g_impact_events()
        self._load_athlete_profiles()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def _load_athlete_profiles(self):
        """Load athlete profiles from existing system if available"""
        if not self.config.use_athlete_profiles:
            return
            
        try:
            athlete_path = Path(self.config.athlete_data_path)
            if not athlete_path.exists():
                logger.warning(f"Athlete data path {athlete_path} does not exist")
                return
            
            # Look for player directories
            for player_dir in athlete_path.glob("player_*"):
                try:
                    player_id = player_dir.name.split("_")[1]
                    # Try to find a CSV file to extract athlete info
                    csv_files = list(player_dir.glob("*.csv"))
                    if csv_files:
                        # Read first few lines to get athlete info
                        with open(csv_files[0], 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                # Parse header and first data row
                                header = lines[0].strip().split(',')
                                data = lines[1].strip().split(',')
                                
                                # Create profile dict
                                profile = {}
                                for i, field in enumerate(header):
                                    if i < len(data):
                                        profile[field] = data[i]
                                
                                self.athlete_profiles[player_id] = profile
                                logger.debug(f"Loaded profile for player {player_id}")
                except Exception as e:
                    logger.warning(f"Failed to load profile for {player_dir.name}: {e}")
            
            logger.info(f"Loaded {len(self.athlete_profiles)} athlete profiles")
        except Exception as e:
            logger.error(f"Error loading athlete profiles: {e}")
        
    def create_heatmap(self, output_file: str = None) -> str:
        """Create a heatmap from collected position data"""
        if not self.heatmap_data['x']:
            logger.warning("No position data available for heatmap")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw football field
        self._draw_football_field(ax)
        
        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(
            self.heatmap_data['x'], self.heatmap_data['y'], 
            bins=[30, 20], 
            range=[[0, self.config.field_length], [0, self.config.field_width]]
        )
        
        # Rotate and flip for proper display
        heatmap = np.rot90(heatmap)
        heatmap = np.flipud(heatmap)
        
        # Display heatmap
        extent = [0, self.config.field_length, 0, self.config.field_width]
        im = ax.imshow(heatmap, extent=extent, cmap='hot', alpha=0.6, 
                      origin='lower', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Position Frequency')
        
        # Labels and title
        ax.set_title('Player Position Heatmap (Live Session)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        # Add statistics
        total_positions = len(self.heatmap_data['x'])
        ax.text(0.02, 0.98, f'Total Positions: {total_positions}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"live_heatmap_{timestamp}.png"
        
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {output_file}")
        
        plt.show()
        return output_file
    
    def save_session_data(self, output_file: str = None) -> str:
        """Save collected session data to JSON file"""
        if not self.heatmap_data['x']:
            logger.warning("No position data available to save")
            return None
            
        session_data = {
            'x': self.heatmap_data['x'],
            'y': self.heatmap_data['y'],
            'session_start': getattr(self, 'session_start_time', time.time()),
            'session_end': time.time(),
            'total_positions': len(self.heatmap_data['x']),
            'field_dimensions': {
                'length': self.config.field_length,
                'width': self.config.field_width
            }
        }
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"session_data_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        logger.info(f"Session data saved to: {output_file}")
        return output_file

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Received shutdown signal, stopping visualization...")
        
        # Save heatmap and session data if enabled
        if self.heatmap_enabled and self.heatmap_data['x']:
            try:
                self.create_heatmap()
                self.save_session_data()
            except Exception as e:
                logger.error(f"Error saving heatmap/session data: {e}")
        
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        sys.exit(0)
    
    def _setup_plot(self):
        """Setup the matplotlib plot with football field"""
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size)
        
        # Set plot limits and labels
        self.ax.set_xlim(-5, self.config.field_length + 5)
        self.ax.set_ylim(-5, self.config.field_width + 5)
        self.ax.set_xlabel("X Position (m)", fontsize=12, color='white')
        self.ax.set_ylabel("Y Position (m)", fontsize=12, color='white')
        # Create dynamic title based on session info
        title = f"Live Position System (LPS) - FIFA Football Ground"
        if self.session_mode:
            title += f" - {self.session_mode.title()} Mode"
        if self.session_duration_minutes:
            title += f" - {self.session_duration_minutes}min"
        
        self.ax.set_title(title, fontsize=14, color='white', fontweight='bold')
        self.ax.set_aspect('equal')
        
        # Set background color from config
        self.ax.set_facecolor(self.config.field_bg_color)
        
        # Draw football field
        self._draw_football_field()
        
        # Plot sensor anchors (corners)
        self._plot_anchors()
        
        # Setup dynamic elements
        self._setup_dynamic_elements()
        
        # Add dimension labels
        self._add_dimension_labels()
        
        # Add status text for session info
        self._add_status_display()
        
    def _draw_football_field(self):
        """Draw FIFA standard football field markings"""
        # Outer boundary
        field = plt.Rectangle((0, 0), self.config.field_length, self.config.field_width, 
                            linewidth=2, edgecolor=self.config.field_line_color, facecolor='none')
        self.ax.add_patch(field)
        
        # Center line
        self.ax.plot([self.config.field_length/2, self.config.field_length/2], 
                    [0, self.config.field_width], color=self.config.field_line_color, linewidth=2)
        
        # Center circle
        center_circle = plt.Circle((self.config.field_length/2, self.config.field_width/2), 
                                 self.config.center_circle_radius, 
                                 color=self.config.field_line_color, fill=False, linewidth=2)
        self.ax.add_patch(center_circle)
        
        # Penalty boxes
        left_penalty = plt.Rectangle((0, (self.config.field_width-self.config.penalty_width)/2), 
                                   self.config.penalty_length, self.config.penalty_width, 
                                   linewidth=2, edgecolor=self.config.field_line_color, facecolor='none')
        right_penalty = plt.Rectangle((self.config.field_length-self.config.penalty_length, 
                                     (self.config.field_width-self.config.penalty_width)/2), 
                                    self.config.penalty_length, self.config.penalty_width, 
                                    linewidth=2, edgecolor=self.config.field_line_color, facecolor='none')
        self.ax.add_patch(left_penalty)
        self.ax.add_patch(right_penalty)
        
        # Goals
        self.ax.plot([0, 0], 
                    [self.config.field_width/2 - self.config.goal_width/2, 
                     self.config.field_width/2 + self.config.goal_width/2], 
                    color=self.config.goal_color, linewidth=4)
        self.ax.plot([self.config.field_length, self.config.field_length], 
                    [self.config.field_width/2 - self.config.goal_width/2, 
                     self.config.field_width/2 + self.config.goal_width/2], 
                    color=self.config.goal_color, linewidth=4)
        
        self.ax.grid(False)
    
    def _plot_anchors(self):
        """Plot sensor anchor positions: A1=top mid, A2=right mid, A3=bottom mid, A4=left mid
        
        Reads from LPS_ANCHOR_1 to LPS_ANCHOR_4 env vars if set, otherwise uses defaults.
        """
        # Default positions (midpoints)
        defaults = [
            (self.config.field_length / 2, self.config.field_width),   # A1: Top mid
            (self.config.field_length, self.config.field_width / 2),   # A2: Right mid
            (self.config.field_length / 2, 0),                         # A3: Bottom mid
            (0, self.config.field_width / 2)                           # A4: Left mid
        ]
        
        # Read from env vars if set
        anchor_positions = []
        for i in range(1, 5):
            env_var = f"LPS_ANCHOR_{i}"
            anchor_str = os.getenv(env_var)
            if anchor_str:
                try:
                    parts = anchor_str.split(',')
                    if len(parts) == 2:
                        x, y = float(parts[0].strip()), float(parts[1].strip())
                        anchor_positions.append((x, y))
                        logger.debug(f"Loaded {env_var} from env: ({x}, {y})")
                    else:
                        anchor_positions.append(defaults[i-1])
                except (ValueError, AttributeError):
                    anchor_positions.append(defaults[i-1])
            else:
                anchor_positions.append(defaults[i-1])
        
        sensors = np.array(anchor_positions)
        logger.info(f"Using anchor positions: A1={anchor_positions[0]}, A2={anchor_positions[1]}, A3={anchor_positions[2]}, A4={anchor_positions[3]}")
        
        self.ax.scatter(sensors[:, 0], sensors[:, 1], 
                       color='red', label="LPS Anchors", s=100, zorder=5)
        
        # Add tag titles for each anchor (A1 above, A2 right, A3 below, A4 left)
        scale = max(self.config.field_length, self.config.field_width) * 0.02
        offsets = [(0, scale), (scale, 0), (0, -scale), (-scale, 0)]   # above, right, below, left
        haligns = ['center', 'left', 'center', 'right']
        valigns = ['bottom', 'center', 'top', 'center']
        for i, (s, o, ha, va) in enumerate(zip(sensors, offsets, haligns, valigns)):
            title = self.config.anchor_titles[i] if i < len(self.config.anchor_titles) else f'A{i+1}'
            self.ax.text(s[0] + o[0], s[1] + o[1], title,
                         fontsize=9, color='white', weight='bold',
                         ha=ha, va=va, zorder=5)
    
    def _setup_dynamic_elements(self):
        """Setup dynamic plot elements for device tracking"""
        # Player positions
        self.points, = self.ax.plot([], [], f'{self.config.player_color[0]}o', 
                                   markersize=self.config.player_size, 
                                   label=self.config.player_label, zorder=6)
        
        # Player labels
        self.labels = [self.ax.text(0, 0, "", fontsize=self.config.label_fontsize, 
                                   color=self.config.label_color, 
                                   weight=self.config.label_weight, zorder=7) 
                      for _ in range(self.config.num_devices)]
        
        # G-impact markers (static, loaded from files)
        self._plot_g_impact_events()
        
        # Legend
        self.ax.legend(loc='upper right', facecolor='black', 
                      edgecolor='white', labelcolor='white')
    
    def _add_dimension_labels(self):
        """Add field dimension labels"""
        self.ax.text(self.config.field_length/2, -3, 
                    f'{self.config.field_length} m', 
                    color='white', ha='center', va='top', 
                    fontsize=12, fontweight='bold')
        self.ax.text(-3, self.config.field_width/2, 
                    f'{self.config.field_width} m', 
                    color='white', ha='right', va='center', 
                    fontsize=12, fontweight='bold', rotation=90)
    
    def _add_status_display(self):
        """Add status display showing session information"""
        if not self.config.status_enabled:
            return
            
        # Status text in configured position
        self.status_text = self.ax.text(
            self.config.status_position[0], 
            self.config.status_position[1], 
            "", 
            fontsize=self.config.status_fontsize, 
            color=self.config.status_color, 
            weight='bold', 
            zorder=20,
            bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor=self.config.status_bg_color, 
                    alpha=self.config.status_bg_alpha)
        )
    
    def _load_g_impact_events(self):
        """Load G-impact events from JSON log files"""
        try:
            g_impact_files = glob.glob(self.config.g_impact_log_pattern)
            logger.info(f"Found {len(g_impact_files)} G-impact log files")
            
            for g_file in g_impact_files:
                try:
                    with open(g_file, 'r') as f:
                        events = json.load(f)
                        if isinstance(events, list):
                            self.g_impact_events.extend(events)
                        else:
                            self.g_impact_events.append(events)
                except Exception as e:
                    logger.warning(f"Failed to load G-impact file {g_file}: {e}")
            
            logger.info(f"Loaded {len(self.g_impact_events)} G-impact events")
        except Exception as e:
            logger.error(f"Error loading G-impact events: {e}")
    
    def _plot_g_impact_events(self):
        """Plot G-impact events as configured markers"""
        if not self.g_impact_events:
            return
            
        xs = [e["x"] for e in self.g_impact_events 
              if e.get("x") is not None and e.get("y") is not None]
        ys = [e["y"] for e in self.g_impact_events 
              if e.get("x") is not None and e.get("y") is not None]
        
        if xs and ys:
            self.ax.scatter(xs, ys, s=self.config.g_impact_size, 
                           c=self.config.g_impact_color, 
                           marker=self.config.g_impact_marker, 
                           label=self.config.g_impact_label, zorder=10)
            
            # Add labels for G-impact events
            for e in self.g_impact_events:
                if e.get("x") is not None and e.get("y") is not None:
                    label_parts = []
                    for field in self.config.g_show_fields:
                        if e.get(field):
                            if field == "device_id":
                                label_parts.append(f"Dev:{e[field]}")
                            elif field == "athlete_id":
                                label_parts.append(f"Ath:{e[field]}")
                            else:
                                label_parts.append(str(e[field]))
                    
                    if label_parts:
                        label = " | ".join(label_parts)
                        self.ax.text(e["x"] + self.config.g_label_offset[0], 
                                   e["y"] + self.config.g_label_offset[1], 
                                   label, fontsize=self.config.g_label_fontsize, 
                                   color=self.config.g_label_color, 
                                   weight=self.config.g_label_weight, zorder=11)
    
    def _setup_mqtt(self):
        """Setup MQTT client and callbacks with retry logic"""
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Set connection options
        self.client.connect_async(self.config.mqtt_broker, self.config.mqtt_port, 
                                 self.config.mqtt_keepalive)
        self.client.loop_start()
        
        # Wait for connection with retry logic
        timeout = self.config.mqtt_timeout
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.connected:
            raise ConnectionError(f"Failed to connect to MQTT broker within {timeout} seconds")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker: {self.config.mqtt_broker}:{self.config.mqtt_port}")
            
            # Subscribe to LPS topics and predictions topics
            for dev_id in self.device_ids:
                # Try both LPS-specific and legacy topics
                lps_topic = self.config.device_topic_pattern.format(device_id=dev_id)
                legacy_topic = self.config.legacy_topic_pattern.format(device_id=dev_id)
                predictions_topic = f"predictions/{dev_id}"
                
                client.subscribe(lps_topic)
                client.subscribe(legacy_topic)
                client.subscribe(predictions_topic)
                logger.debug(f"Subscribed to {lps_topic}, {legacy_topic}, and {predictions_topic}")
        else:
            self.connected = False
            logger.error(f"Failed to connect to MQTT broker with result code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (code: {rc})")
        else:
            logger.info("MQTT disconnected")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback with improved error handling"""
        try:
            logger.debug(f"Received MQTT message on topic: {msg.topic}")
            payload = json.loads(msg.payload.decode())
            
            # Extract device ID using configured field names
            device_id = str(payload.get(self.config.device_id_field, '')).strip()
            if not device_id:
                # Try to extract from topic
                if 'player/' in msg.topic:
                    try:
                        device_id = msg.topic.split('/')[1]
                    except IndexError:
                        logger.warning(f"Could not extract device ID from topic: {msg.topic}")
                        return
                elif msg.topic.startswith("predictions/"):
                    # Extract from predictions/PM001 format
                    try:
                        device_id = msg.topic.split("/")[1]
                    except IndexError:
                        logger.warning(f"Could not extract device ID from predictions topic: {msg.topic}")
                        return
                else:
                    logger.warning(f"No device ID in payload and topic doesn't match pattern: {msg.topic}")
                    return
            
            # Normalize device ID to PM### format (preserve PM prefix)
            original_device_id = device_id
            device_id = self._normalize_device_id(device_id)
            logger.debug(f"Device ID normalized: '{original_device_id}' -> '{device_id}'")
            
            # Extract position data - check multiple possible locations
            x = None
            y = None
            
            # Try direct x, y fields first
            x = payload.get(self.config.x_field)
            y = payload.get(self.config.y_field)
            
            # Try nested position object (from predictions topic)
            if x is None or y is None:
                position_obj = payload.get("position")
                if position_obj and isinstance(position_obj, dict):
                    x = position_obj.get("x")
                    y = position_obj.get("y")
                    logger.debug(f"Found position in nested object: x={x}, y={y}")
            # Try positions object (from lps/+ topic)
            if x is None or y is None:
                positions_obj = payload.get("positions")
                if positions_obj and isinstance(positions_obj, dict):
                    x = positions_obj.get("x")
                    y = positions_obj.get("y")
                    logger.debug(f"Found position in positions object: x={x}, y={y}")
            
            if x is not None and y is not None:
                try:
                    # Check if device_id exists in our tracking list
                    if device_id not in self.device_positions:
                        logger.warning(f"Device {device_id} not in tracking list. Available devices: {list(self.device_positions.keys())[:5]}...")
                        # Try to add it anyway (might be a new device)
                        self.device_positions[device_id] = (np.nan, np.nan, 0.0)
                    
                    # Update position with timestamp
                    self.device_positions[device_id] = (float(x), float(y), time.time())
                    # Log lps/ updates at INFO (primary low-latency source); predictions/ at DEBUG to reduce noise
                    if msg.topic.startswith("lps/"):
                        logger.info(f"[OK] Updated position for device {device_id}: ({x:.2f}, {y:.2f}) from topic {msg.topic}")
                    else:
                        logger.debug(f"Updated position for device {device_id}: ({x:.2f}, {y:.2f}) from topic {msg.topic}")
                    
                    # Collect data for heatmap if enabled
                    if self.heatmap_enabled:
                        self.heatmap_data['x'].append(float(x))
                        self.heatmap_data['y'].append(float(y))
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid position data for device {device_id}: x={x}, y={y}, error={e}")
            else:
                logger.debug(f"No position data in message for device {device_id} (topic: {msg.topic})")
                logger.debug(f"Message payload keys: {list(payload.keys())}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    def _update_visualization(self, frame):
        """Update function for animation"""
        if not self.running:
            return []
        
        # Check session duration
        if self.session_duration_minutes is not None and self.session_duration_minutes > 0:
            elapsed_minutes = (time.time() - self.session_start_time) / 60.0
            if elapsed_minutes >= self.session_duration_minutes:
                logger.info(f"Session duration ({self.session_duration_minutes} minutes) reached, stopping visualization")
                self.running = False
                return []
        
        # Collect current positions
        xs, ys = [], []
        current_time = time.time()
        
        for dev_id in self.device_ids:
            x, y, timestamp = self.device_positions[dev_id]
            
            # Only show positions that are recent (within configured timeout)
            if not np.isnan(x) and not np.isnan(y) and (current_time - timestamp) < self.config.position_timeout:
                xs.append(x)
                ys.append(y)
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        
        # Update player positions
        self.points.set_data(xs, ys)
        
        # Update player labels with athlete names if available
        # Scale label offset proportionally to field size (for small fields, use smaller offset)
        # Base offset is for 105m × 60m field, scale down for smaller fields
        scale_factor = min(self.config.field_length / 105.0, self.config.field_width / 60.0)
        scale_factor = max(0.1, scale_factor)  # Minimum 10% of base offset
        scaled_offset_x = self.config.label_offset[0] * scale_factor
        scaled_offset_y = self.config.label_offset[1] * scale_factor
        
        for idx, (x, y) in enumerate(zip(xs, ys)):
            if not np.isnan(x) and not np.isnan(y):
                dev_id = self.device_ids[idx]
                
                # Create tag title with device ID and athlete name if available
                prefix = getattr(self.config, 'tag_title_prefix', '') or ''
                label_text = f"{prefix}{dev_id}"
                if dev_id in self.athlete_profiles:
                    profile = self.athlete_profiles[dev_id]
                    athlete_id = profile.get('athlete_id', '')
                    if athlete_id:
                        label_text = f"{prefix}{dev_id}\nA{athlete_id}"
                
                self.labels[idx].set_position((x + scaled_offset_x, 
                                             y + scaled_offset_y))
                self.labels[idx].set_text(label_text)
                self.labels[idx].set_visible(True)
            else:
                self.labels[idx].set_visible(False)
        
        # Update status display
        self._update_status_display()
        
        return [self.points] + self.labels + [self.status_text]
    
    def _update_status_display(self):
        """Update the status display with session information"""
        if not hasattr(self, 'status_text') or not self.config.status_enabled:
            return
            
        current_time = time.time()
        elapsed_seconds = current_time - self.session_start_time
        elapsed_minutes = elapsed_seconds / 60.0
        
        # Count active devices
        active_devices = sum(1 for dev_id in self.device_ids 
                           if not np.isnan(self.device_positions[dev_id][0]))
        
        # Build status text based on configuration
        status_parts = []
        
        if self.config.show_active_devices:
            status_parts.append(f"Devices: {active_devices}/{len(self.device_ids)}")
        
        if self.config.show_elapsed_time:
            status_parts.append(f"Elapsed: {elapsed_minutes:.1f}m")
        
        if self.session_duration_minutes and self.config.show_remaining_time:
            remaining_minutes = self.session_duration_minutes - elapsed_minutes
            if remaining_minutes > 0:
                status_parts.append(f"Remaining: {remaining_minutes:.1f}m")
            else:
                status_parts.append("Time: EXPIRED")
        
        if self.session_mode and self.config.show_session_mode:
            status_parts.append(f"Mode: {self.session_mode.title()}")
        
        if status_parts:
            self.status_text.set_text("\n".join(status_parts))
        else:
            self.status_text.set_text("")
    
    def run(self):
        """Start the LPS visualization"""
        try:
            logger.info("Starting LPS Visualization...")
            
            # Setup MQTT connection
            self._setup_mqtt()
            
            # Start animation
            ani = FuncAnimation(self.fig, self._update_visualization, 
                              interval=self.config.update_interval, 
                              blit=False, cache_frame_data=False)
            
            logger.info("LPS Visualization started. Close the window to stop.")
            plt.show()
            
        except KeyboardInterrupt:
            logger.info("Visualization interrupted by user")
        except Exception as e:
            logger.error(f"Error running visualization: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.running = False
        
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        plt.close(self.fig)

def main():
    """Main entry point"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LPS Visualization Tool')
    parser.add_argument('--config', '-c', default='lps_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--devices', '-d', type=int, help='Number of devices to track')
    parser.add_argument('--duration', type=int, help='Duration in minutes (0 = unlimited)')
    parser.add_argument('--mode', choices=['training', 'game'], help='Session mode')
    parser.add_argument('--heatmap', action='store_true', help='Enable heatmap data collection')
    parser.add_argument('--no-heatmap', action='store_true', help='Disable heatmap data collection')
    parser.add_argument('--length', type=float, help='Field length in meters (default: from LPS_FIELD_LENGTH env var or config)')
    parser.add_argument('--width', type=float, help='Field width in meters (default: from LPS_FIELD_WIDTH env var or config)')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = LPSConfig(args.config)
        
        # Apply command line overrides
        if args.debug:
            config.log_level = 'DEBUG'
        if args.devices:
            config.num_devices = args.devices
        if args.duration is not None:
            config.session_duration_minutes = args.duration
        if args.mode:
            config.session_mode = args.mode
        if args.heatmap:
            config.heatmap_enabled = True
        if args.no_heatmap:
            config.heatmap_enabled = False
        if args.length:
            config.field_length = args.length
            # Also set env var so subscriber can use same value
            os.environ["LPS_FIELD_LENGTH"] = str(args.length)
            logger.info(f"📏 Field length set to {args.length}m (from CLI)")
        if args.width:
            config.field_width = args.width
            # Also set env var so subscriber can use same value
            os.environ["LPS_FIELD_WIDTH"] = str(args.width)
            logger.info(f"📏 Field width set to {args.width}m (from CLI)")
        
        # Log final field dimensions being used
        logger.info(f"📐 Visualization using field dimensions: {config.field_length}m × {config.field_width}m")
        
        # Recalculate scaled inner elements if field dimensions changed
        if args.length or args.width:
            # FIFA standard dimensions for scaling reference
            FIFA_LENGTH = 105.0
            FIFA_WIDTH = 60.0
            FIFA_CENTER_CIRCLE_RADIUS = 9.15
            FIFA_PENALTY_LENGTH = 16.5
            FIFA_PENALTY_WIDTH = 40.3
            FIFA_GOAL_WIDTH = 7.32
            
            # Calculate scaling factors
            length_scale = config.field_length / FIFA_LENGTH
            width_scale = config.field_width / FIFA_WIDTH
            avg_scale = (length_scale + width_scale) / 2
            
            # Scale inner elements proportionally
            config.center_circle_radius = FIFA_CENTER_CIRCLE_RADIUS * avg_scale
            config.penalty_length = FIFA_PENALTY_LENGTH * length_scale
            config.penalty_width = FIFA_PENALTY_WIDTH * width_scale
            config.goal_width = FIFA_GOAL_WIDTH * width_scale
        
        logger.info(f"Starting LPS Visualization with {config.num_devices} devices")
        logger.info(f"MQTT Broker: {config.mqtt_broker}:{config.mqtt_port}")
        if config.session_duration_minutes:
            logger.info(f"Session Duration: {config.session_duration_minutes} minutes")
        if config.session_mode:
            logger.info(f"Session Mode: {config.session_mode}")
        
        # Create and run visualizer
        visualizer = LPSVisualizer(config)
        visualizer.run()
        
    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
