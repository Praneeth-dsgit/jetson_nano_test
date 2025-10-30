#!/usr/bin/env python3
"""
Player Position Heatmap Analyzer
Creates heatmaps from session data showing player position density over time.
Supports both offline analysis and real-time MQTT-based heatmap generation.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import glob
import argparse
from datetime import datetime
import paho.mqtt.client as mqtt
import threading
import time
import signal
import sys

class HeatmapAnalyzer:
    """Analyzes player position data and creates heatmaps"""
    
    def __init__(self, field_length: float = 105.0, field_width: float = 68.0):
        self.field_length = field_length
        self.field_width = field_width
        
    def find_latest_session_file(self, directory: str = ".") -> Optional[str]:
        """Find the latest session JSON file"""
        pattern = os.path.join(directory, "*realtime_output*.json")
        files = glob.glob(pattern)
        
        if not files:
            return None
            
        # Sort by modification time, newest first
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    def load_session_data(self, file_path: str) -> Dict:
        """Load session data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise FileNotFoundError(f'Error loading session data: {e}')
    
    def extract_positions(self, session_data: Dict) -> Tuple[List[float], List[float]]:
        """Extract x and y positions from session data"""
        xs = session_data.get('x', [])
        ys = session_data.get('y', [])
        
        # Handle different data formats
        if not isinstance(xs, list):
            xs = [xs] if xs is not None else []
        if not isinstance(ys, list):
            ys = [ys] if ys is not None else []
            
        # Filter out None values and ensure we have valid coordinates
        valid_positions = [(x, y) for x, y in zip(xs, ys) 
                          if x is not None and y is not None 
                          and 0 <= x <= self.field_length 
                          and 0 <= y <= self.field_width]
        
        if not valid_positions:
            raise ValueError("No valid position data found in session")
            
        xs, ys = zip(*valid_positions)
        return list(xs), list(ys)
    
    def draw_football_field(self, ax):
        """Draw a football field with proper markings"""
        # Set field background
        ax.set_facecolor('#228B22')
        ax.set_xlim(0, self.field_length)
        ax.set_ylim(0, self.field_width)
        ax.set_aspect('equal')
        
        # Field outline
        ax.plot([0, self.field_length, self.field_length, 0, 0], 
                [0, 0, self.field_width, self.field_width, 0], 
                color='white', linewidth=2)
        
        # Center line
        ax.plot([self.field_length/2, self.field_length/2], 
                [0, self.field_width], color='white', linewidth=2)
        
        # Center circle
        center_circle = plt.Circle((self.field_length/2, self.field_width/2), 9.15, 
                                 color='white', fill=False, linewidth=2)
        ax.add_patch(center_circle)
        
        # Goal areas
        goal_area_width = 16.5
        goal_area_height = 40.3
        goal_area_y = (self.field_width - goal_area_height) / 2
        
        # Left goal area
        ax.add_patch(plt.Rectangle((0, goal_area_y), goal_area_width, goal_area_height, 
                                 linewidth=2, edgecolor='white', facecolor='none'))
        
        # Right goal area
        ax.add_patch(plt.Rectangle((self.field_length - goal_area_width, goal_area_y), 
                                 goal_area_width, goal_area_height, 
                                 linewidth=2, edgecolor='white', facecolor='none'))
        
        # Penalty areas (larger rectangles)
        penalty_area_width = 40.3
        penalty_area_height = 16.5
        penalty_area_y = (self.field_width - penalty_area_height) / 2
        
        # Left penalty area
        ax.add_patch(plt.Rectangle((0, penalty_area_y), penalty_area_width, penalty_area_height, 
                                 linewidth=2, edgecolor='white', facecolor='none'))
        
        # Right penalty area
        ax.add_patch(plt.Rectangle((self.field_length - penalty_area_width, penalty_area_y), 
                                 penalty_area_width, penalty_area_height, 
                                 linewidth=2, edgecolor='white', facecolor='none'))
    
    def create_heatmap(self, xs: List[float], ys: List[float], 
                      bins: Tuple[int, int] = (30, 20),
                      cmap: str = 'hot', alpha: float = 0.6) -> plt.Figure:
        """Create a heatmap from position data"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw football field
        self.draw_football_field(ax)
        
        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(
            xs, ys, 
            bins=bins, 
            range=[[0, self.field_length], [0, self.field_width]]
        )
        
        # Rotate and flip for proper display
        heatmap = np.rot90(heatmap)
        heatmap = np.flipud(heatmap)
        
        # Display heatmap
        extent = [0, self.field_length, 0, self.field_width]
        im = ax.imshow(heatmap, extent=extent, cmap=cmap, alpha=alpha, 
                      origin='lower', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Position Frequency')
        
        # Labels and title
        ax.set_title('Player Position Heatmap (Full Match Session)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        # Add statistics
        total_positions = len(xs)
        ax.text(0.02, 0.98, f'Total Positions: {total_positions}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def analyze_session(self, file_path: str = None, output_file: str = None) -> str:
        """Analyze a session and create heatmap"""
        
        # Find session file if not provided
        if file_path is None:
            file_path = self.find_latest_session_file()
            if file_path is None:
                raise FileNotFoundError('No session output JSON file found.')
        
        print(f"📊 Analyzing session data from: {file_path}")
        
        # Load and process data
        session_data = self.load_session_data(file_path)
        xs, ys = self.extract_positions(session_data)
        
        print(f"✅ Loaded {len(xs)} position data points")
        
        # Create heatmap
        fig = self.create_heatmap(xs, ys)
        
        # Save or show
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"player_heatmap_{timestamp}.png"
        
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Heatmap saved to: {output_file}")
        
        plt.show()
        return output_file


class RealtimeHeatmapGenerator:
    """Real-time per-player heatmap generator via MQTT"""
    
    def __init__(self, output_dir: str = "player_heatmaps", 
                 field_length: float = 105.0, field_width: float = 68.0,
                 update_interval: int = 30):
        """
        Initialize real-time heatmap generator
        
        Args:
            output_dir: Directory to save heatmaps
            field_length: Football field length in meters
            field_width: Football field width in meters
            update_interval: How often to update heatmaps (seconds)
        """
        self.output_dir = output_dir
        self.field_length = field_length
        self.field_width = field_width
        self.update_interval = update_interval
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Per-player position tracking
        self.player_positions: Dict[str, Dict[str, List[float]]] = {}
        # Format: {athlete_id: {'x': [x1, x2, ...], 'y': [y1, y2, ...]}}
        
        # MQTT setup
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Control flags
        self.running = True
        self.connected = False
        
        # Threading
        self.update_thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.total_positions_received = 0
        self.last_update_time = time.time()
        
        # Heatmap analyzer
        self.analyzer = HeatmapAnalyzer(field_length, field_width)
        
        print(f"🎨 Real-time Heatmap Generator initialized")
        print(f"   Output directory: {os.path.abspath(self.output_dir)}")
        print(f"   Update interval: {self.update_interval}s")
        print(f"   Field size: {field_length}m x {field_width}m")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            print(f"✅ Heatmap Generator connected to MQTT broker")
            # Subscribe to all player sensor data
            client.subscribe("player/+/sensor/data")
            client.subscribe("sensor/data")
            print(f"   Subscribed to player/+/sensor/data")
        else:
            print(f"❌ Heatmap Generator failed to connect to MQTT broker (code: {rc})")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.connected = False
        print(f"⚠️  Heatmap Generator disconnected from MQTT broker (code: {rc})")
    
    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        try:
            data = json.loads(msg.payload.decode())
            
            # Extract player/athlete ID
            athlete_id = str(data.get("athlete_id", "unknown"))
            device_id = str(data.get("device_id", "")).strip()
            
            # Skip if no valid ID
            if athlete_id == "unknown" and not device_id:
                return
            
            # Use athlete_id if available, otherwise device_id
            player_id = athlete_id if athlete_id != "unknown" else device_id
            
            # Extract position data
            x = data.get("x")
            y = data.get("y")
            
            if x is None or y is None:
                return
            
            # Validate position is within field bounds
            if not (0 <= x <= self.field_length and 0 <= y <= self.field_width):
                return
            
            # Store position data
            with self.lock:
                if player_id not in self.player_positions:
                    self.player_positions[player_id] = {'x': [], 'y': []}
                    print(f"🎯 Tracking Player {player_id} for heatmap generation")
                
                self.player_positions[player_id]['x'].append(x)
                self.player_positions[player_id]['y'].append(y)
                self.total_positions_received += 1
                
                # Show progress every 100 positions
                if self.total_positions_received % 100 == 0:
                    print(f"📊 Collected {self.total_positions_received} positions from {len(self.player_positions)} players")
                
        except Exception as e:
            print(f"⚠️  Error processing message for heatmap: {e}")
    
    def _generate_player_heatmap(self, player_id: str, xs: List[float], ys: List[float]):
        """Generate and save heatmap for a specific player"""
        try:
            if len(xs) < 10:  # Need minimum data points
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw football field
            self.analyzer.draw_football_field(ax)
            
            # Create heatmap
            bins = (30, 20)
            heatmap, xedges, yedges = np.histogram2d(
                xs, ys,
                bins=bins,
                range=[[0, self.field_length], [0, self.field_width]]
            )
            
            # Rotate and flip for proper display
            heatmap = np.rot90(heatmap)
            heatmap = np.flipud(heatmap)
            
            # Display heatmap
            extent = [0, self.field_length, 0, self.field_width]
            im = ax.imshow(heatmap, extent=extent, cmap='hot', alpha=0.6,
                          origin='lower', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Position Frequency')
            
            # Labels and title
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(f'Player {player_id} Position Heatmap\n{timestamp}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position (meters)', fontsize=12)
            ax.set_ylabel('Y Position (meters)', fontsize=12)
            
            # Add statistics
            ax.text(0.02, 0.98, f'Total Positions: {len(xs)}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, f"player_{player_id}_heatmap.png")
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return output_file
            
        except Exception as e:
            print(f"⚠️  Error generating heatmap for Player {player_id}: {e}")
            return None
    
    def _update_heatmaps(self):
        """Periodically update all player heatmaps"""
        while self.running:
            time.sleep(self.update_interval)
            
            if not self.connected:
                continue
            
            current_time = time.time()
            time_since_update = current_time - self.last_update_time
            
            with self.lock:
                if not self.player_positions:
                    continue
                
                print(f"\n🎨 Updating heatmaps for {len(self.player_positions)} players...")
                
                for player_id, positions in self.player_positions.items():
                    xs = positions['x']
                    ys = positions['y']
                    
                    if len(xs) >= 10:  # Minimum data points
                        output_file = self._generate_player_heatmap(player_id, xs, ys)
                        if output_file:
                            print(f"   ✅ Player {player_id}: {len(xs)} positions -> {os.path.basename(output_file)}")
                
                print(f"   Total positions processed: {self.total_positions_received}")
                self.last_update_time = current_time
    
    def start(self):
        """Start the real-time heatmap generator"""
        try:
            # Connect to MQTT broker
            print("🔄 Connecting to MQTT broker...")
            self.client.connect("localhost", 1883, 60)
            
            # Start MQTT loop
            self.client.loop_start()
            
            # Wait for connection
            time.sleep(2)
            
            if not self.connected:
                print("⚠️  Failed to connect to MQTT broker")
                return
            
            # Start heatmap update thread
            self.update_thread = threading.Thread(target=self._update_heatmaps, daemon=True)
            self.update_thread.start()
            print(f"✅ Heatmap update thread started (interval: {self.update_interval}s)")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            print(f"\n🎨 Real-time Heatmap Generator running...")
            print(f"   Heatmaps will be saved to: {os.path.abspath(self.output_dir)}/")
            print(f"   Press Ctrl+C to stop\n")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error starting heatmap generator: {e}")
            self.stop()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Stopping heatmap generator...")
        self.stop()
    
    def stop(self):
        """Stop the heatmap generator"""
        self.running = False
        
        # Generate final heatmaps
        print(f"\n💾 Generating final heatmaps...")
        with self.lock:
            for player_id, positions in self.player_positions.items():
                xs = positions['x']
                ys = positions['y']
                
                if len(xs) >= 10:
                    output_file = self._generate_player_heatmap(player_id, xs, ys)
                    if output_file:
                        print(f"   ✅ Player {player_id}: Final heatmap saved ({len(xs)} positions)")
        
        # Stop MQTT
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        print(f"\n📊 HEATMAP GENERATION SUMMARY:")
        print(f"   Total positions received: {self.total_positions_received}")
        print(f"   Players tracked: {len(self.player_positions)}")
        print(f"   Heatmaps saved to: {os.path.abspath(self.output_dir)}/")
        
        sys.exit(0)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Real-time Player Position Heatmap Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start real-time heatmap generation:
  python heatmap_analyzer.py
  python heatmap_analyzer.py --output-dir player_heatmaps --update-interval 30
  python heatmap_analyzer.py --field-length 105 --field-width 68 --update-interval 60
        """
    )
    
    # Configuration arguments
    parser.add_argument('--field-length', type=float, default=105.0, 
                       help='Field length in meters (default: 105.0)')
    parser.add_argument('--field-width', type=float, default=68.0, 
                       help='Field width in meters (default: 68.0)')
    parser.add_argument('--output-dir', default='player_heatmaps',
                       help='Output directory for heatmaps (default: player_heatmaps)')
    parser.add_argument('--update-interval', type=int, default=30,
                       help='Heatmap update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Use non-interactive backend for file output only
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"📊 Real-time Heatmap Generator")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Update interval: {args.update_interval}s")
    print(f"   Field size: {args.field_length}m x {args.field_width}m")
    
    try:
        print("🎨 Starting Real-time Heatmap Generator...")
        
        generator = RealtimeHeatmapGenerator(
            output_dir=args.output_dir,
            field_length=args.field_length,
            field_width=args.field_width,
            update_interval=args.update_interval
        )
        generator.start()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
