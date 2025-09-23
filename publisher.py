#!/usr/bin/env python3
"""
Multi-Mode Data Publisher for Athlete Monitoring:

Supports two distinct modes:
1. TRAINING MODE: Structured training sessions with predictable patterns
2. GAME MODE: Dynamic game scenarios with realistic match conditions

Usage:
    python publisher.py <num_players> [--mode training|game] [--duration minutes] [--players list]
    
Examples:
    python publisher.py 30 --mode training                    # Training session for all 30 players
    python publisher.py 5 --mode game --duration 90           # 90-minute game simulation for 5 random players
    python publisher.py 1 --mode training                     # Single random player training session
    python publisher.py --players [2,9,28,12,5] --mode game   # Game mode for specific players 2,9,28,12,5
    python publisher.py --players [1,3,7] --mode training     # Training mode for specific players 1,3,7

When stopped (Ctrl+C), saves data to appropriate folders based on mode
"""

import paho.mqtt.client as mqtt
import time
import json
import random
import sys
import os
import csv
import signal
import argparse
import math
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import glob
import re

class MultiModeDataPublisher:
    def __init__(self, num_players: int = None, mode: str = "training", 
                 duration_minutes: int = None, specific_players: List[int] = None):
        self.num_players = num_players
        self.mode = mode.lower()
        self.duration_minutes = duration_minutes
        self.specific_players = specific_players
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self._connect_mqtt()
        
        # Validate mode
        if self.mode not in ["training", "game"]:
            raise ValueError("Mode must be 'training' or 'game'")
        
        # Data storage for each player
        self.player_data: Dict[int, List[Dict]] = {}
        self.player_profiles: Dict[int, Dict] = {}
        
        # Select which players to generate data for
        self.selected_players = self._select_players(num_players, specific_players)
        
        # Initialize player profiles (consistent during session)
        self._initialize_player_profiles()
        
        # Session timing
        self.session_start_time = datetime.now()
        self.data_points_per_second = 10  # 10 Hz sampling rate
        self.running = True
        
        # Mode-specific settings
        self._setup_mode_specific_settings()
        
        # Display startup information
        self._display_startup_info()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Connection monitoring
        self.last_connection_check = time.time()
        self.connection_check_interval = 30  # Check every 30 seconds
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print(f"✅ Connected to MQTT broker successfully")
        else:
            print(f"❌ Failed to connect to MQTT broker with result code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback with improved reconnection logic"""
        print(f"⚠️  Disconnected from MQTT broker (code: {rc})")
        if rc != 0:
            print("🔄 Disconnect detected, will attempt reconnection in main loop...")
            # Don't reconnect here - let the main loop handle it
            # This prevents potential threading issues
    
    def _connect_mqtt(self):
        """Connect to MQTT broker with retry logic and better error handling"""
        max_retries = 5
        retry_count = 0
        
        # Disconnect first if already connected
        if self.client.is_connected():
            self.client.disconnect()
        
        while retry_count < max_retries:
            try:
                print(f"🔄 MQTT connection attempt {retry_count + 1}/{max_retries}...")
                result = self.client.connect("localhost", 1883, 60)
                if result == mqtt.MQTT_ERR_SUCCESS:
                    print("✅ MQTT connection established")
                    return
                else:
                    raise Exception(f"Connection failed with code: {result}")
            except Exception as e:
                retry_count += 1
                print(f"❌ MQTT connection attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    print(f"🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("❌ Failed to connect to MQTT broker after 5 attempts")
                    raise ConnectionError("Could not connect to MQTT broker")
    
    def _check_connection_status(self):
        """Check and report MQTT connection status"""
        current_time = time.time()
        if current_time - self.last_connection_check >= self.connection_check_interval:
            if self.client.is_connected():
                print(f"📡 MQTT connection status: Connected")
            else:
                print(f"📡 MQTT connection status: Disconnected")
            self.last_connection_check = current_time
    
    def _select_players(self, num_players: int = None, specific_players: List[int] = None) -> Set[int]:
        """Select which players to generate data for."""
        if specific_players:
            # Use specific player list
            invalid_players = [p for p in specific_players if p < 1 or p > 30]
            if invalid_players:
                raise ValueError(f"Invalid player IDs: {invalid_players}. Must be between 1-30.")
            
            selected = set(specific_players)
            print(f"🎯 Using specific players: {sorted(selected)}")
            return selected
        
        elif num_players is not None:
            if num_players >= 30:
                # Generate for all 30 players
                return set(range(1, 31))
            else:
                # Randomly select specified number of players
                selected = set(random.sample(range(1, 31), num_players))
                print(f"🎲 Randomly selected {num_players} players: {sorted(selected)}")
                return selected
        else:
            raise ValueError("Must specify either num_players or specific_players")
    
    def _initialize_player_profiles(self):
        """Initialize consistent player profiles for the session."""
        for player_id in self.selected_players:
            # Generate consistent profile for this player during the session
            self.player_profiles[player_id] = {
                "athlete_id": player_id,
        "age": random.randint(18, 35),
        "weight": round(random.uniform(60.0, 100.0), 1),
        "height": round(random.uniform(160.0, 200.0), 1),
                "gender": random.choice(["Male", "Female"]),
                "device_id": f"{player_id:03d}"  # 001, 002, etc.
            }
            
            # Initialize data storage for this player
            self.player_data[player_id] = []
            
            # Store additional game-specific attributes
            if self.mode == "game":
                self.player_profiles[player_id].update({
                    "position": random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"]),
                    "fitness_level": random.uniform(0.7, 1.0),  # Current fitness (70-100%)
                    "fatigue_rate": random.uniform(0.8, 1.2),   # How quickly they tire
                    "injury_risk": random.uniform(0.0, 0.3),    # Injury susceptibility
                    "playing_style": random.choice(["Aggressive", "Moderate", "Conservative"])
                })
            
            print(f"   Player {player_id}: Age={self.player_profiles[player_id]['age']}, "
                  f"Weight={self.player_profiles[player_id]['weight']}kg, "
                  f"Height={self.player_profiles[player_id]['height']}cm, "
                  f"Gender={self.player_profiles[player_id]['gender']}")
            
            if self.mode == "game":
                print(f"      Position: {self.player_profiles[player_id]['position']}, "
                      f"Style: {self.player_profiles[player_id]['playing_style']}")
    
    def _setup_mode_specific_settings(self):
        """Setup mode-specific configuration."""
        if self.mode == "training":
            self.session_type = "Training Session"
            self.data_folder = "athlete_training_data"
            self.file_prefix = "TR"
            # Training sessions: structured phases
            self.phase_duration = 10 * 60  # 10-minute phases
            self.phases = ["warmup", "active", "cooldown"]
        else:  # game mode
            self.session_type = "Game Session"
            self.data_folder = "athlete_game_data"
            self.file_prefix = "GM"
            # Game sessions: dynamic periods
            self.phase_duration = 45 * 60 if self.duration_minutes is None else (self.duration_minutes * 60) // 2
            self.phases = ["first_half", "halftime", "second_half"]
            
            # Game-specific events
            self.game_events = []
            self.next_event_time = random.uniform(5, 15)  # Next event in 5-15 seconds
    
    def _display_startup_info(self):
        """Display startup information based on mode."""
        mode_emoji = "🏃" if self.mode == "training" else "⚽"
        duration_text = f" ({self.duration_minutes} min)" if self.duration_minutes else ""
        
        print(f"{mode_emoji} {self.session_type} Publisher Started{duration_text}")
        print(f"📊 Generating data for {len(self.selected_players)} players: {sorted(self.selected_players)}")
        print(f"📡 Publishing to MQTT and collecting {self.mode} data...")
        print(f"💾 Data will be saved to: {self.data_folder}/")
        print(f"⏹️  Press Ctrl+C to stop and save {self.mode} data")
    
    def _get_next_sequence_number(self, player_id: int) -> int:
        """Get the next sequence number for a player based on mode."""
        player_dir = f"{self.data_folder}/player_{player_id}"
        
        if not os.path.exists(player_dir):
            return 1
        
        # Find existing files with the appropriate prefix
        pattern = f"{self.file_prefix}*.csv"
        files = glob.glob(os.path.join(player_dir, pattern))
        
        if not files:
            return 1
        
        # Extract sequence numbers
        sequence_numbers = []
        for file_path in files:
            filename = os.path.basename(file_path)
            match = re.search(f'{self.file_prefix}(\\d+)', filename)
            if match:
                sequence_numbers.append(int(match.group(1)))
        
        # Return next sequence number
        return max(sequence_numbers) + 1 if sequence_numbers else 1
    
    def _generate_sensor_data(self, player_id: int, timestamp: datetime) -> Dict:
        """Generate realistic sensor data for a player based on mode."""
        profile = self.player_profiles[player_id]
        time_factor = (timestamp - self.session_start_time).total_seconds()
        
        if self.mode == "training":
            return self._generate_training_data(profile, time_factor, timestamp)
        else:
            return self._generate_game_data(profile, time_factor, timestamp)
    
    def _generate_training_data(self, profile: Dict, time_factor: float, timestamp: datetime) -> Dict:
        """Generate structured training session data."""
        # Training has predictable phases: warm-up, active, cool-down
        activity_phase = (time_factor / 60) % 10  # 10-minute cycles
        
        if activity_phase < 2:
            # Warm-up phase - lower intensity, gradual increase
            phase_name = "warmup"
            acc_multiplier = 0.3 + (activity_phase / 2) * 0.3  # 0.3 to 0.6
            hr_base = 70 + (activity_phase / 2) * 20  # 70 to 90 bpm
            intensity = "Low"
        elif activity_phase < 7:
            # Active phase - higher intensity, sustained effort
            phase_name = "active"
            acc_multiplier = 0.8 + random.uniform(-0.2, 0.4)  # 0.6 to 1.2
            hr_base = 120 + random.uniform(-15, 25)  # 105 to 145 bpm
            intensity = "High"
        else:
            # Cool-down phase - decreasing intensity
            phase_name = "cooldown"
            cooldown_progress = (activity_phase - 7) / 3  # 0 to 1
            acc_multiplier = 0.6 * (1 - cooldown_progress * 0.7)  # 0.6 to 0.18
            hr_base = 100 - cooldown_progress * 30  # 100 to 70 bpm
            intensity = "Low"
        
        # Generate sensor data with training characteristics
        acc_x = round(random.uniform(-1.5, 1.5) * acc_multiplier, 6)
        acc_y = round(random.uniform(-1.5, 1.5) * acc_multiplier, 6)
        acc_z = round(random.uniform(9.0, 11.0) + random.uniform(-0.8, 0.8) * acc_multiplier, 6)
        
        gyro_x = round(random.uniform(-0.8, 0.8) * acc_multiplier, 6)
        gyro_y = round(random.uniform(-0.8, 0.8) * acc_multiplier, 6)
        gyro_z = round(random.uniform(-0.8, 0.8) * acc_multiplier, 6)
        
        # Heart rate with training progression
        hr_max = 220 - profile["age"]
        hr_variation = random.uniform(-8, 12)
        heart_rate = max(60, min(hr_max, hr_base + hr_variation))
        
        return self._create_sensor_dict(profile, timestamp, acc_x, acc_y, acc_z, 
                                      gyro_x, gyro_y, gyro_z, heart_rate, 
                                      phase_name, intensity)
    
    def _generate_game_data(self, profile: Dict, time_factor: float, timestamp: datetime) -> Dict:
        """Generate dynamic game scenario data."""
        # Game has unpredictable intensity based on events and player state
        
        # Determine game phase
        if self.duration_minutes:
            total_duration = self.duration_minutes * 60
            if time_factor < total_duration * 0.45:
                phase_name = "first_half"
            elif time_factor < total_duration * 0.55:
                phase_name = "halftime"
            else:
                phase_name = "second_half"
        else:
            # Default 90-minute game
            if time_factor < 45 * 60:
                phase_name = "first_half"
            elif time_factor < 50 * 60:
                phase_name = "halftime"
            else:
                phase_name = "second_half"
        
        # Player fatigue increases over time
        fatigue_factor = min(1.0, time_factor / (60 * 60))  # Fatigue over 60 minutes
        player_fatigue = fatigue_factor * profile.get("fatigue_rate", 1.0)
        
        # Generate game events (sprints, tackles, shots, etc.)
        event_intensity = self._get_game_event_intensity(time_factor, profile)
        
        # Position-based activity patterns
        position_multiplier = self._get_position_multiplier(profile.get("position", "Midfielder"))
        
        # Playing style affects intensity
        style_multiplier = {"Aggressive": 1.3, "Moderate": 1.0, "Conservative": 0.7}[
            profile.get("playing_style", "Moderate")
        ]
        
        # Calculate overall intensity
        base_intensity = 0.6 if phase_name == "halftime" else 0.8
        total_intensity = (base_intensity + event_intensity) * position_multiplier * style_multiplier
        total_intensity *= (1.0 - player_fatigue * 0.3)  # Reduce with fatigue
        
        # Generate more variable sensor data for games
        acc_multiplier = total_intensity * random.uniform(0.7, 1.5)
        
        acc_x = round(random.uniform(-3.0, 3.0) * acc_multiplier, 6)
        acc_y = round(random.uniform(-3.0, 3.0) * acc_multiplier, 6)
        acc_z = round(random.uniform(8.5, 12.5) + random.uniform(-1.5, 1.5) * acc_multiplier, 6)
        
        gyro_x = round(random.uniform(-2.0, 2.0) * acc_multiplier, 6)
        gyro_y = round(random.uniform(-2.0, 2.0) * acc_multiplier, 6)
        gyro_z = round(random.uniform(-2.0, 2.0) * acc_multiplier, 6)
        
        # Heart rate with game dynamics
        hr_max = 220 - profile["age"]
        hr_base = 90 + total_intensity * 60  # 90-150 base range
        hr_variation = random.uniform(-15, 25)  # More variation in games
        heart_rate = max(70, min(hr_max, hr_base + hr_variation))
        
        # Determine intensity level
        if total_intensity < 0.4:
            intensity = "Low"
        elif total_intensity < 0.8:
            intensity = "Medium"
        else:
            intensity = "High"
        
        return self._create_sensor_dict(profile, timestamp, acc_x, acc_y, acc_z, 
                                      gyro_x, gyro_y, gyro_z, heart_rate, 
                                      phase_name, intensity)
    
    def _get_game_event_intensity(self, time_factor: float, profile: Dict) -> float:
        """Generate random game events that affect intensity."""
        # Check if it's time for a new event
        if time_factor >= self.next_event_time:
            event_type = random.choice([
                "sprint", "tackle", "shot", "pass", "dribble", "jump", "rest", "walk"
            ])
            
            event_intensities = {
                "sprint": 0.8, "tackle": 0.6, "shot": 0.7, "pass": 0.2,
                "dribble": 0.4, "jump": 0.5, "rest": -0.3, "walk": 0.1
            }
            
            # Schedule next event
            self.next_event_time = time_factor + random.uniform(3, 12)
            
            # Log significant events
            if event_type in ["sprint", "tackle", "shot"] and random.random() < 0.1:
                self.game_events.append({
                    "time": time_factor,
                    "player_id": profile["athlete_id"],
                    "event": event_type,
                    "intensity": event_intensities[event_type]
                })
            
            return event_intensities[event_type]
        
        return 0.0  # No special event
    
    def _get_position_multiplier(self, position: str) -> float:
        """Get activity multiplier based on player position."""
        position_multipliers = {
            "Forward": 1.2,      # High activity, lots of sprints
            "Midfielder": 1.3,   # Highest activity, box-to-box
            "Defender": 0.9,     # Moderate activity, positional
            "Goalkeeper": 0.4    # Low activity, mostly stationary
        }
        return position_multipliers.get(position, 1.0)
    
    def _create_sensor_dict(self, profile: Dict, timestamp: datetime, 
                          acc_x: float, acc_y: float, acc_z: float,
                          gyro_x: float, gyro_y: float, gyro_z: float,
                          heart_rate: float, phase: str, intensity: str) -> Dict:
        """Create the sensor data dictionary."""
        
        base_data = {
            "device_id": profile["device_id"],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "athlete_id": profile["athlete_id"],
            "age": profile["age"],
            "weight": profile["weight"],
            "height": profile["height"],
            "gender": 1 if profile["gender"] == "Male" else 0,  # For MQTT
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z,
        "mag_x": round(random.uniform(-65.0, 65.0), 2),
        "mag_y": round(random.uniform(-65.0, 65.0), 2),
        "mag_z": round(random.uniform(-65.0, 65.0), 2),
            "heart_rate_bpm": round(heart_rate, 1),
            "session_phase": phase,
            "intensity_level": intensity,
            "mode": self.mode
        }
        
        # Add mode-specific data
        if self.mode == "game":
            base_data.update({
                "position": profile.get("position", "Unknown"),
                "playing_style": profile.get("playing_style", "Moderate"),
                "fitness_level": profile.get("fitness_level", 1.0),
                "fatigue_rate": profile.get("fatigue_rate", 1.0)
            })
        
        return base_data
    
    def _save_session_data(self):
        """Save collected session data to CSV files based on mode."""
        mode_emoji = "🏃" if self.mode == "training" else "⚽"
        print(f"\n💾 Saving {self.mode} data...")
        
        base_dir = self.data_folder
        os.makedirs(base_dir, exist_ok=True)
        
        session_timestamp = self.session_start_time.strftime("%Y_%m_%d-%H_%M_%S")
        
        for player_id in self.selected_players:
            if not self.player_data[player_id]:
                continue
                
            # Create player directory
            player_dir = os.path.join(base_dir, f"player_{player_id}")
            os.makedirs(player_dir, exist_ok=True)
            
            # Get next sequence number
            sequence_num = self._get_next_sequence_number(player_id)
            
            # Create filename following the pattern
            device_id = f"{player_id:03d}"
            filename = f"{self.file_prefix}{sequence_num}_A{player_id}_D{device_id}_{session_timestamp}.csv"
            filepath = os.path.join(player_dir, filename)
            
            # Define CSV fieldnames based on mode
            base_fieldnames = [
                'timestamp', 'athlete_id', 'age', 'weight', 'height', 'gender',
                'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate',
                'session_phase', 'intensity_level', 'mode'
            ]
            
            if self.mode == "game":
                base_fieldnames.extend(['position', 'playing_style', 'fitness_level', 'fatigue_rate'])
            
            # Write CSV data
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=base_fieldnames)
                writer.writeheader()
                
                for data_point in self.player_data[player_id]:
                    # Convert data for CSV format
                    csv_row = {
                        'timestamp': data_point['timestamp'],
                        'athlete_id': data_point['athlete_id'],
                        'age': data_point['age'],
                        'weight': data_point['weight'],
                        'height': data_point['height'],
                        'gender': self.player_profiles[player_id]['gender'],  # Use string format
                        'acc_x': data_point['acc_x'],
                        'acc_y': data_point['acc_y'],
                        'acc_z': data_point['acc_z'],
                        'gyro_x': data_point['gyro_x'],
                        'gyro_y': data_point['gyro_y'],
                        'gyro_z': data_point['gyro_z'],
                        'heart_rate': int(data_point['heart_rate_bpm']),
                        'session_phase': data_point['session_phase'],
                        'intensity_level': data_point['intensity_level'],
                        'mode': data_point['mode']
                    }
                    
                    # Add game-specific fields
                    if self.mode == "game":
                        csv_row.update({
                            'position': data_point.get('position', ''),
                            'playing_style': data_point.get('playing_style', ''),
                            'fitness_level': data_point.get('fitness_level', 1.0),
                            'fatigue_rate': data_point.get('fatigue_rate', 1.0)
                        })
                    
                    writer.writerow(csv_row)
            
            data_count = len(self.player_data[player_id])
            duration = (datetime.now() - self.session_start_time).total_seconds()
            
            print(f"   ✅ Player {player_id}: {data_count} data points saved to {filename}")
            print(f"      📁 Location: {filepath}")
            print(f"      ⏱️  Duration: {duration:.1f}s, Rate: {data_count/duration:.1f} Hz")
            
            # Show mode-specific summary
            if self.mode == "game" and self.game_events:
                player_events = [e for e in self.game_events if e['player_id'] == player_id]
                if player_events:
                    print(f"      🎮 Game events: {len(player_events)} significant actions")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\n🛑 Stopping data collection...")
        self.running = False
        self._save_session_data()
        
        total_points = sum(len(data) for data in self.player_data.values())
        duration = (datetime.now() - self.session_start_time).total_seconds()
        
        mode_emoji = "🏃" if self.mode == "training" else "⚽"
        print(f"\n📊 {self.session_type.upper()} SUMMARY:")
        print(f"   Mode: {self.mode.title()} {mode_emoji}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Players: {len(self.selected_players)}")
        print(f"   Total data points: {total_points}")
        print(f"   Average rate: {total_points/duration:.1f} Hz")
        print(f"   Data saved to: {self.data_folder}/")
        
        # Mode-specific summary
        if self.mode == "game" and self.game_events:
            print(f"   Game events: {len(self.game_events)} significant actions recorded")
            event_types = {}
            for event in self.game_events:
                event_types[event['event']] = event_types.get(event['event'], 0) + 1
            print(f"   Event breakdown: {dict(event_types)}")
        
        # Stop MQTT loop and disconnect
        print("🔄 Stopping MQTT network loop...")
        self.client.loop_stop()
        self.client.disconnect()
        sys.exit(0)
    
    def run(self):
        """Main publishing loop with MQTT loop and connection monitoring."""
        try:
            # Start MQTT network loop to handle network communication
            print("🔄 Starting MQTT network loop...")
            self.client.loop_start()
            
            # Wait a moment for initial connection to establish
            time.sleep(1)
            
            while self.running:
                current_time = datetime.now()
                
                # Periodic connection status check
                self._check_connection_status()
                
                # Check MQTT connection status
                if not self.client.is_connected():
                    print("⚠️ MQTT disconnected, attempting reconnection...")
                    try:
                        self._connect_mqtt()
                        time.sleep(2)  # Give time for reconnection
                        if not self.client.is_connected():
                            print("❌ Reconnection failed, continuing without publishing...")
                            time.sleep(5)  # Wait before next attempt
                            continue
                        else:
                            print("✅ MQTT reconnected successfully")
                    except Exception as e:
                        print(f"❌ Reconnection error: {e}")
                        time.sleep(5)  # Wait before next attempt
                        continue
                
                # Generate and publish data for each selected player
                for player_id in self.selected_players:
                    try:
                        # Generate sensor data
                        sensor_data = self._generate_sensor_data(player_id, current_time)
                        
                        # Store for later saving
                        self.player_data[player_id].append(sensor_data.copy())
                        
                        # Publish to MQTT with QoS 1 for reliable delivery
                        topic = f"player/{sensor_data['device_id']}/sensor/data"
                        result = self.client.publish(topic, json.dumps(sensor_data), qos=1)
                        
                        # Check if publish was successful
                        if result.rc != mqtt.MQTT_ERR_SUCCESS:
                            print(f"⚠️ Failed to publish for player {player_id}: {result.rc}")
                        
                        # Show progress occasionally
                        if len(self.player_data[player_id]) % 100 == 0:
                            print(f"Player {player_id}: {len(self.player_data[player_id])} data points collected")
                            
                    except Exception as e:
                        print(f"❌ Error generating/publishing data for player {player_id}: {e}")
                        continue
                
                # Sleep to maintain desired frequency
                time.sleep(1.0 / self.data_points_per_second)
                
        except Exception as e:
            print(f"❌ Error during data generation: {e}")
            self._save_session_data()
            sys.exit(1)
        finally:
            # Stop MQTT loop and disconnect
            print("🔄 Stopping MQTT network loop...")
            self.client.loop_stop()
            self.client.disconnect()


def parse_player_list(player_string: str) -> List[int]:
    """Parse player list from string format like '[2,9,28,12,5]' or '2,9,28,12,5'."""
    try:
        # Remove brackets and whitespace
        clean_string = player_string.strip().strip('[]')
        
        # Split by comma and convert to integers
        player_ids = [int(x.strip()) for x in clean_string.split(',') if x.strip()]
        
        # Validate player IDs
        invalid_ids = [p for p in player_ids if p < 1 or p > 30]
        if invalid_ids:
            raise ValueError(f"Invalid player IDs: {invalid_ids}. Must be between 1-30.")
        
        # Remove duplicates while preserving order
        unique_players = []
        seen = set()
        for p in player_ids:
            if p not in seen:
                unique_players.append(p)
                seen.add(p)
        
        return unique_players
        
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError("Player list must contain only numbers separated by commas")
        raise e
    except Exception as e:
        raise ValueError(f"Invalid player list format: {e}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Mode Data Publisher for Athlete Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random player selection:
  python publisher.py 30 --mode training                    # All 30 players
  python publisher.py 5 --mode game --duration 90           # 5 random players
  python publisher.py 1 --mode training                     # 1 random player
  
  # Specific player selection:
  python publisher.py --players [2,9,28,12,5] --mode game   # Specific players for game
  python publisher.py --players [1,3,7] --mode training     # Specific players for training
  python publisher.py --players 2,9,28,12,5 --mode game     # Alternative format (no brackets)
  
  # Mixed usage:
  python publisher.py --players [11,22] --mode game --duration 45  # 45-min game for players 11,22
        """
    )
    
    # Make num_players optional when using --players
    parser.add_argument("num_players", type=int, nargs='?',
                       help="Number of random players (1-30). Not used if --players is specified.")
    parser.add_argument("--mode", choices=["training", "game"], default="training",
                       help="Data generation mode (default: training)")
    parser.add_argument("--duration", type=int, metavar="MINUTES",
                       help="Session duration in minutes (game mode only)")
    parser.add_argument("--players", type=str, metavar="LIST",
                       help="Specific player IDs as comma-separated list: [2,9,28,12,5] or 2,9,28,12,5")
    
    args = parser.parse_args()
    
    # Parse specific players if provided
    specific_players = None
    if args.players:
        try:
            specific_players = parse_player_list(args.players)
            print(f"🎯 Parsed specific players: {specific_players}")
        except ValueError as e:
            print(f"❌ Error parsing player list: {e}")
            print("💡 Format examples: [2,9,28,12,5] or 2,9,28,12,5")
            sys.exit(1)
    
    # Validate arguments
    if not specific_players and not args.num_players:
        print("❌ Must specify either num_players or --players list")
        parser.print_help()
        sys.exit(1)
    
    if args.num_players and specific_players:
        print("⚠️  Both num_players and --players specified. Using --players list.")
        args.num_players = None
    
    if args.num_players and (args.num_players < 1 or args.num_players > 30):
        print("❌ Number of players must be between 1 and 30")
        sys.exit(1)
    
    if args.duration and args.mode == "training":
        print("⚠️  Duration parameter ignored for training mode")
        args.duration = None
    
    if args.mode == "game" and args.duration and args.duration < 10:
        print("❌ Game duration must be at least 10 minutes")
        sys.exit(1)
    
    try:
        # Create and run publisher
        publisher = MultiModeDataPublisher(
            num_players=args.num_players,
            mode=args.mode, 
            duration_minutes=args.duration,
            specific_players=specific_players
        )
        publisher.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
