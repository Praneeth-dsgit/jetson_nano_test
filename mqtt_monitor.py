#!/usr/bin/env python3
"""
MQTT Connection Monitor
Monitors MQTT broker status and message flow in real-time
"""

import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime

class MQTTMonitor:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.connected = False
        self.message_count = 0
        self.last_message_time = 0
        self.device_activity = {}
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"✅ {self.timestamp()} - Connected to MQTT Broker successfully")
            # Subscribe to all topics
            client.subscribe("player/+/sensor/data")
            client.subscribe("sensor/data")
            client.subscribe("+/predictions")
            print(f"📡 {self.timestamp()} - Subscribed to monitoring topics")
        else:
            self.connected = False
            print(f"❌ {self.timestamp()} - Failed to connect to MQTT Broker (code: {rc})")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"⚠️  {self.timestamp()} - Disconnected from MQTT Broker (code: {rc})")
    
    def on_message(self, client, userdata, msg):
        self.message_count += 1
        self.last_message_time = time.time()
        topic = msg.topic
        
        try:
            # Try to parse as JSON
            data = json.loads(msg.payload.decode())
            
            if "sensor/data" in topic or "player/" in topic:
                # Sensor data message
                device_id = data.get("device_id", "unknown")
                player_id = data.get("athlete_id", device_id)
                
                if device_id not in self.device_activity:
                    self.device_activity[device_id] = {"count": 0, "last_seen": 0}
                
                self.device_activity[device_id]["count"] += 1
                self.device_activity[device_id]["last_seen"] = time.time()
                
                print(f"📥 {self.timestamp()} - Sensor data from Player {player_id} (Device {device_id}) - Total messages: {self.message_count}")
                
            elif "/predictions" in topic:
                # Prediction result message
                device_id = topic.split("/")[0]
                print(f"📤 {self.timestamp()} - Prediction result for Device {device_id}")
                
        except json.JSONDecodeError:
            print(f"📨 {self.timestamp()} - Non-JSON message on {topic}: {msg.payload.decode()[:50]}...")
        except Exception as e:
            print(f"⚠️  {self.timestamp()} - Error processing message: {e}")
    
    def timestamp(self):
        return datetime.now().strftime("%H:%M:%S")
    
    def print_status(self):
        status = "🟢 Connected" if self.connected else "🔴 Disconnected"
        print(f"\n📊 {self.timestamp()} - MQTT Monitor Status:")
        print(f"   Connection: {status}")
        print(f"   Total messages: {self.message_count}")
        
        if self.last_message_time > 0:
            time_since_last = time.time() - self.last_message_time
            print(f"   Last message: {time_since_last:.1f}s ago")
        else:
            print(f"   Last message: Never")
            
        if self.device_activity:
            print(f"   Active devices: {len(self.device_activity)}")
            for device_id, info in self.device_activity.items():
                time_since = time.time() - info["last_seen"]
                print(f"     Device {device_id}: {info['count']} messages, last seen {time_since:.1f}s ago")
        print("-" * 60)
    
    def run(self):
        print("🔍 Starting MQTT Monitor...")
        print("📡 Attempting to connect to MQTT broker at localhost:1883")
        
        try:
            self.client.connect("localhost", 1883, 60)
            self.client.loop_start()
            
            last_status_time = 0
            
            while True:
                current_time = time.time()
                
                # Print status every 10 seconds
                if current_time - last_status_time >= 10:
                    self.print_status()
                    last_status_time = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n🛑 {self.timestamp()} - Monitor stopped by user")
        except Exception as e:
            print(f"❌ {self.timestamp()} - Error: {e}")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    monitor = MQTTMonitor()
    monitor.run()
