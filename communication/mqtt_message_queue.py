#!/usr/bin/env python3
"""
MQTT Message Queue with Persistent Storage and Retry Logic

This module provides reliable MQTT message delivery with:
- Persistent message storage
- Automatic retry with exponential backoff
- Message queuing and batch processing
- Delivery confirmation tracking
- Dead letter queue for failed messages
"""

import json
import time
import sqlite3
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import paho.mqtt.client as mqtt
import os
from dataclasses import dataclass, asdict
from enum import Enum

class MessageStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class QueuedMessage:
    """Represents a message in the queue."""
    id: str
    topic: str
    payload: str
    qos: int
    retain: bool
    timestamp: float
    status: MessageStatus
    retry_count: int
    max_retries: int
    next_retry_time: float
    error_message: Optional[str] = None

class MQTTMessageQueue:
    """
    Reliable MQTT message queue with persistent storage and retry logic.
    
    Features:
    - Persistent SQLite storage for message queuing
    - Automatic retry with exponential backoff
    - Delivery confirmation tracking
    - Dead letter queue for permanently failed messages
    - Batch processing for efficiency
    """
    
    def __init__(self, 
                 db_path: str = "mqtt_message_queue.db",
                 max_retries: int = 5,
                 retry_delay: float = 1.0,
                 max_retry_delay: float = 300.0,
                 batch_size: int = 10,
                 enable_logging: bool = True):
        """
        Initialize the MQTT message queue.
        
        Args:
            db_path: Path to SQLite database for persistent storage
            max_retries: Maximum number of retry attempts
            retry_delay: Initial retry delay in seconds
            max_retry_delay: Maximum retry delay in seconds
            batch_size: Number of messages to process in each batch
            enable_logging: Whether to enable detailed logging
        """
        self.db_path = db_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.batch_size = batch_size
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Initialize database
        self._init_database()
        
        # Message queue and processing
        self.message_queue = deque()
        self.processing = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # MQTT client (will be set by external code)
        self.mqtt_client = None
        
        # Statistics
        self.stats = {
            "messages_queued": 0,
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "messages_retried": 0,
            "queue_size": 0
        }
        
        # Start processing thread
        self._start_processing_thread()
    
    def _init_database(self):
        """Initialize SQLite database for message storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    qos INTEGER NOT NULL,
                    retain BOOLEAN NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    max_retries INTEGER NOT NULL,
                    next_retry_time REAL NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status_retry ON messages(status, next_retry_time)')
            
            conn.commit()
            conn.close()
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Message queue database initialized: {self.db_path}")
                
        except Exception as e:
            error_msg = f"Failed to initialize message queue database: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def set_mqtt_client(self, client: mqtt.Client):
        """Set the MQTT client for message publishing."""
        self.mqtt_client = client
        
        # Set up delivery confirmation callback
        client.on_publish = self._on_publish_callback
        
        if self.enable_logging and self.logger:
            self.logger.info("MQTT client set for message queue")
    
    def _on_publish_callback(self, client, userdata, mid):
        """Callback for MQTT publish confirmation."""
        try:
            # Update message status to delivered
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE messages 
                SET status = ?, error_message = NULL
                WHERE id = ?
            ''', (MessageStatus.DELIVERED.value, str(mid)))
            
            conn.commit()
            conn.close()
            
            self.stats["messages_delivered"] += 1
            
            if self.enable_logging and self.logger:
                self.logger.debug(f"Message {mid} delivery confirmed")
                
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error updating message delivery status: {e}")
    
    def queue_message(self, 
                     topic: str, 
                     payload: str, 
                     qos: int = 1, 
                     retain: bool = False,
                     message_id: Optional[str] = None) -> str:
        """
        Queue a message for reliable delivery.
        
        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of service level
            retain: Whether to retain the message
            message_id: Optional custom message ID
            
        Returns:
            Message ID
        """
        if message_id is None:
            message_id = f"{int(time.time() * 1000)}_{hash(payload) % 10000}"
        
        message = QueuedMessage(
            id=message_id,
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            timestamp=time.time(),
            status=MessageStatus.PENDING,
            retry_count=0,
            max_retries=self.max_retries,
            next_retry_time=time.time()
        )
        
        try:
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO messages 
                (id, topic, payload, qos, retain, timestamp, status, retry_count, max_retries, next_retry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.id, message.topic, message.payload, message.qos, message.retain,
                message.timestamp, message.status.value, message.retry_count, 
                message.max_retries, message.next_retry_time
            ))
            
            conn.commit()
            conn.close()
            
            # Add to in-memory queue
            self.message_queue.append(message)
            
            self.stats["messages_queued"] += 1
            self.stats["queue_size"] = len(self.message_queue)
            
            if self.enable_logging and self.logger:
                # Use debug level to reduce verbosity - only log when debugging
                self.logger.debug(f"Message queued: {message_id} -> {topic}")
            
            return message_id
            
        except Exception as e:
            error_msg = f"Failed to queue message: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _start_processing_thread(self):
        """Start the message processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
            self.processing_thread.start()
            self.processing = True
            
            if self.enable_logging and self.logger:
                self.logger.info("Message processing thread started")
    
    def _process_messages(self):
        """Main message processing loop."""
        while not self.stop_event.is_set():
            try:
                # Load pending messages from database
                self._load_pending_messages()
                
                # Process messages in batches
                if self.message_queue:
                    self._process_batch()
                
                # Sleep before next iteration
                time.sleep(0.1)
                
            except Exception as e:
                # Silently handle errors during shutdown
                if self.stop_event.is_set():
                    break
                if self.enable_logging and self.logger:
                    self.logger.error(f"Error in message processing loop: {e}")
                time.sleep(1.0)  # Wait longer on error
    
    def stop(self):
        """Stop the message queue processing thread gracefully."""
        if self.processing:
            self.stop_event.set()
            if self.processing_thread and self.processing_thread.is_alive():
                # Wait for thread to finish (with timeout)
                self.processing_thread.join(timeout=1.0)
            self.processing = False
            if self.enable_logging and self.logger:
                self.logger.info("Message queue processing thread stopped")
    
    def _load_pending_messages(self):
        """Load pending messages from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = time.time()
            cursor.execute('''
                SELECT id, topic, payload, qos, retain, timestamp, status, retry_count, max_retries, next_retry_time, error_message
                FROM messages 
                WHERE status IN (?, ?) AND next_retry_time <= ?
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (MessageStatus.PENDING.value, MessageStatus.RETRYING.value, current_time, self.batch_size * 2))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Add to queue if not already present
            existing_ids = {msg.id for msg in self.message_queue}
            for row in rows:
                message_id = row[0]
                if message_id not in existing_ids:
                    message = QueuedMessage(
                        id=message_id,
                        topic=row[1],
                        payload=row[2],
                        qos=row[3],
                        retain=bool(row[4]),
                        timestamp=row[5],
                        status=MessageStatus(row[6]),
                        retry_count=row[7],
                        max_retries=row[8],
                        next_retry_time=row[9],
                        error_message=row[10]
                    )
                    self.message_queue.append(message)
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error loading pending messages: {e}")
    
    def _process_batch(self):
        """Process a batch of messages."""
        if not self.mqtt_client or not self.mqtt_client.is_connected():
            if self.enable_logging and self.logger:
                self.logger.warning("MQTT client not connected - skipping message processing")
            return
        
        batch = []
        current_time = time.time()
        
        # Collect messages for batch processing
        while len(batch) < self.batch_size and self.message_queue:
            message = self.message_queue.popleft()
            
            # Skip if not ready for retry
            if message.next_retry_time > current_time:
                self.message_queue.append(message)  # Put back
                break
            
            batch.append(message)
        
        # Process batch
        for message in batch:
            try:
                self._send_message(message)
            except Exception as e:
                if self.enable_logging and self.logger:
                    self.logger.error(f"Error sending message {message.id}: {e}")
                self._handle_send_failure(message, str(e))
    
    def _send_message(self, message: QueuedMessage):
        """Send a single message via MQTT."""
        try:
            # Update status to retrying if this is a retry
            if message.retry_count > 0:
                self._update_message_status(message.id, MessageStatus.RETRYING)
                self.stats["messages_retried"] += 1
            
            # Publish message
            result = self.mqtt_client.publish(
                message.topic, 
                message.payload, 
                qos=message.qos, 
                retain=message.retain
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                # Update status to sent
                self._update_message_status(message.id, MessageStatus.SENT)
                self.stats["messages_sent"] += 1
                
                if self.enable_logging and self.logger:
                    self.logger.debug(f"Message sent: {message.id} -> {message.topic}")
            else:
                # Handle publish failure
                error_msg = f"Publish failed with code: {result.rc}"
                self._handle_send_failure(message, error_msg)
                
        except Exception as e:
            error_msg = f"Send error: {str(e)}"
            self._handle_send_failure(message, error_msg)
    
    def _handle_send_failure(self, message: QueuedMessage, error_message: str):
        """Handle message send failure with retry logic."""
        message.retry_count += 1
        
        if message.retry_count >= message.max_retries:
            # Move to failed status
            self._update_message_status(message.id, MessageStatus.FAILED, error_message)
            self.stats["messages_failed"] += 1
            
            if self.enable_logging and self.logger:
                self.logger.error(f"Message permanently failed: {message.id} - {error_message}")
        else:
            # Schedule retry with exponential backoff
            delay = min(self.retry_delay * (2 ** (message.retry_count - 1)), self.max_retry_delay)
            message.next_retry_time = time.time() + delay
            
            self._update_message_status(message.id, MessageStatus.PENDING, error_message)
            
            # Put back in queue for retry
            self.message_queue.append(message)
            
            if self.enable_logging and self.logger:
                self.logger.warning(f"Message {message.id} failed, retrying in {delay:.1f}s (attempt {message.retry_count}/{message.max_retries})")
    
    def _update_message_status(self, message_id: str, status: MessageStatus, error_message: Optional[str] = None):
        """Update message status in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE messages 
                SET status = ?, error_message = ?
                WHERE id = ?
            ''', (status.value, error_message, message_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error updating message status: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get status counts
            cursor.execute('''
                SELECT status, COUNT(*) 
                FROM messages 
                GROUP BY status
            ''')
            
            status_counts = dict(cursor.fetchall())
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM messages')
            total_messages = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_messages": total_messages,
                "status_counts": status_counts,
                "queue_size": len(self.message_queue),
                "processing": self.processing,
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error getting queue stats: {e}")
            return {"error": str(e)}
    
    def clear_delivered_messages(self, older_than_hours: int = 24):
        """Clear delivered messages older than specified hours."""
        try:
            cutoff_time = time.time() - (older_than_hours * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM messages 
                WHERE status = ? AND timestamp < ?
            ''', (MessageStatus.DELIVERED.value, cutoff_time))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Cleared {deleted_count} delivered messages older than {older_than_hours} hours")
            
            return deleted_count
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error clearing delivered messages: {e}")
            return 0
    
    def stop_processing(self):
        """Stop message processing."""
        self.stop_event.set()
        self.processing = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        if self.enable_logging and self.logger:
            self.logger.info("Message processing stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_processing()


# Convenience function for integration
def create_mqtt_message_queue(db_path: str = "mqtt_message_queue.db",
                             max_retries: int = 5,
                             enable_logging: bool = True) -> MQTTMessageQueue:
    """
    Create an MQTT message queue with default settings.
    
    Args:
        db_path: Path to SQLite database for persistent storage
        max_retries: Maximum number of retry attempts
        enable_logging: Whether to enable detailed logging
        
    Returns:
        Configured MQTTMessageQueue instance
    """
    return MQTTMessageQueue(
        db_path=db_path,
        max_retries=max_retries,
        enable_logging=enable_logging
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create message queue
    queue = create_mqtt_message_queue()
    
    # Test queuing messages
    print("Testing MQTT message queue...")
    
    # Queue some test messages
    message_ids = []
    for i in range(5):
        msg_id = queue.queue_message(
            topic=f"test/topic/{i}",
            payload=f"Test message {i}",
            qos=1
        )
        message_ids.append(msg_id)
        print(f"Queued message: {msg_id}")
    
    # Get stats
    stats = queue.get_queue_stats()
    print(f"\nQueue stats: {stats}")
    
    # Stop processing
    queue.stop_processing()
    print("Message queue test completed")
