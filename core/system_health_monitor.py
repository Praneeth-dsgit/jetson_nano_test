#!/usr/bin/env python3
"""
System Health Monitor for Athlete Monitoring System

This module provides comprehensive system health monitoring including:
- Real-time system metrics collection
- Health status dashboards
- Performance monitoring and alerting
- System resource utilization tracking
- Service availability monitoring
- Historical data collection and analysis
"""

import os
import json
import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import sqlite3
import numpy as np

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class SystemMetric:
    """Represents a system metric measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class HealthAlert:
    """Represents a health alert."""
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class SystemHealthMonitor:
    """
    Comprehensive system health monitoring and alerting system.
    
    Features:
    - Real-time system metrics collection
    - Configurable thresholds and alerting
    - Historical data storage and analysis
    - Health status dashboards
    - Performance trend analysis
    - Service availability monitoring
    """
    
    def __init__(self, 
                 db_path: str = "system_health.db",
                 collection_interval: int = 30,
                 history_retention_days: int = 30,
                 enable_logging: bool = True):
        """
        Initialize the system health monitor.
        
        Args:
            db_path: Path to SQLite database for storing metrics
            collection_interval: Metrics collection interval in seconds
            history_retention_days: Number of days to retain historical data
            enable_logging: Whether to enable detailed logging
        """
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.history_retention_days = history_retention_days
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Initialize database
        self._init_database()
        
        # Metrics collection
        self.metrics_buffer = deque(maxlen=1000)
        self.alerts_buffer = deque(maxlen=100)
        self.collection_thread = None
        self.running = False
        
        # Thresholds for different metrics
        self.thresholds = {
            'cpu_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_percent': {'warning': 80.0, 'critical': 95.0},
            'disk_percent': {'warning': 85.0, 'critical': 95.0},
            'gpu_memory_percent': {'warning': 80.0, 'critical': 95.0},
            'temperature': {'warning': 70.0, 'critical': 85.0},
            'load_average': {'warning': 2.0, 'critical': 4.0},
            'network_errors': {'warning': 10, 'critical': 50},
            'process_count': {'warning': 200, 'critical': 300}
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Statistics
        self.stats = {
            "metrics_collected": 0,
            "alerts_generated": 0,
            "collection_errors": 0,
            "start_time": datetime.now()
        }
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    status TEXT NOT NULL,
                    threshold_warning REAL,
                    threshold_critical REAL,
                    metadata TEXT
                )
            ''')
            
            # Create indexes separately
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON system_metrics (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON system_metrics (metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON system_metrics (status)')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP
                )
            ''')
            
            # Create indexes for alerts table separately
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON health_alerts (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON health_alerts (severity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON health_alerts (resolved)')
            
            conn.commit()
            conn.close()
            
            if self.enable_logging and self.logger:
                self.logger.info(f"System health database initialized: {self.db_path}")
                
        except Exception as e:
            error_msg = f"Failed to initialize system health database: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def start_monitoring(self):
        """Start the system health monitoring."""
        if self.running:
            if self.enable_logging and self.logger:
                self.logger.warning("System health monitoring already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        if self.enable_logging and self.logger:
            self.logger.info(f"System health monitoring started (interval: {self.collection_interval}s)")
    
    def stop_monitoring(self):
        """Stop the system health monitoring."""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        if self.enable_logging and self.logger:
            self.logger.info("System health monitoring stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.stats["collection_errors"] += 1
                if self.enable_logging and self.logger:
                    self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_system_metrics(self):
        """Collect all system metrics."""
        timestamp = datetime.now()
        
        # CPU metrics
        self._collect_cpu_metrics(timestamp)
        
        # Memory metrics
        self._collect_memory_metrics(timestamp)
        
        # Disk metrics
        self._collect_disk_metrics(timestamp)
        
        # Network metrics
        self._collect_network_metrics(timestamp)
        
        # Process metrics
        self._collect_process_metrics(timestamp)
        
        # GPU metrics (if available)
        self._collect_gpu_metrics(timestamp)
        
        # System load
        self._collect_load_metrics(timestamp)
        
        # Temperature (if available)
        self._collect_temperature_metrics(timestamp)
        
        # Store metrics in database
        self._store_metrics_batch()
        
        self.stats["metrics_collected"] += len(self.metrics_buffer)
    
    def _collect_cpu_metrics(self, timestamp: datetime):
        """Collect CPU-related metrics."""
        try:
            # CPU percentage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric(timestamp, 'cpu_percent', cpu_percent, '%', 
                           self.thresholds['cpu_percent'])
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self._add_metric(timestamp, 'cpu_frequency', cpu_freq.current, 'MHz')
            
            # CPU count
            self._add_metric(timestamp, 'cpu_count', psutil.cpu_count(), 'cores')
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error collecting CPU metrics: {e}")
    
    def _collect_memory_metrics(self, timestamp: datetime):
        """Collect memory-related metrics."""
        try:
            # Virtual memory
            virtual_mem = psutil.virtual_memory()
            self._add_metric(timestamp, 'memory_percent', virtual_mem.percent, '%',
                           self.thresholds['memory_percent'])
            self._add_metric(timestamp, 'memory_available', virtual_mem.available / (1024**3), 'GB')
            self._add_metric(timestamp, 'memory_used', virtual_mem.used / (1024**3), 'GB')
            self._add_metric(timestamp, 'memory_total', virtual_mem.total / (1024**3), 'GB')
            
            # Swap memory
            swap_mem = psutil.swap_memory()
            self._add_metric(timestamp, 'swap_percent', swap_mem.percent, '%')
            self._add_metric(timestamp, 'swap_used', swap_mem.used / (1024**3), 'GB')
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error collecting memory metrics: {e}")
    
    def _collect_disk_metrics(self, timestamp: datetime):
        """Collect disk-related metrics."""
        try:
            # Disk usage for all partitions
            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    usage_percent = (partition_usage.used / partition_usage.total) * 100
                    
                    self._add_metric(timestamp, 'disk_percent', usage_percent, '%',
                                   self.thresholds['disk_percent'],
                                   metadata={'mountpoint': partition.mountpoint})
                    
                    self._add_metric(timestamp, 'disk_free', partition_usage.free / (1024**3), 'GB',
                                   metadata={'mountpoint': partition.mountpoint})
                    
                except PermissionError:
                    continue  # Skip partitions we can't access
                    
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error collecting disk metrics: {e}")
    
    def _collect_network_metrics(self, timestamp: datetime):
        """Collect network-related metrics."""
        try:
            # Network I/O
            net_io = psutil.net_io_counters()
            self._add_metric(timestamp, 'network_bytes_sent', net_io.bytes_sent / (1024**2), 'MB')
            self._add_metric(timestamp, 'network_bytes_recv', net_io.bytes_recv / (1024**2), 'MB')
            self._add_metric(timestamp, 'network_packets_sent', net_io.packets_sent, 'packets')
            self._add_metric(timestamp, 'network_packets_recv', net_io.packets_recv, 'packets')
            
            # Network errors
            network_errors = net_io.errin + net_io.errout
            self._add_metric(timestamp, 'network_errors', network_errors, 'errors',
                           self.thresholds['network_errors'])
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error collecting network metrics: {e}")
    
    def _collect_process_metrics(self, timestamp: datetime):
        """Collect process-related metrics."""
        try:
            # Process count
            process_count = len(psutil.pids())
            self._add_metric(timestamp, 'process_count', process_count, 'processes',
                           self.thresholds['process_count'])
            
            # Top processes by CPU usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage and get top 5
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            top_processes = processes[:5]
            
            self._add_metric(timestamp, 'top_cpu_process', 
                           top_processes[0]['cpu_percent'] if top_processes else 0, '%',
                           metadata={'processes': top_processes})
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error collecting process metrics: {e}")
    
    def _collect_gpu_metrics(self, timestamp: datetime):
        """Collect GPU-related metrics (if available)."""
        try:
            # Try to get GPU metrics using nvidia-ml-py or similar
            # This is a placeholder - actual implementation would depend on available GPU monitoring tools
            import subprocess
            
            # Try nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            gpu_util = float(parts[0])
                            gpu_mem_used = float(parts[1])
                            gpu_mem_total = float(parts[2])
                            gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
                            
                            self._add_metric(timestamp, 'gpu_utilization', gpu_util, '%',
                                           metadata={'gpu_id': i})
                            self._add_metric(timestamp, 'gpu_memory_percent', gpu_mem_percent, '%',
                                           self.thresholds['gpu_memory_percent'],
                                           metadata={'gpu_id': i})
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # nvidia-smi not available
                
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.debug(f"GPU metrics not available: {e}")
    
    def _collect_load_metrics(self, timestamp: datetime):
        """Collect system load metrics."""
        try:
            # Load average (Unix-like systems)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                self._add_metric(timestamp, 'load_average_1m', load_avg[0], 'load',
                               self.thresholds['load_average'])
                self._add_metric(timestamp, 'load_average_5m', load_avg[1], 'load')
                self._add_metric(timestamp, 'load_average_15m', load_avg[2], 'load')
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.debug(f"Load metrics not available: {e}")
    
    def _collect_temperature_metrics(self, timestamp: datetime):
        """Collect temperature metrics (if available)."""
        try:
            # Try to get temperature using psutil (if available)
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    for entry in entries:
                        self._add_metric(timestamp, 'temperature', entry.current, '°C',
                                       self.thresholds['temperature'],
                                       metadata={'sensor': name, 'label': entry.label})
                        
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.debug(f"Temperature metrics not available: {e}")
    
    def _add_metric(self, timestamp: datetime, name: str, value: float, unit: str,
                   thresholds: Optional[Dict[str, float]] = None, metadata: Optional[Dict] = None):
        """Add a metric to the buffer."""
        status = HealthStatus.HEALTHY
        
        if thresholds:
            if value >= thresholds.get('critical', float('inf')):
                status = HealthStatus.CRITICAL
            elif value >= thresholds.get('warning', float('inf')):
                status = HealthStatus.WARNING
        
        metric = SystemMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            status=status,
            threshold_warning=thresholds.get('warning') if thresholds else None,
            threshold_critical=thresholds.get('critical') if thresholds else None,
            metadata=metadata
        )
        
        self.metrics_buffer.append(metric)
        
        # Check for alerts
        if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            self._check_and_generate_alert(metric)
    
    def _check_and_generate_alert(self, metric: SystemMetric):
        """Check if metric should generate an alert."""
        try:
            # Check if we already have an unresolved alert for this metric
            existing_alert = None
            for alert in self.alerts_buffer:
                if (alert.metric_name == metric.metric_name and 
                    not alert.resolved and 
                    alert.severity == metric.status.value):
                    existing_alert = alert
                    break
            
            if not existing_alert:
                # Generate new alert
                alert = HealthAlert(
                    timestamp=metric.timestamp,
                    alert_type=f"{metric.metric_name}_{metric.status.value}",
                    severity=metric.status.value,
                    message=f"{metric.metric_name} is {metric.status.value}: {metric.value:.2f} {metric.unit}",
                    metric_name=metric.metric_name,
                    current_value=metric.value,
                    threshold_value=metric.threshold_critical if metric.status == HealthStatus.CRITICAL else metric.threshold_warning
                )
                
                self.alerts_buffer.append(alert)
                self.stats["alerts_generated"] += 1
                
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        if self.enable_logging and self.logger:
                            self.logger.error(f"Error in alert callback: {e}")
                
                if self.enable_logging and self.logger:
                    self.logger.warning(f"Health alert: {alert.message}")
        
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error generating alert: {e}")
    
    def _store_metrics_batch(self):
        """Store metrics from buffer to database."""
        if not self.metrics_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric in list(self.metrics_buffer):
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, metric_name, value, unit, status, threshold_warning, threshold_critical, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    metric.status.value,
                    metric.threshold_warning,
                    metric.threshold_critical,
                    json.dumps(metric.metadata) if metric.metadata else None
                ))
            
            conn.commit()
            conn.close()
            
            # Clear buffer after storing
            self.metrics_buffer.clear()
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error storing metrics: {e}")
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        try:
            # Get latest metrics for each type
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT metric_name, value, unit, status, timestamp
                FROM system_metrics 
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY timestamp DESC
            ''')
            
            recent_metrics = cursor.fetchall()
            conn.close()
            
            # Group by metric name and get latest
            latest_metrics = {}
            for row in recent_metrics:
                metric_name = row[0]
                if metric_name not in latest_metrics:
                    latest_metrics[metric_name] = {
                        'value': row[1],
                        'unit': row[2],
                        'status': row[3],
                        'timestamp': row[4]
                    }
            
            # Determine overall health status
            overall_status = HealthStatus.HEALTHY
            critical_count = 0
            warning_count = 0
            
            for metric in latest_metrics.values():
                if metric['status'] == 'critical':
                    critical_count += 1
                    overall_status = HealthStatus.CRITICAL
                elif metric['status'] == 'warning' and overall_status != HealthStatus.CRITICAL:
                    warning_count += 1
                    overall_status = HealthStatus.WARNING
            
            return {
                'overall_status': overall_status.value,
                'critical_metrics': critical_count,
                'warning_metrics': warning_count,
                'healthy_metrics': len(latest_metrics) - critical_count - warning_count,
                'latest_metrics': latest_metrics,
                'monitoring_stats': self.stats.copy(),
                'uptime_hours': (datetime.now() - self.stats['start_time']).total_seconds() / 3600
            }
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error getting health status: {e}")
            return {'error': str(e)}
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, value, unit, status, metadata
                FROM system_metrics 
                WHERE metric_name = ? AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp ASC
            '''.format(hours), (metric_name,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'timestamp': row[0],
                    'value': row[1],
                    'unit': row[2],
                    'status': row[3],
                    'metadata': json.loads(row[4]) if row[4] else None
                }
                for row in results
            ]
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error getting metric history: {e}")
            return []
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Add a callback function for health alerts."""
        self.alert_callbacks.append(callback)
    
    def cleanup_old_data(self):
        """Clean up old metrics and alerts data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean up old metrics
            cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_date.isoformat(),))
            metrics_deleted = cursor.rowcount
            
            # Clean up old resolved alerts
            cursor.execute('DELETE FROM health_alerts WHERE resolved = 1 AND timestamp < ?', (cutoff_date.isoformat(),))
            alerts_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Cleaned up {metrics_deleted} old metrics and {alerts_deleted} old alerts")
            
            return {'metrics_deleted': metrics_deleted, 'alerts_deleted': alerts_deleted}
            
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error cleaning up old data: {e}")
            return {'error': str(e)}


# Convenience function for integration
def create_system_health_monitor(collection_interval: int = 30,
                                history_retention_days: int = 30,
                                enable_logging: bool = True) -> SystemHealthMonitor:
    """
    Create a system health monitor with default settings.
    
    Args:
        collection_interval: Metrics collection interval in seconds
        history_retention_days: Number of days to retain historical data
        enable_logging: Whether to enable detailed logging
        
    Returns:
        Configured SystemHealthMonitor instance
    """
    return SystemHealthMonitor(
        collection_interval=collection_interval,
        history_retention_days=history_retention_days,
        enable_logging=enable_logging
    )


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create health monitor
    monitor = create_system_health_monitor(collection_interval=10)
    
    # Add alert callback
    def alert_callback(alert):
        print(f"🚨 ALERT: {alert.message}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Start monitoring
    print("Starting system health monitoring...")
    monitor.start_monitoring()
    
    try:
        # Monitor for 2 minutes
        time.sleep(120)
        
        # Get health status
        health = monitor.get_current_health_status()
        print(f"\nSystem Health Status: {health['overall_status']}")
        print(f"Critical metrics: {health['critical_metrics']}")
        print(f"Warning metrics: {health['warning_metrics']}")
        print(f"Healthy metrics: {health['healthy_metrics']}")
        
        # Get CPU history
        cpu_history = monitor.get_metric_history('cpu_percent', hours=1)
        if cpu_history:
            avg_cpu = np.mean([m['value'] for m in cpu_history])
            print(f"Average CPU usage (last hour): {avg_cpu:.1f}%")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        print("System health monitoring stopped")
