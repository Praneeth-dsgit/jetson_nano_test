# SQLite Database Viewer Guide

This guide shows you multiple ways to view and inspect SQLite databases in your project.

## 📊 Database Files in Project

Your project uses two main SQLite databases:

1. **`system_health.db`** - Stores system metrics and health alerts
   - Tables: `system_metrics`, `health_alerts`

2. **`mqtt_message_queue.db`** - Stores MQTT message queue
   - Tables: `messages`

## 🐍 Method 1: Python Script (Recommended)

Use the provided `view_db.py` script:

```bash
# View a specific database
python view_db.py system_health.db

# View with row limit
python view_db.py system_health.db --limit 10

# View all databases
python view_db.py --all

# Find all database files
python view_db.py --list
```

## 💻 Method 2: SQLite Command Line Tool

### On Windows:
```bash
# Open SQLite command line
sqlite3 system_health.db

# Or if sqlite3 is not in PATH:
# Download SQLite from https://www.sqlite.org/download.html
```

### On Linux/Mac:
```bash
# Install if needed
sudo apt install sqlite3  # Ubuntu/Debian
brew install sqlite3      # macOS

# Open database
sqlite3 system_health.db
```

### Useful SQLite Commands:

```sql
-- List all tables
.tables

-- Show table structure
.schema system_metrics

-- View all data (limit rows)
SELECT * FROM system_metrics LIMIT 10;

-- Count rows
SELECT COUNT(*) FROM system_metrics;

-- View recent metrics
SELECT * FROM system_metrics 
ORDER BY timestamp DESC 
LIMIT 20;

-- View alerts
SELECT * FROM health_alerts 
WHERE resolved = 0 
ORDER BY timestamp DESC;

-- View message queue stats
SELECT status, COUNT(*) 
FROM messages 
GROUP BY status;

-- Exit SQLite
.quit
```

## 🖥️ Method 3: GUI Tools

### DB Browser for SQLite (Free, Cross-platform)
1. Download from: https://sqlitebrowser.org/
2. Open database file
3. Browse tables, run queries, view data

### VS Code Extension
1. Install "SQLite Viewer" extension
2. Right-click `.db` file → "Open Database"
3. View tables and data in VS Code

### DBeaver (Free, Cross-platform)
1. Download from: https://dbeaver.io/
2. Create new SQLite connection
3. Browse database visually

## 📝 Quick Reference Queries

### System Health Database

```sql
-- View all metrics
SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 50;

-- View by metric type
SELECT * FROM system_metrics 
WHERE metric_name = 'cpu_percent' 
ORDER BY timestamp DESC;

-- View warnings and critical alerts
SELECT * FROM system_metrics 
WHERE status IN ('warning', 'critical')
ORDER BY timestamp DESC;

-- View unresolved alerts
SELECT * FROM health_alerts 
WHERE resolved = 0 
ORDER BY timestamp DESC;

-- Metric statistics
SELECT 
    metric_name,
    COUNT(*) as count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value
FROM system_metrics
GROUP BY metric_name;
```

### MQTT Message Queue Database

```sql
-- View pending messages
SELECT * FROM messages 
WHERE status = 'pending' 
ORDER BY timestamp;

-- View failed messages
SELECT * FROM messages 
WHERE status = 'failed' 
ORDER BY timestamp DESC;

-- Message statistics
SELECT 
    status,
    COUNT(*) as count,
    AVG(retry_count) as avg_retries
FROM messages
GROUP BY status;

-- View recent messages
SELECT id, topic, status, retry_count, timestamp
FROM messages
ORDER BY timestamp DESC
LIMIT 20;
```

## 🔍 Method 4: Python Interactive Shell

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('system_health.db')

# View as DataFrame (requires pandas)
df = pd.read_sql_query("SELECT * FROM system_metrics LIMIT 10", conn)
print(df)

# Or use cursor
cursor = conn.cursor()
cursor.execute("SELECT * FROM system_metrics LIMIT 10")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()
```

## 📍 Common Database Locations

- `system_health.db` - Root directory or `core/` directory
- `mqtt_message_queue.db` - Root directory or `core/` directory
- These are created automatically when the system runs

## 🛠️ Troubleshooting

### Database is locked
- Close any other programs accessing the database
- Wait for the application to finish writing

### Database not found
- Use `python view_db.py --list` to find all databases
- Check if the application has run yet (databases are created on first run)

### Permission denied
- Check file permissions
- On Windows, ensure no other process has the file open

