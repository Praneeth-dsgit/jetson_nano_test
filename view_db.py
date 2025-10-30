#!/usr/bin/env python3
"""
SQLite Database Viewer for Jetson Nano ML System

This script provides an easy way to view contents of SQLite databases used in the project:
- system_health.db (system metrics and alerts)
- mqtt_message_queue.db (MQTT message queue)

Usage:
    python view_db.py system_health.db
    python view_db.py mqtt_message_queue.db
    python view_db.py --all  # View all databases
"""

import sqlite3
import sys
import os
from datetime import datetime
from typing import Optional, List, Dict
import argparse


def print_table_structure(cursor: sqlite3.Cursor, table_name: str) -> None:
    """Print the structure of a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    print(f"\n{'='*80}")
    print(f"Table: {table_name}")
    print(f"{'='*80}")
    print(f"{'Column':<20} {'Type':<15} {'Not Null':<10} {'Default':<15} {'Primary Key':<12}")
    print("-" * 80)
    
    for col in columns:
        cid, name, col_type, not_null, default_val, pk = col
        print(f"{name:<20} {col_type:<15} {'Yes' if not_null else 'No':<10} "
              f"{str(default_val) if default_val else 'None':<15} {'Yes' if pk else 'No':<12}")
    print()


def print_table_data(cursor: sqlite3.Cursor, table_name: str, limit: Optional[int] = None) -> None:
    """Print data from a table."""
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]
    
    print(f"\n{'='*80}")
    print(f"Table: {table_name} ({total_rows} total rows)")
    print(f"{'='*80}")
    
    if total_rows == 0:
        print("(No data in table)")
        return
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Build query
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Print header
    print(" | ".join(f"{col:<15}" for col in columns))
    print("-" * 80)
    
    # Print rows
    for row in rows:
        formatted_row = []
        for i, val in enumerate(row):
            if val is None:
                formatted_row.append("NULL")
            elif isinstance(val, (int, float)):
                formatted_row.append(str(val))
            elif isinstance(val, str):
                # Truncate long strings
                if len(val) > 50:
                    formatted_row.append(val[:47] + "...")
                else:
                    formatted_row.append(val)
            else:
                formatted_row.append(str(val))
        print(" | ".join(f"{val:<15}" for val in formatted_row))
    
    if limit and total_rows > limit:
        print(f"\n... ({total_rows - limit} more rows not shown)")
    print()


def view_system_health_db(db_path: str = "system_health.db", limit: Optional[int] = None) -> None:
    """View system health database contents."""
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    print(f"\n{'#'*80}")
    print(f"SYSTEM HEALTH DATABASE: {db_path}")
    print(f"{'#'*80}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    if not tables:
        print("No tables found in database")
        conn.close()
        return
    
    for table in tables:
        print_table_structure(cursor, table)
        print_table_data(cursor, table, limit)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    if 'system_metrics' in tables:
        cursor.execute("SELECT COUNT(*) FROM system_metrics")
        metrics_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT metric_name) FROM system_metrics")
        unique_metrics = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM system_metrics WHERE status = 'warning'")
        warnings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM system_metrics WHERE status = 'critical'")
        criticals = cursor.fetchone()[0]
        
        print(f"Total Metrics: {metrics_count}")
        print(f"Unique Metric Types: {unique_metrics}")
        print(f"Warnings: {warnings}")
        print(f"Critical Alerts: {criticals}")
    
    if 'health_alerts' in tables:
        cursor.execute("SELECT COUNT(*) FROM health_alerts")
        alerts_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM health_alerts WHERE resolved = 0")
        unresolved = cursor.fetchone()[0]
        
        print(f"\nTotal Alerts: {alerts_count}")
        print(f"Unresolved Alerts: {unresolved}")
    
    conn.close()


def view_mqtt_queue_db(db_path: str = "mqtt_message_queue.db", limit: Optional[int] = None) -> None:
    """View MQTT message queue database contents."""
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    print(f"\n{'#'*80}")
    print(f"MQTT MESSAGE QUEUE DATABASE: {db_path}")
    print(f"{'#'*80}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    if not tables:
        print("No tables found in database")
        conn.close()
        return
    
    for table in tables:
        print_table_structure(cursor, table)
        print_table_data(cursor, table, limit)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    if 'messages' in tables:
        cursor.execute("SELECT COUNT(*) FROM messages")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE status = 'pending'")
        pending = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE status = 'delivered'")
        delivered = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE status = 'failed'")
        failed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages WHERE status = 'retrying'")
        retrying = cursor.fetchone()[0]
        
        print(f"Total Messages: {total}")
        print(f"Pending: {pending}")
        print(f"Delivered: {delivered}")
        print(f"Failed: {failed}")
        print(f"Retrying: {retrying}")
    
    conn.close()


def list_all_databases() -> List[str]:
    """Find all SQLite databases in the project."""
    db_files = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment and cache directories
        if 'env' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))
    return db_files


def main():
    parser = argparse.ArgumentParser(
        description='View SQLite database contents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_db.py system_health.db
  python view_db.py mqtt_message_queue.db
  python view_db.py core/system_health.db --limit 10
  python view_db.py --all
  python view_db.py --list
        """
    )
    
    parser.add_argument(
        'database',
        nargs='?',
        help='Path to SQLite database file (e.g., system_health.db)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='View all databases found in the project'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all database files found in the project'
    )
    
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=50,
        help='Limit number of rows to display (default: 50)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nSearching for SQLite databases...")
        db_files = list_all_databases()
        if db_files:
            print("\nFound databases:")
            for db in db_files:
                size = os.path.getsize(db) if os.path.exists(db) else 0
                print(f"  {db} ({size:,} bytes)")
        else:
            print("No database files found.")
        return
    
    if args.all:
        db_files = list_all_databases()
        if not db_files:
            print("No database files found.")
            return
        
        for db in db_files:
            if 'system_health' in db.lower():
                view_system_health_db(db, args.limit)
            elif 'mqtt' in db.lower() or 'message' in db.lower():
                view_mqtt_queue_db(db, args.limit)
            else:
                print(f"\nUnknown database type: {db}")
                # Try to view it generically
                try:
                    conn = sqlite3.connect(db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    for table in tables:
                        print_table_structure(cursor, table)
                        print_table_data(cursor, table, args.limit)
                    conn.close()
                except Exception as e:
                    print(f"Error viewing {db}: {e}")
        return
    
    if not args.database:
        parser.print_help()
        print("\n💡 Tip: Use --list to find all database files")
        return
    
    db_path = args.database
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("\n💡 Tip: Use --list to find all database files")
        return
    
    # Determine database type and view accordingly
    if 'system_health' in db_path.lower():
        view_system_health_db(db_path, args.limit)
    elif 'mqtt' in db_path.lower() or 'message' in db_path.lower():
        view_mqtt_queue_db(db_path, args.limit)
    else:
        # Generic viewer
        print(f"\nViewing database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            print("No tables found in database")
        else:
            for table in tables:
                print_table_structure(cursor, table)
                print_table_data(cursor, table, args.limit)
        
        conn.close()


if __name__ == "__main__":
    main()

