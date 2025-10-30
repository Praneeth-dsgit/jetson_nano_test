"""
Database modules with connection pooling and reliability.

This package contains database functionality:
- db: Enhanced database class with connection pooling
"""

from .db import Database, DatabaseConnectionPool, create_database_with_pool

__version__ = "1.0.0"

__all__ = [
    'Database',
    'DatabaseConnectionPool', 
    'create_database_with_pool'
]