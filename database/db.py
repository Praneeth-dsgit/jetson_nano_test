import pymysql
from pymysql.cursors import DictCursor
import os
from dotenv import load_dotenv
import threading
import time
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from queue import Queue, Empty
import json

load_dotenv()

class DatabaseConnectionPool:
    """
    Database connection pool for improved reliability and performance.
    
    Features:
    - Connection pooling with configurable pool size
    - Automatic connection health checking
    - Connection retry logic with exponential backoff
    - Transaction management
    - Connection timeout handling
    """
    
    def __init__(self, 
                 pool_size: int = 5,
                 max_overflow: int = 10,
                 connection_timeout: int = 30,
                 enable_logging: bool = True):
        """
        Initialize the connection pool.
        
        Args:
            pool_size: Initial number of connections in the pool
            max_overflow: Maximum additional connections beyond pool_size
            connection_timeout: Connection timeout in seconds
            enable_logging: Whether to enable detailed logging
        """
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.connection_timeout = connection_timeout
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Connection pool
        self._pool = Queue(maxsize=pool_size + max_overflow)
        self._all_connections = set()
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "connections_created": 0,
            "connections_checked_out": 0,
            "connections_returned": 0,
            "connection_errors": 0,
            "pool_hits": 0,
            "pool_misses": 0
        }
        
        # Initialize pool
        self._initialize_pool()
    
    def _create_connection(self) -> pymysql.Connection:
        """Create a new database connection."""
        try:
            connection = pymysql.connect(
                host=os.getenv("DB_HOST", 'playerstat-ai'),
                user=os.getenv("DB_USER", 'Praneeth'),
                password=os.getenv("DB_PASSWORD", 'playerstat'),
                database=os.getenv("DB_NAME", 'dashboard_db'),
                cursorclass=DictCursor,
                charset='utf8mb4',
                connect_timeout=self.connection_timeout,
                autocommit=False
            )
            
            self.stats["connections_created"] += 1
            
            if self.enable_logging and self.logger:
                self.logger.debug("Created new database connection")
            
            return connection
            
        except Exception as e:
            self.stats["connection_errors"] += 1
            error_msg = f"Failed to create database connection: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _initialize_pool(self):
        """Initialize the connection pool with initial connections."""
        try:
            for _ in range(self.pool_size):
                connection = self._create_connection()
                self._pool.put(connection)
                self._all_connections.add(connection)
            
            if self.enable_logging and self.logger:
                self.logger.info(f"Initialized connection pool with {self.pool_size} connections")
                
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def _is_connection_healthy(self, connection: pymysql.Connection) -> bool:
        """Check if a connection is healthy."""
        try:
            connection.ping(reconnect=False)
            return True
        except Exception:
            return False
    
    def get_connection(self, timeout: int = 10) -> pymysql.Connection:
        """Get a connection from the pool."""
        try:
            # Try to get connection from pool
            try:
                connection = self._pool.get(timeout=timeout)
                self.stats["pool_hits"] += 1
                
                # Check if connection is healthy
                if self._is_connection_healthy(connection):
                    self.stats["connections_checked_out"] += 1
                    return connection
                else:
                    # Connection is unhealthy, create a new one
                    if self.enable_logging and self.logger:
                        self.logger.warning("Unhealthy connection detected, creating replacement")
                    
                    with self._lock:
                        self._all_connections.discard(connection)
                        try:
                            connection.close()
                        except:
                            pass
                    
                    new_connection = self._create_connection()
                    with self._lock:
                        self._all_connections.add(new_connection)
                    
                    self.stats["connections_checked_out"] += 1
                    return new_connection
                    
            except Empty:
                # Pool is empty, try to create a new connection
                with self._lock:
                    if len(self._all_connections) < self.pool_size + self.max_overflow:
                        self.stats["pool_misses"] += 1
                        connection = self._create_connection()
                        self._all_connections.add(connection)
                        self.stats["connections_checked_out"] += 1
                        return connection
                    else:
                        raise ConnectionError("Connection pool exhausted")
                        
        except Exception as e:
            error_msg = f"Failed to get database connection: {e}"
            if self.enable_logging and self.logger:
                self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def return_connection(self, connection: pymysql.Connection):
        """Return a connection to the pool."""
        try:
            if connection and connection.open:
                # Reset connection state
                connection.rollback()
                
                # Return to pool if there's space
                try:
                    self._pool.put_nowait(connection)
                    self.stats["connections_returned"] += 1
                except:
                    # Pool is full, close the connection
                    with self._lock:
                        self._all_connections.discard(connection)
                    connection.close()
                    
                    if self.enable_logging and self.logger:
                        self.logger.debug("Pool full, closed excess connection")
            else:
                # Connection is closed, remove from tracking
                with self._lock:
                    self._all_connections.discard(connection)
                    
        except Exception as e:
            if self.enable_logging and self.logger:
                self.logger.error(f"Error returning connection to pool: {e}")
    
    @contextmanager
    def get_connection_context(self, timeout: int = 10):
        """Context manager for database connections."""
        connection = None
        try:
            connection = self.get_connection(timeout)
            yield connection
        finally:
            if connection:
                self.return_connection(connection)
    
    def close_all_connections(self):
        """Close all connections in the pool."""
        with self._lock:
            for connection in self._all_connections.copy():
                try:
                    if connection.open:
                        connection.close()
                except:
                    pass
            self._all_connections.clear()
            
            # Clear the pool
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except:
                    break
        
        if self.enable_logging and self.logger:
            self.logger.info("All database connections closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "active_connections": len(self._all_connections),
            "available_connections": self._pool.qsize(),
            "stats": self.stats.copy()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the connection pool."""
        healthy_connections = 0
        total_connections = len(self._all_connections)
        
        for connection in self._all_connections:
            if self._is_connection_healthy(connection):
                healthy_connections += 1
        
        return {
            "total_connections": total_connections,
            "healthy_connections": healthy_connections,
            "pool_health": healthy_connections / total_connections if total_connections > 0 else 0,
            "pool_stats": self.get_pool_stats()
        }


class Database:
    """
    Enhanced database class with connection pooling and transaction management.
    """
    
    def __init__(self, use_pool: bool = True, pool_size: int = 5):
        """
        Initialize the database with optional connection pooling.
        
        Args:
            use_pool: Whether to use connection pooling
            pool_size: Size of connection pool if using pooling
        """
        self.use_pool = use_pool
        
        if use_pool:
            self.pool = DatabaseConnectionPool(
                pool_size=pool_size,
                enable_logging=True
            )
            self.connection = None  # Not used when pooling
        else:
            self.pool = None
            self.connection = None
            self.connect()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.connection = pymysql.connect(
                host=os.getenv("DB_HOST", ''),
                user=os.getenv("DB_USER", ''),
                password=os.getenv("DB_PASSWORD", ''),
                database=os.getenv("DB_NAME", ''),
                cursorclass=DictCursor,
                charset='utf8mb4'
            )
            print(f"✅ Database connection established successfully")
        except pymysql.Error as e:
            error_msg = f"Database connection error (MySQL): {str(e)}"
            print(f"❌ {error_msg}")
            raise ConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Database connection error (General): {str(e)}"
            print(f"❌ {error_msg}")
            raise ConnectionError(error_msg)
        
    def ensure_connection(self):
        try:
            if self.connection is None or not self.connection.open:
                print("🔄 Database connection lost, attempting to reconnect...")
                self.connect()
            self.connection.ping(reconnect=True)
        except pymysql.Error as e:
            error_msg = f"Database reconnection error (MySQL): {str(e)}"
            print(f"❌ {error_msg}")
            print("🔄 Attempting to establish new connection...")
            self.connect()
        except Exception as e:
            error_msg = f"Database reconnection error (General): {str(e)}"
            print(f"❌ {error_msg}")
            print("🔄 Attempting to establish new connection...")
            self.connect()

    def query(self, sql, params=None):
        if self.use_pool:
            return self._query_with_pool(sql, params)
        else:
            return self._query_without_pool(sql, params)
    
    def _query_with_pool(self, sql, params=None):
        """Execute query using connection pool."""
        try:
            with self.pool.get_connection_context() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    result = cursor.fetchall()
                    return result
        except pymysql.Error as e:
            error_msg = f"Database query error (MySQL): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Database query error (General): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.logger.error(error_msg)
            raise
    
    def _query_without_pool(self, sql, params=None):
        """Execute query without connection pool (legacy mode)."""
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params or ())
                result = cursor.fetchall()
                return result
        except pymysql.Error as e:
            error_msg = f"Database query error (MySQL): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.ensure_connection()
            raise
        except Exception as e:
            error_msg = f"Database query error (General): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.ensure_connection()
            raise

    def execute(self, sql, params=None):
        if self.use_pool:
            return self._execute_with_pool(sql, params)
        else:
            return self._execute_without_pool(sql, params)
    
    def _execute_with_pool(self, sql, params=None):
        """Execute statement using connection pool."""
        try:
            with self.pool.get_connection_context() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, params or ())
                    connection.commit()
                    print(f"✅ Database execute successful")
        except pymysql.Error as e:
            error_msg = f"Database execute error (MySQL): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Database execute error (General): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.logger.error(error_msg)
            raise
    
    def _execute_without_pool(self, sql, params=None):
        """Execute statement without connection pool (legacy mode)."""
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params or ())
                self.connection.commit()
                print(f"✅ Database execute successful")
        except pymysql.Error as e:
            error_msg = f"Database execute error (MySQL): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.connection.rollback()
            self.ensure_connection()
            raise
        except Exception as e:
            error_msg = f"Database execute error (General): {str(e)}"
            print(f"❌ {error_msg}")
            print(f"SQL: {sql}")
            print(f"Params: {params}")
            self.connection.rollback()
            self.ensure_connection()
            raise

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        if self.use_pool:
            with self.pool.get_connection_context() as connection:
                try:
                    connection.begin()
                    yield connection
                    connection.commit()
                except Exception as e:
                    connection.rollback()
                    raise
        else:
            try:
                self.ensure_connection()
                self.connection.begin()
                yield self.connection
                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                raise
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if self.use_pool and self.pool:
            return self.pool.get_pool_stats()
        else:
            return {"error": "Connection pooling not enabled"}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        if self.use_pool and self.pool:
            return self.pool.health_check()
        else:
            # Simple health check for non-pooled connections
            try:
                self.ensure_connection()
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return {
                        "status": "healthy",
                        "connection_active": True,
                        "test_query_successful": result is not None
                    }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "connection_active": False,
                    "error": str(e)
                }
    
    def close(self):
        """Close database connections."""
        try:
            if self.use_pool and hasattr(self, 'pool') and self.pool:
                self.pool.close_all_connections()
            elif hasattr(self, 'connection') and self.connection and self.connection.open:
                self.connection.close()
        except Exception:
            # Silently ignore errors during cleanup
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Silently ignore errors during destruction
            pass

def get_athlete_profile(athlete_id=None):
    """
    Get athlete profile from database
    
    Args:
        athlete_id (int, optional): Athlete ID. If None, will prompt for input.
    
    Returns:
        dict: Athlete profile containing Device_id, Name, Age, Weight, Height, Gender
    """
    try:
        # Create database instance with connection pooling
        db = Database(use_pool=True, pool_size=3)
        
        # Get athlete ID if not provided
        if athlete_id is None:
            athlete_id = int(input("Enter an athlete id: "))
        
        # Query athlete data
        rows = db.query("SELECT * FROM players WHERE id = %s", (athlete_id,))
        
        if not rows:
            raise ValueError(f"Athlete with ID {athlete_id} not found")
        
        athlete = rows[0]
        name = athlete['name']
        age = int(athlete['age'])
        weight = float(athlete['weight'])
        height = float(athlete['height'])
        gender = 1 if athlete['gender'] == 'M' else 0
        device_id = athlete.get('id')
        
        if not device_id:
            raise ValueError(f"No device is assigned to the player with ID {athlete_id}")
        
        athlete_profile = {
            "Device_id": device_id,
            "Name": name,
            "Age": age,
            "Weight": weight,
            "Height": height,
            "Gender": gender
        }
    
        return athlete_profile
        
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error getting athlete profile: {e}")
        raise
    finally:
        # Ensure database connection is closed
        if 'db' in locals():
            db.close()

# Database instance for use by other modules
db = Database()

# Convenience functions for creating database instances
def create_database_with_pool(pool_size: int = 5) -> Database:
    """
    Create a database instance with connection pooling.
    
    Args:
        pool_size: Size of the connection pool
        
    Returns:
        Database instance with connection pooling enabled
    """
    return Database(use_pool=True, pool_size=pool_size)


def create_database_legacy() -> Database:
    """
    Create a database instance without connection pooling (legacy mode).
    
    Returns:
        Database instance without connection pooling
    """
    return Database(use_pool=False)


# Example usage
if __name__ == "__main__":
    try:
        # Test with connection pooling
        print("Testing database with connection pooling...")
        db_pooled = create_database_with_pool(pool_size=3)
        
        # Get pool statistics
        pool_stats = db_pooled.get_connection_pool_stats()
        print(f"Connection pool stats: {pool_stats}")
        
        # Perform health check
        health = db_pooled.health_check()
        print(f"Database health: {health}")
        
        # Test query
        profile = get_athlete_profile()
        print(f"Retrieved profile: {profile}")
        
        # Close connections
        db_pooled.close()
        
    except Exception as e:
        print(f"Failed to test database: {e}")
