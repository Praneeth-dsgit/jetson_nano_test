import pymysql
from pymysql.cursors import DictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    
    def __init__(self):
        self.connection = None
        self.connect()

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
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            raise
        
    def ensure_connection(self):
        try:
            if self.connection is None or not self.connection.open:
                self.connect()
            self.connection.ping(reconnect=True)
        except Exception as e:
            print(f"Database reconnection error: {str(e)}")
            self.connect()

    def query(self, sql, params=None):
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params or ())
                return cursor.fetchall()
        except Exception as e:
            print(f"Database query error: {str(e)}")
            self.ensure_connection()
            raise

    def execute(self, sql, params=None):
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params or ())
                self.connection.commit()
        except Exception as e:
            print(f"Database execute error: {str(e)}")
            self.connection.rollback()
            self.ensure_connection()
            raise

    def close(self):
        """Explicitly close the database connection"""
        if self.connection and self.connection.open:
            self.connection.close()

    def __del__(self):
        self.close()

def get_athlete_profile(athlete_id=None):
    """
    Get athlete profile from database
    
    Args:
        athlete_id (int, optional): Athlete ID. If None, will prompt for input.
    
    Returns:
        dict: Athlete profile containing Device_id, Name, Age, Weight, Height, Gender
    """
    try:
        # Create database instance
        db = Database()
        
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

# Example usage
if __name__ == "__main__":
    try:
        profile = get_athlete_profile()
        print(f"Retrieved profile: {profile}")
    except Exception as e:
        print(f"Failed to get athlete profile: {e}")
