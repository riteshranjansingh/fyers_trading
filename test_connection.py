"""
Test script for FYERS API connection
"""
from src.api.connection import FyersConnection

def main():
    # Initialize connection
    print("Initializing FYERS API connection...")
    conn = FyersConnection()
    
    # Connect to API
    session = conn.connect()
    
    # Check if connection is successful
    if session:
        print("Connection successful!")
        
        # Test by fetching profile
        profile = session.get_profile()
        if "code" in profile and profile["code"] == 200:
            print(f"Profile data: {profile['data']}")
        else:
            print(f"Failed to fetch profile: {profile}")
    else:
        print("Connection failed!")

if __name__ == "__main__":
    main()