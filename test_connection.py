"""
Test script for FYERS API connection
"""
from src.api.connection import FyersConnection
import json

def main():
    # Initialize connection
    print("Initializing FYERS API connection...")
    conn = FyersConnection()
    
    # Authenticate and get session
    print("Authenticating with FYERS API...")
    if conn.authenticate():
        print("Authentication successful!")
        
        # Get the session
        session = conn.get_session()
        
        # Check if connection is successful
        if session:
            print("Connection successful!")
            
            # Test by fetching profile
            profile = session.get_profile()
            if profile.get("s") == "ok":
                print(f"\nProfile data: {json.dumps(profile['data'], indent=2)}")
            else:
                print(f"Failed to fetch profile: {profile}")
        else:
            print("Connection failed!")
    else:
        print("Authentication failed!")

if __name__ == "__main__":
    main()