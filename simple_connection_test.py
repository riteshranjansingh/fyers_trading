"""
Simple test script for FYERS API v3 connection
"""
from src.api.connection import get_auth_code, get_access_token, create_fyers_session
import json

def main():
    print("=" * 50)
    print("FYERS API v3 Connection Test")
    print("=" * 50)
    
    try:
        print("\nStep 1: Getting authorization code...")
        auth_code = get_auth_code()
        print(f"Auth code received: {auth_code[:10]}..." if len(auth_code) > 10 else auth_code)
        
        print("\nStep 2: Generating access token...")
        access_token = get_access_token(auth_code)
        
        if access_token:
            print(f"Access token generated: {access_token[:10]}...")
            
            print("\nStep 3: Creating FYERS session...")
            fyers = create_fyers_session(access_token)
            
            if fyers:
                print("\nTesting connection by fetching profile...")
                try:
                    # Using the profile API endpoint as per documentation
                    profile = fyers.get_profile()
                    print(f"API Response: {json.dumps(profile, indent=2)}")
                    
                    if profile.get("s") == "ok":
                        print("\nConnection successful!")
                        print(f"Profile data: {profile}")
                    else:
                        print(f"\nConnection test failed with response: {profile}")
                        
                        # Let's try a different API endpoint as a test
                        print("\nTrying an alternative API call (get_funds)...")
                        funds = fyers.funds()
                        print(f"Funds API Response: {json.dumps(funds, indent=2)}")
                except Exception as e:
                    print(f"Error making API call: {str(e)}")
                    print("Let's try with a simpler test call...")
                    try:
                        # Make a very basic API call to test connection
                        simple_test = fyers.get_funds()
                        print(f"Simple test response: {json.dumps(simple_test, indent=2)}")
                    except Exception as e2:
                        print(f"Error with simple test: {str(e2)}")
            else:
                print("\nFailed to create FYERS session")
        else:
            print("\nFailed to generate access token")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()