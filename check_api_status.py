import requests

def check_fyers_api_status():
    try:
        response = requests.get("https://api.fyers.in/api/v3", timeout=5)
        print(f"Status code: {response.status_code}")
        return response.status_code < 500  # Returns True if it's not a server error
    except Exception as e:
        print(f"Error connecting to FYERS API: {str(e)}")
        return False

print("Checking FYERS API status...")
if check_fyers_api_status():
    print("FYERS API appears to be responding")
else:
    print("FYERS API might be down or experiencing issues")