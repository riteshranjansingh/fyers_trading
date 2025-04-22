"""
FYERS API v3 connection module - Robust version
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
import webbrowser
from urllib.parse import urlparse, parse_qs
from fyers_apiv3 import fyersModel

# Add the project root to the path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.credentials import FYERS_APP_ID, FYERS_APP_SECRET, FYERS_CLIENT_ID, FYERS_REDIRECT_URI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FyersConnection:
    """
    A class to handle FYERS API authentication and connection
    with token caching and auto-renewal
    """
    
    def __init__(self, token_file_path=None):
        """
        Initialize the connection handler
        
        Args:
            token_file_path: Path to file for caching tokens (default: None)
        """
        self.app_id = FYERS_APP_ID
        self.app_secret = FYERS_APP_SECRET
        self.client_id = FYERS_CLIENT_ID
        self.redirect_uri = FYERS_REDIRECT_URI
        self.token_file = token_file_path or os.path.join(os.path.dirname(__file__), '..', '..', 'token_cache.json')
        self.fyers = None
        self.access_token = None
        self.token_expiry = None
    
    def _extract_auth_code(self, redirect_url):
        """Extract auth code from redirect URL"""
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        
        if 'auth_code' in query_params:
            return query_params['auth_code'][0]
        return None
    
    def _get_auth_code(self):
        """Get authorization code through user interaction"""
        try:
            session = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.app_secret,
                redirect_uri=self.redirect_uri,
                response_type="code"
            )
            
            auth_url = session.generate_authcode()
            
            print(f"\nPlease visit this URL to authorize the application:")
            print(auth_url)
            print("\nAfter logging in, you'll be redirected to your redirect URI.")
            
            try:
                webbrowser.open(auth_url)
                print("(Browser window should have opened with the URL)")
            except:
                print("Could not open browser automatically. Please copy and paste the URL manually.")
            
            redirect_url = input("\nPlease paste the FULL redirect URL you were sent to: ")
            auth_code = self._extract_auth_code(redirect_url)
            
            if auth_code:
                print(f"Successfully extracted auth_code: {auth_code[:10]}...")
                return auth_code
            else:
                print("Could not extract auth_code from the URL.")
                manual_code = input("Please manually enter the auth_code if you can see it: ")
                return manual_code.strip()
        except Exception as e:
            logger.error(f"Error getting auth code: {str(e)}")
            return None
    
    def _get_access_token(self, auth_code):
        """Get access token using auth code"""
        try:
            session = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.app_secret,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response.get("access_token"):
                # Calculate token expiry (typically 24 hours from now)
                self.access_token = response["access_token"]
                self.token_expiry = (datetime.now() + timedelta(hours=24)).timestamp()
                
                # Save token to file
                self._save_token()
                
                logger.info("Access token generated and saved successfully")
                return self.access_token
            else:
                logger.error(f"Failed to get access token: {response}")
                return None
        except Exception as e:
            logger.error(f"Error getting access token: {str(e)}")
            return None
    
    def _save_token(self):
        """Save token to file"""
        if not self.access_token or not self.token_expiry:
            return False
        
        token_data = {
            "access_token": self.access_token,
            "expiry": self.token_expiry
        }
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.token_file)), exist_ok=True)
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
            logger.info(f"Token saved to {self.token_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving token: {str(e)}")
            return False
    
    def _load_token(self):
        """Load token from file"""
        try:
            if not os.path.exists(self.token_file):
                logger.info("No token file found")
                return False
            
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            self.access_token = token_data.get("access_token")
            self.token_expiry = token_data.get("expiry")
            
            # Check if token is expired
            if not self.token_expiry or datetime.now().timestamp() >= self.token_expiry:
                logger.info("Loaded token is expired")
                return False
            
            logger.info("Token loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading token: {str(e)}")
            return False
    
    def authenticate(self, force_new=False):
        """
        Authenticate with FYERS API
        
        Args:
            force_new: Force new authentication even if valid token exists
            
        Returns:
            bool: True if authenticated successfully, False otherwise
        """
        # Try to load token from file if not forcing new auth
        if not force_new and self._load_token():
            logger.info("Using cached token")
            return self._create_session()
        
        # Get new token
        logger.info("Getting new authentication token")
        auth_code = self._get_auth_code()
        if not auth_code:
            return False
        
        if self._get_access_token(auth_code):
            return self._create_session()
        
        return False
    
    def _create_session(self):
        """Create FYERS API session"""
        if not self.access_token:
            logger.error("No access token available")
            return False
        
        try:
            self.fyers = fyersModel.FyersModel(client_id=self.app_id, token=self.access_token)
            logger.info("FYERS session created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating FYERS session: {str(e)}")
            return False
    
    def get_session(self):
        """
        Get active FYERS session, creating one if needed
        
        Returns:
            FyersModel instance or None if authentication fails
        """
        if not self.fyers and not self.authenticate():
            return None
        return self.fyers
    
    def test_connection(self):
        """Test the API connection by fetching profile data"""
        session = self.get_session()
        if not session:
            return False
        
        try:
            profile = session.get_profile()
            if profile.get("s") == "ok":
                logger.info("Connection test successful")
                return True
            
            logger.error(f"Connection test failed: {profile}")
            return False
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False