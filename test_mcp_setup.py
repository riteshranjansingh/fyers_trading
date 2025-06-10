#!/usr/bin/env python3
"""
Test script to verify MCP server setup and FYERS integration.
Run this before setting up the MCP server to ensure everything works.
"""

import sys
import os
import json
from datetime import datetime, timedelta

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test FYERS modules
        from src.api.connection import FyersConnection
        from src.api.data import FyersDataFetcher
        from src.api.symbol_manager import SymbolManager
        from src.api.data_utils import add_technical_indicators
        from src.utils.logging_config import setup_logging
        print("✅ FYERS modules imported successfully")
    except ImportError as e:
        print(f"❌ FYERS module import failed: {e}")
        return False
    
    try:
        # Test MCP modules
        from mcp.server.models import InitializationOptions
        from mcp.server import Server
        from mcp.types import Tool, TextContent
        from mcp.server.stdio import stdio_server
        print("✅ MCP modules imported successfully")
    except ImportError as e:
        print(f"❌ MCP module import failed: {e}")
        print("💡 Run: pip install mcp")
        return False
    
    return True

def check_credentials():
    """Check if FYERS credentials are properly configured."""
    print("🔍 Checking FYERS credentials...")
    
    try:
        from config.credentials import FYERS_APP_ID, FYERS_APP_SECRET, FYERS_CLIENT_ID, FYERS_REDIRECT_URI
        
        # Check if credentials are still template values
        if (FYERS_APP_ID == "YOUR_APP_ID" or 
            FYERS_APP_SECRET == "YOUR_APP_SECRET" or 
            FYERS_CLIENT_ID == "YOUR_CLIENT_ID" or 
            FYERS_REDIRECT_URI == "YOUR_REDIRECT_URI"):
            print("❌ FYERS credentials are still using template values")
            print("💡 Please update config/credentials.py with your actual FYERS API credentials")
            return False
        
        print("✅ FYERS credentials are configured")
        return True
        
    except ImportError as e:
        print("❌ Could not import credentials")
        print("💡 Make sure config/credentials.py exists and is properly configured")
        return False

def check_authentication_status():
    """Check if user is already authenticated with FYERS."""
    print("\n🔍 Checking FYERS authentication status...")
    
    try:
        from src.api.connection import FyersConnection
        connection = FyersConnection()
        
        # Check if there's a cached token
        if connection._load_token():
            print("✅ Found cached authentication token")
            
            # Test if the token still works
            try:
                if connection._create_session():
                    if connection.test_connection():
                        print("✅ Authentication token is valid and working")
                        return True, connection
                    else:
                        print("❌ Authentication token exists but API test failed")
                        print("💡 Token might be expired, need to re-authenticate")
                        return False, None
                else:
                    print("❌ Could not create session with cached token")
                    return False, None
            except Exception as e:
                print(f"❌ Token validation failed: {e}")
                return False, None
        else:
            print("❌ No cached authentication token found")
            print("💡 Need to authenticate with FYERS first")
            return False, None
            
    except Exception as e:
        print(f"❌ Authentication check failed: {e}")
        return False, None

def authenticate_fyers():
    """Guide user through FYERS authentication."""
    print("\n🔐 FYERS Authentication Required")
    print("=" * 30)
    print("FYERS requires daily authentication. This will:")
    print("1. Open your browser to FYERS login page")
    print("2. Ask you to paste the redirect URL")
    print("3. Save the token for future use")
    print()
    
    user_input = input("Do you want to authenticate now? (y/n): ")
    
    if user_input.lower() != 'y':
        print("⏭️  Skipping authentication. MCP server tests will be limited.")
        return None
    
    try:
        from src.api.connection import FyersConnection
        print("\n⏳ Starting FYERS authentication...")
        
        connection = FyersConnection()
        if connection.authenticate():
            print("✅ FYERS authentication successful!")
            return connection
        else:
            print("❌ FYERS authentication failed")
            return None
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return None

def test_symbol_manager():
    """Test symbol manager functionality."""
    print("\n🔍 Testing symbol manager...")
    
    try:
        from src.api.symbol_manager import SymbolManager
        symbol_manager = SymbolManager()
        
        # Test symbol search
        symbols = symbol_manager.search(name="SBIN", exact_match=True)
        if symbols:
            print(f"✅ Symbol search working - found {len(symbols)} SBIN symbols")
            return True, symbol_manager
        else:
            print("⚠️  No SBIN symbols found - this might be normal if symbol files aren't downloaded")
            
            # Try to download symbol files
            print("⏳ Attempting to download symbol files...")
            results = symbol_manager.download_all_symbol_files()
            
            if any(results.values()):
                print("✅ Symbol files downloaded successfully")
                
                # Test search again
                symbols = symbol_manager.search(name="SBIN", exact_match=True)
                if symbols:
                    print(f"✅ Symbol search now working - found {len(symbols)} SBIN symbols")
                    return True, symbol_manager
            
            print("❌ Symbol manager test failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Symbol manager test failed: {e}")
        return False, None

def test_data_fetcher(connection, symbol_manager):
    """Test data fetching with a simple request."""
    print("\n🔍 Testing data fetcher...")
    
    if not connection:
        print("❌ Cannot test data fetcher - no authenticated connection")
        return False
    
    try:
        from src.api.data import FyersDataFetcher
        
        data_fetcher = FyersDataFetcher(connection, symbol_manager)
        
        # Test fetching a small amount of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Just 5 days
        
        print("⏳ Fetching 5 days of SBIN data...")
        df = data_fetcher.get_historical_data(
            symbol="NSE:SBIN-EQ",
            resolution="1d",
            date_from=start_date,
            date_to=end_date
        )
        
        if not df.empty:
            print(f"✅ Data fetching successful - got {len(df)} records")
            print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
            return True
        else:
            print("❌ Data fetching failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ Data fetcher test failed: {e}")
        return False

def generate_config_template():
    """Generate Claude Desktop config template."""
    print("\n📝 Generating Claude Desktop config template...")
    
    # Get the absolute path to the MCP server file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_server_path = os.path.join(current_dir, "fyers_mcp_server.py")
    
    config = {
        "mcpServers": {
            "fyers-trading": {
                "command": "python",
                "args": [mcp_server_path],
                "env": {
                    "PYTHONPATH": current_dir
                }
            }
        }
    }
    
    print("📋 Add this to your Claude Desktop config file:")
    print("   Location: ~/.config/claude-desktop/claude_desktop_config.json (Linux/Mac)")
    print("   Location: %APPDATA%\\Claude\\claude_desktop_config.json (Windows)")
    print("\nConfig JSON:")
    print(json.dumps(config, indent=2))
    
    return config

def main():
    """Run all tests with proper authentication flow."""
    print("🚀 FYERS MCP Server Setup Test")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import test failed. Please fix imports before proceeding.")
        return False
    
    # Check credentials configuration
    if not check_credentials():
        print("\n❌ Please configure your FYERS credentials first.")
        print("💡 Edit config/credentials.py with your actual FYERS API details")
        return False
    
    # Check authentication status
    is_authenticated, connection = check_authentication_status()
    
    if not is_authenticated:
        # Try to authenticate
        connection = authenticate_fyers()
        
        if not connection:
            print("\n⚠️  Skipping FYERS-dependent tests due to authentication issues.")
            print("🚀 You can still test the MCP server basic functionality!")
            
            # Test symbol manager (doesn't need authentication)
            symbol_success, symbol_manager = test_symbol_manager()
            
            # Generate config
            generate_config_template()
            
            print("\n" + "=" * 50)
            print("🎯 Partial setup completed!")
            print("\nNext steps:")
            print("1. Complete FYERS authentication when ready")
            print("2. Update Claude Desktop config")
            print("3. Test MCP server with Claude")
            return True
    
    print("✅ FYERS authentication confirmed!")
    
    # Test symbol manager
    symbol_success, symbol_manager = test_symbol_manager()
    
    # Test data fetcher (only if we have authentication)
    if connection and symbol_manager:
        test_data_fetcher(connection, symbol_manager)
    
    # Generate config
    generate_config_template()
    
    print("\n" + "=" * 50)
    print("🎉 Full setup test completed!")
    print("\nNext steps:")
    print("1. Update Claude Desktop config with the JSON above")
    print("2. Restart Claude Desktop")
    print("3. Test the MCP server in Claude")
    
    return True

if __name__ == "__main__":
    main()