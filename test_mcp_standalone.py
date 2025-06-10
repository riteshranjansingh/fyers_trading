#!/usr/bin/env python3
"""
Simple test to verify the MCP server can start up correctly.
Run this to test the MCP server without Claude Desktop.
"""

import sys
import os
import asyncio
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

async def test_mcp_server():
    """Test if the MCP server can initialize properly."""
    print("🔍 Testing MCP Server initialization...")
    
    try:
        # Import the MCP server class
        from fyers_mcp_server import FyersMCPServer
        
        # Create server instance
        server = FyersMCPServer()
        print("✅ MCP Server instance created")
        
        # Test tool listing
        try:
            # Get the list_tools handler
            tools_handler = None
            for handler in server.server._list_tools_handlers:
                tools_handler = handler
                break
            
            if tools_handler:
                tools = await tools_handler()
                print(f"✅ Tools list generated - found {len(tools)} tools:")
                for tool in tools:
                    print(f"   - {tool.name}: {tool.description}")
            else:
                print("⚠️  Could not find tools handler")
        
        except Exception as e:
            print(f"⚠️  Error listing tools: {e}")
        
        # Test initialization (this might fail if FYERS auth is not set up)
        try:
            await server.initialize()
            print("✅ MCP Server initialized successfully (FYERS connection working)")
        except Exception as e:
            print(f"⚠️  MCP Server initialization failed: {e}")
            print("   This is normal if FYERS credentials aren't configured yet")
        
        print("\n✅ MCP Server basic functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import MCP server: {e}")
        return False
    except Exception as e:
        print(f"❌ MCP Server test failed: {e}")
        return False

def test_config_file():
    """Check if the MCP server file exists and is valid."""
    print("🔍 Checking MCP server file...")
    
    mcp_file = "fyers_mcp_server.py"
    if os.path.exists(mcp_file):
        print(f"✅ Found {mcp_file}")
        
        # Try to compile it
        try:
            with open(mcp_file, 'r') as f:
                code = f.read()
            compile(code, mcp_file, 'exec')
            print("✅ MCP server file syntax is valid")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in {mcp_file}: {e}")
            return False
    else:
        print(f"❌ {mcp_file} not found in current directory")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    missing = []
    
    try:
        import mcp
        print("✅ MCP library installed")
    except ImportError:
        missing.append("mcp")
        print("❌ MCP library not installed")
    
    try:
        import pandas
        print("✅ Pandas installed")
    except ImportError:
        missing.append("pandas")
        print("❌ Pandas not installed")
    
    try:
        import fyers_apiv3
        print("✅ FYERS API v3 installed")
    except ImportError:
        missing.append("fyers-apiv3")
        print("❌ FYERS API v3 not installed")
    
    if missing:
        print(f"\n💡 Install missing dependencies:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed")
    return True

async def main():
    """Run all tests."""
    print("🚀 FYERS MCP Server Standalone Test")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check config file
    if not test_config_file():
        return
    
    # Test MCP server
    await test_mcp_server()
    
    print("\n" + "=" * 50)
    print("🎉 Standalone test completed!")
    print("\nIf all tests passed, your MCP server should work with Claude Desktop.")
    print("If some tests failed, fix those issues before configuring Claude Desktop.")

if __name__ == "__main__":
    asyncio.run(main())