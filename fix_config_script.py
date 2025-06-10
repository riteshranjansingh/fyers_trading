#!/usr/bin/env python3
"""
Script to fix Claude Desktop config file issues.
Run this to create a clean config file.
"""

import json
import os
from pathlib import Path

def get_config_path():
    """Get the correct Claude Desktop config path for this OS."""
    home = Path.home()
    
    # Try different possible locations
    possible_paths = [
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",  # Mac
        home / ".config" / "claude-desktop" / "claude_desktop_config.json",  # Linux
        Path(os.environ.get('APPDATA', '')) / "Claude" / "claude_desktop_config.json"  # Windows
    ]
    
    # Check which one exists or create the most likely one
    for path in possible_paths:
        if path.parent.exists():
            return path
    
    # Default to Mac path since user is on Mac
    return possible_paths[0]

def create_clean_config():
    """Create a clean Claude Desktop config file."""
    
    config = {
        "mcpServers": {
            "fyers-trading": {
                "command": "/Users/Work/Documents/projects/ai/fyers/fyers_trading/venv/bin/python3",
                "args": ["/Users/Work/Documents/projects/ai/fyers/fyers_trading/fyers_mcp_server.py"],
                "env": {
                    "PYTHONPATH": "/Users/Work/Documents/projects/ai/fyers/fyers_trading"
                }
            }
        }
    }
    
    config_path = get_config_path()
    
    print(f"🔍 Claude Desktop config path: {config_path}")
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write clean config
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ Clean config file created successfully!")
        print(f"📍 Location: {config_path}")
        
        # Verify the file by reading it back
        with open(config_path, 'r', encoding='utf-8') as f:
            test_config = json.load(f)
        
        print("✅ Config file validates correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return False

def check_existing_config():
    """Check the existing config file for issues."""
    config_path = get_config_path()
    
    if not config_path.exists():
        print("❌ Config file doesn't exist yet")
        return False
    
    print(f"🔍 Checking existing config: {config_path}")
    
    try:
        # Read as bytes first to check for BOM or hidden characters
        with open(config_path, 'rb') as f:
            raw_content = f.read()
        
        print(f"📊 File size: {len(raw_content)} bytes")
        print(f"🔤 First 20 bytes: {raw_content[:20]}")
        
        # Check for BOM
        if raw_content.startswith(b'\xef\xbb\xbf'):
            print("⚠️  File has UTF-8 BOM - this might cause issues")
        
        # Try to read as text
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📝 File content preview:")
        print(content[:100] + "..." if len(content) > 100 else content)
        
        # Try to parse JSON
        config = json.loads(content)
        print("✅ JSON is valid!")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"   Error at position {e.pos}")
        return False
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return False

def main():
    """Main function to fix config issues."""
    print("🔧 Claude Desktop Config Fixer")
    print("=" * 40)
    
    # First check existing config
    if check_existing_config():
        print("\n✅ Existing config is valid - no changes needed!")
        print("🔄 Try restarting Claude Desktop")
        return
    
    print("\n🛠️  Creating new clean config file...")
    
    if create_clean_config():
        print("\n🎉 Config file fixed!")
        print("\nNext steps:")
        print("1. Restart Claude Desktop")
        print("2. Check if MCP server connects")
        print("3. Test with: 'Search for SBIN symbols'")
    else:
        print("\n❌ Failed to create config file")
        print("💡 Try creating the file manually in a text editor")

if __name__ == "__main__":
    main()