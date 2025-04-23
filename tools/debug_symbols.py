"""
Script to debug the symbol master files and understand the structure.
"""
import os
import sys
import json
import logging
from collections import Counter

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_file(file_path):
    """Load a JSON file and return its content."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return []

def main():
    """Main function to debug symbol files."""
    # Define paths to symbol files
    # Try different possible locations
    base_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "symbols"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "data", "symbols"),
        "data/symbols",
        "src/data/symbols"
    ]
    
    # Find the first existing path
    base_path = None
    for path in base_paths:
        if os.path.exists(path):
            base_path = path
            logger.info(f"Found symbol files directory: {base_path}")
            break
    
    if not base_path:
        logger.error("Could not find symbol files directory. Please specify the correct path.")
        # Ask user for path
        user_path = input("Enter the full path to the symbols directory: ")
        if os.path.exists(user_path):
            base_path = user_path
        else:
            logger.error(f"Path not found: {user_path}")
            return
    
    # List all symbol files
    symbol_files = {}
    for filename in os.listdir(base_path):
        if filename.endswith("_sym_master.json"):
            segment = filename.split("_sym_master.json")[0]
            symbol_files[segment] = os.path.join(base_path, filename)
    
    logger.info(f"Found symbol files: {list(symbol_files.keys())}")
    
    # Check each file
    for segment, file_path in symbol_files.items():
        symbols = load_json_file(file_path)
        
        if not symbols:
            logger.warning(f"No symbols found in {segment} file")
            continue
        
        logger.info(f"\n{segment}: {len(symbols)} symbols")
        
        # Analyze keys in first symbol to understand structure
        if len(symbols) > 0:
            logger.info(f"Keys in first symbol: {list(symbols[0].keys())}")
            
            # Check for specific fields we expect to use
            key_counts = Counter()
            for symbol in symbols:
                for key in symbol.keys():
                    key_counts[key] += 1
            
            logger.info(f"Fields present in records: {dict(key_counts)}")
        
        # Search for 'SBIN'
        found = False
        for i, symbol in enumerate(symbols):
            # Try different ways the symbol might be referenced
            for key in ['name', 'symbol_name', 'symbol']:
                if key in symbol and 'SBIN' in str(symbol.get(key, '')):
                    logger.info(f"Found SBIN in {segment} at index {i} with key '{key}':")
                    logger.info(json.dumps(symbol, indent=2))
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            logger.info(f"Could not find 'SBIN' in {segment} file")

if __name__ == "__main__":
    main()