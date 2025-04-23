"""
Script to debug the symbol master files with more robust error handling.
"""

import os
import sys
import json
import logging
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_file(file_path):
    """Load a JSON file and return its content."""
    try:
        logger.info(f"Loading file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON with type: {type(data)}")
            return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def analyze_json_structure(data, file_name):
    """Analyze the structure of the JSON data."""
    if data is None:
        logger.error("No data to analyze")
        return
        
    # Check if it's a dictionary
    if isinstance(data, dict):
        logger.info(f"File {file_name} contains a dictionary with keys: {list(data.keys())}")
        
        # Check each key to see if there might be symbol data
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                logger.info(f"Key '{key}' contains a list with {len(value)} items")
                if len(value) > 0:
                    logger.info(f"First item in '{key}' list is: {type(value[0])}")
                    if isinstance(value[0], dict):
                        logger.info(f"First item keys: {list(value[0].keys())}")
                        search_for_symbol(value, "SBIN", key)
            elif isinstance(value, dict):
                logger.info(f"Key '{key}' contains a nested dictionary with keys: {list(value.keys())}")
    
    # Check if it's a list
    elif isinstance(data, list):
        logger.info(f"File {file_name} contains a list with {len(data)} items")
        
        # Look at the first few items to understand structure
        for i in range(min(3, len(data))):
            if isinstance(data[i], dict):
                logger.info(f"Item {i} is a dictionary with keys: {list(data[i].keys())}")
            else:
                logger.info(f"Item {i} is a {type(data[i])}")
        
        # Search for SBIN
        search_for_symbol(data, "SBIN", "root")
    
    # Other types
    else:
        logger.info(f"File {file_name} contains data of type: {type(data)}")

def search_for_symbol(data_list, symbol_name, container_name):
    """Search for a symbol in a list of dictionaries."""
    if not isinstance(data_list, list):
        logger.warning(f"Cannot search in {container_name} - not a list")
        return
        
    if len(data_list) == 0:
        logger.warning(f"List in {container_name} is empty")
        return
        
    # If first item is not a dictionary, we can't search effectively
    if not isinstance(data_list[0], dict):
        logger.warning(f"Items in {container_name} are not dictionaries, but {type(data_list[0])}")
        return
    
    # Count how many items have each key
    key_counter = Counter()
    for item in data_list[:100]:  # Check first 100 items
        if isinstance(item, dict):
            for key in item.keys():
                key_counter[key] += 1
    
    logger.info(f"Top 10 most common keys in {container_name}: {key_counter.most_common(10)}")
    
    # Look for the symbol in any text field
    found = False
    for i, item in enumerate(data_list):
        if not isinstance(item, dict):
            continue
            
        for key, value in item.items():
            if isinstance(value, str) and symbol_name.upper() in value.upper():
                logger.info(f"Found '{symbol_name}' in item {i}, key '{key}': {value}")
                logger.info(f"Full item: {json.dumps(item, indent=2)}")
                found = True
                break
        
        if found:
            break
    
    if not found:
        logger.info(f"Symbol '{symbol_name}' not found in {container_name}")

def main():
    """Main function to debug symbol files."""
    # Get fyers_trading directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations
    possible_paths = [
        os.path.join(current_dir, "data", "symbols"),
        os.path.join(current_dir, "src", "data", "symbols"),
        "data/symbols",
        "src/data/symbols"
    ]
    
    # Find all symbol master files
    symbol_files = []
    for base_path in possible_paths:
        if os.path.exists(base_path):
            logger.info(f"Checking directory: {base_path}")
            for filename in os.listdir(base_path):
                if filename.endswith(".json"):
                    full_path = os.path.join(base_path, filename)
                    symbol_files.append((filename, full_path))
    
    if not symbol_files:
        logger.error("No symbol files found. Please specify the path manually.")
        manual_path = input("Enter the full path to a symbol file: ")
        if os.path.exists(manual_path) and manual_path.endswith(".json"):
            filename = os.path.basename(manual_path)
            symbol_files.append((filename, manual_path))
        else:
            logger.error(f"Invalid path: {manual_path}")
            return
    
    logger.info(f"Found symbol files: {[f[0] for f in symbol_files]}")
    
    # Analyze each file
    for filename, file_path in symbol_files:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Analyzing file: {filename}")
        logger.info(f"{'=' * 50}")
        
        data = load_json_file(file_path)
        if data:
            analyze_json_structure(data, filename)

if __name__ == "__main__":
    main()