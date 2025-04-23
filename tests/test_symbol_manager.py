"""
Script to test the SymbolManager functionality.

This script demonstrates how to:
- Initialize the symbol manager
- Download symbol master files
- Search for symbols
- Get information about specific symbols
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.symbol_manager import SymbolManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test symbol manager functionality."""
    try:
        # Initialize the symbol manager - will download symbol files if needed
        logger.info("Initializing SymbolManager")
        start_time = time.time()
        symbol_manager = SymbolManager()
        
        # Show loading time
        load_time = time.time() - start_time
        logger.info(f"Symbol manager initialized in {load_time:.2f} seconds")
        
        # Print symbol manager status
        logger.info(f"Symbol manager status:\n{symbol_manager}")
        
        # Show available exchanges and symbol types
        logger.info(f"Available exchanges: {symbol_manager.get_exchanges()}")
        logger.info(f"Available symbol types: {symbol_manager.get_symbol_types()}")
        
        # Test searching for symbols
        search_term = "SBIN"
        logger.info(f"\nSearching for symbols with name containing '{search_term}'")
        symbols = symbol_manager.search(name=search_term)
        logger.info(f"Found {len(symbols)} matching symbols")
        
        # Display first 5 symbols
        if symbols:
            logger.info("First 5 matches:")
            for i, symbol in enumerate(symbols[:5]):
                formatted = symbol_manager.format_symbol(symbol)
                logger.info(f"{i+1}. {formatted} - {symbol.get('description', 'N/A')}")
        
        # Test getting info for a specific symbol
        test_symbol = "NSE:SBIN-EQ"
        logger.info(f"\nGetting info for symbol: {test_symbol}")
        symbol_info = symbol_manager.get_symbol_info(test_symbol)
        
        if symbol_info:
            logger.info("Symbol information:")
            for key, value in sorted(symbol_info.items()):
                logger.info(f"  {key}: {value}")
            
            # Get the fyToken
            fytoken = symbol_manager.get_fytoken(test_symbol)
            logger.info(f"fyToken for {test_symbol}: {fytoken}")
            
            # Lookup by fyToken
            if fytoken:
                reverse_lookup = symbol_manager.get_symbol_by_fytoken(fytoken)
                if reverse_lookup:
                    reverse_symbol = symbol_manager.format_symbol(reverse_lookup)
                    logger.info(f"Symbol for fyToken {fytoken}: {reverse_symbol}")
        else:
            logger.warning(f"No information found for {test_symbol}")
        
        # Test getting all equity symbols from NSE
        logger.info("\nGetting all NSE equity symbols")
        nse_equities = symbol_manager.get_all_symbols(exchange="NSE", symbol_type="EQ")
        logger.info(f"Found {len(nse_equities)} NSE equity symbols")
        
        # Show first 10 symbols
        if nse_equities:
            logger.info("First 10 NSE equity symbols:")
            for i, symbol in enumerate(nse_equities[:10]):
                logger.info(f"{i+1}. {symbol}")
                
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")

if __name__ == "__main__":
    main()