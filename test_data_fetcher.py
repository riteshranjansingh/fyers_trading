"""
Script to test the FyersDataFetcher functionality.

This script demonstrates how to:
- Create a connection to the FYERS API
- Initialize the data fetcher
- Fetch historical data for a symbol
- Get current market quotes
- Get market depth information
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.connection import FyersConnection
from src.api.data import FyersDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test data fetcher functionality."""
    try:
        # Initialize connection
        logger.info("Initializing connection to FYERS API")
        connection = FyersConnection()
        
        # Test connection
        if not connection.test_connection():
            logger.error("Connection test failed")
            return
            
        logger.info("Connection successful!")
        
        # Initialize data fetcher
        data_fetcher = FyersDataFetcher(connection)
        
        # Use a known symbol for testing
        symbol = "NSE:SBIN-EQ"
        logger.info(f"Using symbol {symbol} for testing")
        
        # Test historical data fetching
        logger.info(f"Fetching historical data for {symbol}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        historical_data = data_fetcher.get_historical_data(
            symbol=symbol,
            resolution="1d",
            date_from=start_date,
            date_to=end_date
        )
        
        if historical_data.empty:
            logger.warning("No historical data received")
        else:
            logger.info(f"Retrieved {len(historical_data)} data points")
            logger.info("\nHistorical data sample:")
            print(historical_data.head())
            
            # Calculate basic statistics
            logger.info("\nBasic statistics:")
            print(historical_data.describe())
        
        # Test market quotes
        logger.info(f"\nFetching current market quote for {symbol}")
        quotes = data_fetcher.get_market_quotes([symbol])
        logger.info(f"Quote data: {quotes}")
        
        # Test market depth
        logger.info(f"\nFetching market depth for {symbol}")
        depth = data_fetcher.get_market_depth(symbol)
        logger.info(f"Market depth data: {depth}")
                
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")

if __name__ == "__main__":
    main()