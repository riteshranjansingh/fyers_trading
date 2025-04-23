"""
Script to test the enhanced data fetcher with symbol resolution.

This script demonstrates how to:
- Initialize the symbol manager
- Use the data fetcher with symbol resolution
- Fetch extended historical data
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
from src.api.symbol_manager import SymbolManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test data fetcher with symbol management."""
    try:
        # Initialize connection
        logger.info("Initializing connection to FYERS API")
        connection = FyersConnection()
        
        # Test connection
        if not connection.test_connection():
            logger.error("Connection test failed")
            return
            
        logger.info("Connection successful!")
        
        # Initialize symbol manager
        logger.info("Initializing symbol manager")
        symbol_manager = SymbolManager()
        logger.info(f"Symbol manager status:\n{symbol_manager}")
        
        # Display available exchanges and symbol types
        exchanges = symbol_manager.get_exchanges()
        symbol_types = symbol_manager.get_symbol_types()
        logger.info(f"Available exchanges: {exchanges}")
        logger.info(f"Available symbol types: {symbol_types}")
        
        # Initialize data fetcher with symbol manager
        data_fetcher = FyersDataFetcher(connection, symbol_manager)
        
        # Test symbol resolution
        simple_symbol = "SBIN"
        logger.info(f"Testing symbol resolution for: {simple_symbol}")
        
        # Search for the symbol first to show it works
        matches = symbol_manager.search(name=simple_symbol, exact_match=True)
        if matches:
            logger.info(f"Found {len(matches)} matches for {simple_symbol}")
            for i, match in enumerate(matches[:3]):  # Show first 3 matches
                logger.info(f"Match {i+1}: {symbol_manager.format_symbol(match)}")
            
            # Use the first match for our test
            resolved_symbol = symbol_manager.format_symbol(matches[0])
            logger.info(f"Selected symbol for testing: {resolved_symbol}")
        else:
            logger.warning(f"No matches found for {simple_symbol}, using NSE:SBIN-EQ")
            resolved_symbol = "NSE:SBIN-EQ"
        
        # Fetch data for the last 30 days using the resolved symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"Fetching historical data for {resolved_symbol}")
        historical_data = data_fetcher.get_historical_data(
            symbol=resolved_symbol,
            resolution="1d",
            date_from=start_date,
            date_to=end_date
        )
        
        if historical_data.empty:
            logger.warning("No historical data received")
        else:
            logger.info(f"Retrieved {len(historical_data)} data points")
            logger.info("\nHistorical data sample (symbol resolution):")
            print(historical_data.head())
            print("\nHistorical data statistics:")
            print(historical_data.describe())
        
        # Test extended data fetching with the same symbol
        logger.info("\nTesting extended data fetching (multiple API calls)")
        
        # Try to get 2 years of daily data (exceeds the 366-day limit for a single call)
        extended_start = end_date - timedelta(days=2*365)  # ~2 years
        
        logger.info(f"Fetching extended historical data for {resolved_symbol} from {extended_start.date()} to {end_date.date()}")
        extended_data = data_fetcher.get_extended_historical_data(
            symbol=resolved_symbol,
            resolution="1d",
            date_from=extended_start,
            date_to=end_date
        )
        
        if extended_data.empty:
            logger.warning("No extended historical data received")
        else:
            logger.info(f"Retrieved {len(extended_data)} data points spanning approximately 2 years")
            logger.info("\nExtended data summary:")
            print(f"Date range: {extended_data.index.min()} to {extended_data.index.max()}")
            print(f"Number of trading days: {len(extended_data)}")
            logger.info("\nExtended data sample (first 5 rows):")
            print(extended_data.head())
            logger.info("\nExtended data sample (last 5 rows):")
            print(extended_data.tail())
        
        # Save the extended data to a CSV file
        if not extended_data.empty:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, f"{simple_symbol}_2yr_daily.csv")
            extended_data.to_csv(csv_path)
            logger.info(f"Saved extended data to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")

if __name__ == "__main__":
    main()