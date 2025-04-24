"""
Verification script to test project structure and functionality after restructuring.
"""
"""
Verification script to test project structure and functionality after restructuring.
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
from src.utils.logging_config import setup_logging
logger = setup_logging()

# Import from project modules
from src.api.connection import FyersConnection
from src.api.data import FyersDataFetcher
from src.api.symbol_manager import SymbolManager


# import sys
# import os
# import logging
# from datetime import datetime, timedelta
# import pandas as pd

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Import from project modules
# from src.api.connection import FyersConnection
# from src.api.data import FyersDataFetcher
# from src.api.symbol_manager import SymbolManager

def main():
    logger.info("Starting verification of project structure and functionality")
    
    # Step 1: Test connection
    logger.info("Step 1: Testing connection to FYERS API")
    connection = FyersConnection()
    
    if not connection.authenticate():
        logger.error("Failed to authenticate with FYERS API")
        return False
    
    logger.info("Successfully authenticated with FYERS API")
    
    # Step 2: Initialize symbol manager
    logger.info("Step 2: Initializing symbol manager")
    symbol_manager = SymbolManager()
    
    # Step 3: Initialize data fetcher
    logger.info("Step 3: Initializing data fetcher")
    data_fetcher = FyersDataFetcher(connection, symbol_manager)
    
    # Step 4: Fetch historical data
    logger.info("Step 4: Fetching historical data")
    symbol = "NSE:SBIN-EQ"  # State Bank of India
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Start with 30 days as a quick test
    
    try:
        data = data_fetcher.get_historical_data(
            symbol=symbol,
            resolution="1d",
            date_from=start_date,
            date_to=end_date
        )
        
        if data.empty:
            logger.warning("No data returned for the symbol")
        else:
            logger.info(f"Successfully fetched {len(data)} records")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            logger.info("\nSample data:")
            print(data.head())
            
            # Try calculating some statistics
            logger.info("\nBasic statistics:")
            print(data.describe())
            
            # If 30-day fetch worked, try a longer period
            logger.info("\nTesting extended data fetching for 2 years")
            start_date_2yr = end_date - timedelta(days=2*365)
            
            extended_data = data_fetcher.get_extended_historical_data(
                symbol=symbol,
                resolution="1d",
                date_from=start_date_2yr,
                date_to=end_date
            )
            
            if extended_data.empty:
                logger.warning("No extended data returned")
            else:
                logger.info(f"Successfully fetched {len(extended_data)} records spanning approximately 2 years")
                logger.info(f"Extended date range: {extended_data.index.min()} to {extended_data.index.max()}")
                
                # Save to CSV to verify file operations
                csv_path = f"{symbol.replace(':', '_').replace('-', '_')}_2yr_verify.csv"
                extended_data.to_csv(csv_path)
                logger.info(f"Saved extended data to {csv_path}")
            
        logger.info("Verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()