"""
FYERS API data module
Handles data fetching from FYERS API
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FyersData:
    """
    Class to handle data fetching from FYERS API
    """
    def __init__(self, fyers_session):
        """
        Initialize with a valid FYERS session
        """
        self.fyers = fyers_session
        
    def get_historical_data(self, symbol, resolution="D", date_from=None, date_to=None, days_back=30):
        """
        Fetch historical data for a symbol
        
        Parameters:
        - symbol: Trading symbol (e.g., "NSE:SBIN-EQ")
        - resolution: Timeframe of candles (1, 2, 3, 5, 10, 15, 20, 30, 60, "D", "W", "M")
        - date_from: Start date in format "YYYY-MM-DD"
        - date_to: End date in format "YYYY-MM-DD"
        - days_back: Number of days to look back if dates not provided
        
        Returns:
        - Pandas DataFrame with historical data
        """
        # Process dates
        if not date_to:
            date_to = datetime.now().strftime("%Y-%m-%d")
            
        if not date_from:
            date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
        # Convert dates to timestamps
        from_timestamp = int(datetime.strptime(date_from, "%Y-%m-%d").timestamp())
        to_timestamp = int(datetime.strptime(date_to, "%Y-%m-%d").timestamp())
        
        logger.info(f"Fetching historical data for {symbol} from {date_from} to {date_to}")
        
        # Prepare data request
        data_req = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": 1,  # UNIX timestamp
            "range_from": from_timestamp,
            "range_to": to_timestamp,
            "cont_flag": 1  # Continuous data for futures and options
        }
        
        try:
            # Fetch the data
            resp = self.fyers.history(data_req)
            
            if "code" in resp and resp["code"] == 200:
                # Convert to DataFrame
                candles = resp["candles"]
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                
                logger.info(f"Successfully fetched {len(df)} records")
                return df
            else:
                logger.error(f"Failed to fetch data: {resp}")
                return None
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def get_quote(self, symbols):
        """
        Get current market quotes for symbols
        
        Parameters:
        - symbols: List of symbols or a single symbol string
        
        Returns:
        - Dictionary with quote data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        logger.info(f"Fetching quotes for {symbols}")
        
        try:
            resp = self.fyers.quotes({"symbols": ",".join(symbols)})
            
            if "code" in resp and resp["code"] == 200:
                logger.info("Successfully fetched quotes")
                return resp["d"]
            else:
                logger.error(f"Failed to fetch quotes: {resp}")
                return None
        except Exception as e:
            logger.error(f"Error fetching quotes: {str(e)}")
            return None
    
    def get_market_depth(self, symbol):
        """
        Get market depth (order book) for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with market depth data
        """
        logger.info(f"Fetching market depth for {symbol}")
        
        try:
            resp = self.fyers.market_depth({"symbol": symbol})
            
            if "code" in resp and resp["code"] == 200:
                logger.info("Successfully fetched market depth")
                return resp["d"]
            else:
                logger.error(f"Failed to fetch market depth: {resp}")
                return None
        except Exception as e:
            logger.error(f"Error fetching market depth: {str(e)}")
            return None