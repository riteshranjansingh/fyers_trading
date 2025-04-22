import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
from .connection import FyersConnection

logger = logging.getLogger(__name__)

class FyersDataFetcher:
    """
    Class to handle all data fetching operations with the FYERS API v3.
    
    This class is responsible for:
    - Fetching historical market data
    - Retrieving current market quotes
    - Managing symbol information
    - Data preprocessing and formatting
    """
    
    def __init__(self, connection: FyersConnection):
        """
        Initialize the data fetcher with a FyersConnection instance.
        
        Args:
            connection: An authenticated FyersConnection instance
        """
        self.connection = connection
        self.fyers = connection.get_session()
        
    def get_historical_data(
        self, 
        symbol: str, 
        resolution: str,
        date_from: Union[str, datetime],
        date_to: Optional[Union[str, datetime]] = None,
        continuous: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price data from FYERS API.
        
        Args:
            symbol: Trading symbol in FYERS format (e.g., "NSE:SBIN-EQ")
            resolution: Timeframe of the data (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo)
            date_from: Start date in YYYY-MM-DD format or datetime object
            date_to: End date in YYYY-MM-DD format or datetime object (default: today)
            continuous: Whether to fetch data with continuity or not
            
        Returns:
            DataFrame with historical price data (OHLCV)
        """
        # Convert dates to string format if datetime objects
        if isinstance(date_from, datetime):
            date_from = date_from.strftime("%Y-%m-%d")
            
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(date_to, datetime):
            date_to = date_to.strftime("%Y-%m-%d")
            
        # Map resolution to FYERS API format
        resolution_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "1d": "D", "1w": "W", "1mo": "M"
        }
        
        if resolution not in resolution_map:
            raise ValueError(f"Invalid resolution: {resolution}. Must be one of {list(resolution_map.keys())}")
            
        # Prepare request parameters
        params = {
            "symbol": symbol,
            "resolution": resolution_map[resolution],
            "date_format": "1",  # YYYY-MM-DD format
            "range_from": date_from,
            "range_to": date_to,
            "cont_flag": "1" if continuous else "0"
        }
        
        try:
            logger.info(f"Fetching historical data for {symbol} from {date_from} to {date_to}")
            response = self.fyers.history(params)
            
            # Check response structure
            if response.get("s") != "ok":
                logger.warning(f"Error in response for {symbol}: {response}")
                return pd.DataFrame()
                
            # Get candles from the response
            candles = response.get("candles", [])
            if not candles:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
                
            # Process the response into a DataFrame
            data = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            
            # Convert timestamp to datetime
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
            data.set_index("timestamp", inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def get_market_quotes(self, symbols: List[str]) -> Dict:
        """
        Get current market quotes for specified symbols.
        
        Args:
            symbols: List of trading symbols in FYERS format
            
        Returns:
            Dictionary with quote information
        """
        try:
            logger.info(f"Fetching market quotes for {symbols}")
            
            # Convert list to comma-separated string if needed
            symbols_str = ",".join(symbols) if isinstance(symbols, list) else symbols
            
            # Prepare request parameters - using quotes_snapshot for FYERS API v3
            params = {"symbols": symbols_str}
            response = self.fyers.quotes(params)
            
            # Check for successful response
            if response.get("s") == "ok" and "d" in response:
                return response["d"]
            else:
                logger.warning(f"Unexpected response format for quotes: {response}")
                return {}
            
        except Exception as e:
            logger.error(f"Error fetching market quotes: {str(e)}")
            raise
    
    def get_symbol_info(self, search_text: str) -> List[Dict]:
        """
        Search for symbols matching the search text.
        
        Args:
            search_text: Text to search for in symbol names
            
        Returns:
            List of matching symbols with their information
        """
        try:
            logger.info(f"Searching for symbols with text: {search_text}")
            
            # Using the correct method for FYERS API v3
            response = self.fyers.symbol_master({"symbol_text": search_text})
            
            # Check if response is successful
            if response.get("s") == "ok" and "symbols" in response.get("d", {}):
                return response["d"]["symbols"]
            else:
                logger.warning(f"Symbol search returned unexpected format: {response}")
                return []
            
        except Exception as e:
            logger.error(f"Error searching symbols: {str(e)}")
            raise
    
    def get_market_depth(self, symbol: str) -> Dict:
        """
        Get market depth (order book) for a symbol.
        
        Args:
            symbol: Trading symbol in FYERS format
            
        Returns:
            Dictionary with market depth information
        """
        try:
            logger.info(f"Fetching market depth for {symbol}")
            
            params = {"symbol": symbol, "ohlcv_flag": "1"}
            response = self.fyers.depth(params)
            
            # Check for successful response
            if response.get("s") == "ok" and "d" in response:
                return response["d"]
            else:
                logger.warning(f"Unexpected response format for market depth: {response}")
                return {}
            
        except Exception as e:
            logger.error(f"Error fetching market depth: {str(e)}")
            raise
            
    def download_bulk_data(
        self,
        symbols: List[str],
        resolution: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols and optionally save to disk.
        
        Args:
            symbols: List of symbols to download data for
            resolution: Timeframe for the data
            start_date: Start date for historical data
            end_date: End date for historical data (default: today)
            save_path: Directory path to save data (optional)
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        for symbol in symbols:
            try:
                df = self.get_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    date_from=start_date,
                    date_to=end_date
                )
                
                result[symbol] = df
                
                # Save to disk if path provided
                if save_path and not df.empty:
                    symbol_filename = symbol.replace(":", "_").replace("-", "_")
                    file_path = f"{save_path}/{symbol_filename}_{resolution}.csv"
                    df.to_csv(file_path)
                    logger.info(f"Saved data for {symbol} to {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to download data for {symbol}: {str(e)}")
                
        return result