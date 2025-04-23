"""
Data fetching module for FYERS API v3.

This module provides tools for retrieving market data, including:
- Historical price data
- Market quotes
- Market depth information
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Tuple
from .connection import FyersConnection
from .symbol_manager import SymbolManager

logger = logging.getLogger(__name__)

class FyersDataFetcher:
    """
    Class to handle all data fetching operations with the FYERS API v3.
    
    This class is responsible for:
    - Fetching historical market data
    - Retrieving current market quotes
    - Managing market depth information
    - Data preprocessing and formatting
    """
    
    def __init__(self, connection: FyersConnection, symbol_manager: Optional[SymbolManager] = None):
        """
        Initialize the data fetcher with a FyersConnection instance.
        
        Args:
            connection: An authenticated FyersConnection instance
            symbol_manager: Optional SymbolManager instance for symbol handling
        """
        self.connection = connection
        self.fyers = connection.get_session()
        self.symbol_manager = symbol_manager or SymbolManager()
    
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
               or a partial symbol that can be resolved by the symbol manager
            resolution: Timeframe of the data (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo)
            date_from: Start date in YYYY-MM-DD format or datetime object
            date_to: End date in YYYY-MM-DD format or datetime object (default: today)
            continuous: Whether to fetch data with continuity or not
            
        Returns:
            DataFrame with historical price data (OHLCV)
        """
        # Try to resolve the symbol if it's not in the correct format
        if ':' not in symbol and self.symbol_manager:
            # Search for the symbol
            matches = self.symbol_manager.search(name=symbol, exact_match=True)
            if matches and len(matches) > 0:
                # Format the first matching symbol
                symbol = self.symbol_manager.format_symbol(matches[0])
                logger.info(f"Resolved symbol to: {symbol}")
            else:
                logger.warning(f"Could not resolve symbol: {symbol}")
                
        # Convert dates to string format if datetime objects
        if isinstance(date_from, datetime):
            date_from = date_from.strftime("%Y-%m-%d")
            
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(date_to, datetime):
            date_to = date_to.strftime("%Y-%m-%d")
            
        # Map resolution to FYERS API format
        resolution_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "1d": "D", "1w": "W", "1mo": "M"
        }
        
        if resolution not in resolution_map:
            raise ValueError(f"Invalid resolution: {resolution}. Must be one of {list(resolution_map.keys())}")
            
        # Prepare request parameters according to FYERS API docs
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
            response = self.fyers.history(data=params)  # Using 'data=' as per documentation
            
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
            # Resolve symbols if needed
            resolved_symbols = []
            for symbol in symbols:
                if ':' not in symbol and self.symbol_manager:
                    matches = self.symbol_manager.search(name=symbol, exact_match=True)
                    if matches and len(matches) > 0:
                        resolved = self.symbol_manager.format_symbol(matches[0])
                        logger.info(f"Resolved symbol '{symbol}' to: '{resolved}'")
                        resolved_symbols.append(resolved)
                    else:
                        logger.warning(f"Could not resolve symbol: {symbol}, using as-is")
                        resolved_symbols.append(symbol)
                else:
                    resolved_symbols.append(symbol)
            
            logger.info(f"Fetching market quotes for {resolved_symbols}")
            
            # Convert list to comma-separated string if needed
            symbols_str = ",".join(resolved_symbols)
            
            # Prepare request parameters according to documentation
            params = {"symbols": symbols_str}
            response = self.fyers.quotes(data=params)  # Using 'data=' as per documentation
            
            # Check for successful response
            if response.get("s") == "ok" and "d" in response:
                return response["d"]
            else:
                logger.warning(f"Unexpected response format for quotes: {response}")
                return {}
            
        except Exception as e:
            logger.error(f"Error fetching market quotes: {str(e)}")
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
            # Resolve symbol if needed
            if ':' not in symbol and self.symbol_manager:
                matches = self.symbol_manager.search(name=symbol, exact_match=True)
                if matches and len(matches) > 0:
                    symbol = self.symbol_manager.format_symbol(matches[0])
                    logger.info(f"Resolved symbol to: {symbol}")
                else:
                    logger.warning(f"Could not resolve symbol: {symbol}")
            
            logger.info(f"Fetching market depth for {symbol}")
            
            params = {"symbol": symbol, "ohlcv_flag": "1"}
            response = self.fyers.depth(data=params)  # Using 'data=' as per documentation
            
            # Check for successful response
            if response.get("s") == "ok" and "d" in response:
                return response["d"]
            else:
                logger.warning(f"Unexpected response format for market depth: {response}")
                return {}
            
        except Exception as e:
            logger.error(f"Error fetching market depth: {str(e)}")
            raise
    
    def get_extended_historical_data(
        self,
        symbol: str,
        resolution: str,
        date_from: Union[str, datetime],
        date_to: Optional[Union[str, datetime]] = None,
        continuous: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for extended periods, automatically handling API limitations.
        
        This method will automatically split requests based on FYERS API limits:
        - 100 days for minute resolutions
        - 366 days for daily resolutions
        - 30 days for seconds resolutions
        
        Args:
            symbol: Trading symbol in FYERS format
            resolution: Timeframe of the data (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo)
            date_from: Start date
            date_to: End date (default: today)
            continuous: Whether to fetch data with continuity or not
            
        Returns:
            DataFrame with combined historical price data
        """
        # Try to resolve the symbol if it's not in the correct format
        if ':' not in symbol and self.symbol_manager:
            matches = self.symbol_manager.search(name=symbol, exact_match=True)
            if matches and len(matches) > 0:
                symbol = self.symbol_manager.format_symbol(matches[0])
                logger.info(f"Resolved symbol to: {symbol}")
            else:
                logger.warning(f"Could not resolve symbol: {symbol}")
                
        # Convert dates to datetime objects if they're strings
        if isinstance(date_from, str):
            date_from = datetime.strptime(date_from, "%Y-%m-%d")
        
        if date_to is None:
            date_to = datetime.now()
        elif isinstance(date_to, str):
            date_to = datetime.strptime(date_to, "%Y-%m-%d")
        
        # Determine the maximum period per request based on resolution
        if resolution in ["1m", "5m", "15m", "30m", "1h"]:
            # For minute resolutions: maximum 100 days per request
            max_days = 100
        elif resolution in ["1d", "1w", "1mo"]:
            # For daily/weekly/monthly resolutions: maximum 366 days per request
            max_days = 366
        else:
            # For second resolutions or unknown: maximum 30 days per request
            max_days = 30
        
        # Calculate date ranges for multiple requests if needed
        all_data = []
        current_from = date_from
        
        while current_from < date_to:
            # Calculate the end date for this chunk
            current_to = current_from + timedelta(days=max_days)
            
            # Make sure we don't go past the overall end date
            if current_to > date_to:
                current_to = date_to
            
            logger.info(f"Fetching chunk from {current_from.date()} to {current_to.date()}")
            
            # Fetch data for this chunk
            chunk_data = self.get_historical_data(
                symbol=symbol,
                resolution=resolution,
                date_from=current_from,
                date_to=current_to,
                continuous=continuous
            )
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
            
            # Move to the next chunk
            current_from = current_to + timedelta(days=1)
        
        # Combine all chunks into a single DataFrame
        if all_data:
            combined_data = pd.concat(all_data)
            
            # Remove any duplicates
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            
            # Sort by timestamp
            combined_data = combined_data.sort_index()
            
            return combined_data
        else:
            logger.warning(f"No data found for {symbol} in the specified date range")
            return pd.DataFrame()
            
    def download_bulk_data(
        self,
        symbols: List[str],
        resolution: str,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        save_path: Optional[str] = None,
        use_extended: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols and optionally save to disk.
        
        Args:
            symbols: List of symbols to download data for
            resolution: Timeframe for the data
            start_date: Start date for historical data
            end_date: End date for historical data (default: today)
            save_path: Directory path to save data (optional)
            use_extended: Whether to use extended data fetching for long periods
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        result = {}
        
        for symbol in symbols:
            try:
                # Resolve symbol if needed
                if ':' not in symbol and self.symbol_manager:
                    matches = self.symbol_manager.search(name=symbol, exact_match=True)
                    if matches and len(matches) > 0:
                        resolved_symbol = self.symbol_manager.format_symbol(matches[0])
                        logger.info(f"Resolved symbol '{symbol}' to: '{resolved_symbol}'")
                        symbol = resolved_symbol
                    else:
                        logger.warning(f"Could not resolve symbol: {symbol}")
                
                # Fetch data
                if use_extended:
                    df = self.get_extended_historical_data(
                        symbol=symbol,
                        resolution=resolution,
                        date_from=start_date,
                        date_to=end_date
                    )
                else:
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