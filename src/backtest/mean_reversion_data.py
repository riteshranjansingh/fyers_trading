"""
Data handling module for mean reversion strategy.

This module handles:
1. Loading 15-minute intraday data from FYERS API
2. Calculating technical indicators (ADX, Bollinger Bands, RSI, etc.)
3. Processing and cleaning data for backtesting
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import percentileofscore

# Configure logging
logger = logging.getLogger(__name__)

class MeanReversionData:
    """
    Data handler for mean reversion backtesting.
    Handles data loading, preprocessing, and indicator calculation.
    """
    
    def __init__(self, connection=None):
        """
        Initialize the data handler.
        
        Args:
            connection: FyersConnection instance
        """
        self.connection = connection
        self.data = {}  # Symbol -> DataFrame
        self.timeframe = "15"  # 15-minute data
    
    def load_intraday_data(
        self, 
        symbol: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        data_fetcher=None
    ) -> pd.DataFrame:
        """
        Load intraday data for a symbol, handling FYERS API limitations.
        
        Args:
            symbol: Symbol to load data for
            start_date: Start date for data
            end_date: End date for data (default: today)
            data_fetcher: FyersDataFetcher instance
            
        Returns:
            DataFrame with intraday data
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Convert string dates to datetime
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        logger.info(f"Loading intraday data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Check if we have a data fetcher
        if data_fetcher is None:
            if self.connection is None:
                raise ValueError("No connection provided. Cannot fetch data.")
                
            from ..api.data import FyersDataFetcher
            data_fetcher = FyersDataFetcher(self.connection)
        
        # FYERS limits data to 100 days per request, so we need to make multiple requests
        all_data = []
        current_end = end_date
        
        while current_end >= start_date:
            # Calculate chunk start date (max 100 days)
            chunk_start = max(start_date, current_end - timedelta(days=99))
            
            logger.info(f"Fetching chunk from {chunk_start.date()} to {current_end.date()}")
            
            try:
                # Fetch data for this chunk
                chunk_data = data_fetcher.get_historical_data(
                    symbol=symbol,
                    resolution=self.timeframe + "m",  # e.g., "15m"
                    date_from=chunk_start,
                    date_to=current_end
                )
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    logger.info(f"Retrieved {len(chunk_data)} bars for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol} from {chunk_start.date()} to {current_end.date()}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
            # Move to the next chunk
            current_end = chunk_start - timedelta(days=1)
        
        # Combine all chunks
        if all_data:
            combined_data = pd.concat(all_data)
            
            # Sort by timestamp
            combined_data = combined_data.sort_index()
            
            # Remove duplicates
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            
            # Filter to trading hours only (9:15 AM to 3:30 PM)
            combined_data = self._filter_trading_hours(combined_data)
            
            self.data[symbol] = combined_data
            return combined_data
        else:
            logger.error(f"No data loaded for {symbol}")
            return pd.DataFrame()
    
    def _filter_trading_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to only include trading hours.
        
        Args:
            data: DataFrame with time-indexed data
            
        Returns:
            Filtered DataFrame
        """
        # Extract hour and minute
        hour = data.index.hour
        minute = data.index.minute
        
        # Create time as float (hour.minute)
        time_float = hour + minute / 100
        
        # Filter for trading hours (9:15 AM to 3:30 PM)
        return data[(time_float >= 9.15) & (time_float <= 15.30)]
    
    def load_sample_data(self, symbol: str, years: int = 2) -> pd.DataFrame:
        """
        Generate sample 15-minute intraday data for testing.
        
        Args:
            symbol: Symbol name
            years: Number of years of data to generate
            
        Returns:
            DataFrame with sample data
        """
        # Calculate number of trading days
        num_days = years * 252  # Approx. 252 trading days per year
        
        # Calculate number of 15-minute bars per day (9:15 AM to 3:30 PM)
        bars_per_day = ((15 * 60 + 30) - (9 * 60 + 15)) // 15  # 25 bars per day
        
        # Create date range for each trading day
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Adjust to last trading day (Friday if today is weekend)
        while end_date.weekday() > 4:  # 5 and 6 are weekend
            end_date -= timedelta(days=1)
            
        dates = []
        for i in range(num_days):
            day = end_date - timedelta(days=i)
            
            # Skip weekends
            if day.weekday() >= 5:  # 5 and 6 are weekend
                continue
                
            # Add this day to our list
            dates.append(day)
        
        # Reverse to get chronological order
        dates = dates[::-1]
        
        # Create timestamps for each 15-minute bar
        timestamps = []
        for day in dates:
            for bar in range(bars_per_day):
                hour = 9 + (bar * 15 + 15) // 60
                minute = (bar * 15 + 15) % 60
                timestamps.append(day.replace(hour=hour, minute=minute))
        
        # Generate OHLCV data with realistic patterns
        n = len(timestamps)
        
        # Start with a random walk for close prices
        np.random.seed(42)  # For reproducibility
        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        
        # Add some regime changes and mean reversion patterns
        for i in range(5):
            # Add a trend
            start_idx = np.random.randint(0, n - n//5)
            length = np.random.randint(n//10, n//5)
            trend = np.linspace(0, np.random.choice([-20, 20]), length)
            close[start_idx:start_idx+length] += trend
            
            # Add mean reversion after trend
            reversion_start = start_idx + length
            reversion_length = min(length, n - reversion_start)
            if reversion_length > 0:
                reversion = np.linspace(trend[-1], 0, reversion_length)
                close[reversion_start:reversion_start+reversion_length] += reversion
        
        # Generate OHLC based on close
        daily_vol = 2.0  # Daily volatility
        bar_vol = daily_vol / np.sqrt(bars_per_day)
        
        high = close + np.random.uniform(0, bar_vol * 2, n)
        low = close - np.random.uniform(0, bar_vol * 2, n)
        open_price = low + np.random.uniform(0, (high - low), n)
        
        # Ensure high >= open, close, low and low <= open, close, high
        for i in range(n):
            high[i] = max(high[i], open_price[i], close[i])
            low[i] = min(low[i], open_price[i], close[i])
        
        # Volume with some randomness and correlation to price moves
        volume = np.random.randint(1000, 10000, n)
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=timestamps)
        
        # Store data
        self.data[symbol] = data
        
        return data
    
    def calculate_indicators(self, symbol: str) -> pd.DataFrame:
        """
        Calculate all technical indicators for a symbol.
        
        Args:
            symbol: Symbol to calculate indicators for
            
        Returns:
            DataFrame with added indicators
        """
        if symbol not in self.data or self.data[symbol].empty:
            logger.error(f"No data available for {symbol}")
            return pd.DataFrame()
            
        # Get data for this symbol
        data = self.data[symbol].copy()
        
        # 1. Calculate ADX
        data = self._calculate_adx(data)
        
        # 2. Calculate Bollinger Bands
        data = self._calculate_bollinger_bands(data)
        
        # 3. Calculate RSI
        data = self._calculate_rsi(data)
        
        # 4. Calculate Historical Volatility Percentile (proxy for IV Rank)
        data = self._calculate_hv_percentile(data)
        
        # 5. Calculate ATR
        data = self._calculate_atr(data)
        
        # 6. Volume indicators
        data = self._calculate_volume_indicators(data)
        
        # 7. Calculate gap indicators
        data = self._calculate_gap_indicators(data)
        
        # Update the stored data
        self.data[symbol] = data
        
        return data
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: DataFrame with OHLC data
            period: Period for ADX calculation
            
        Returns:
            DataFrame with ADX added
        """
        # Calculate True Range
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DM and -DM
        pdm = high - high.shift(1)
        ndm = low.shift(1) - low
        
        pdm = pd.Series(np.where((pdm > ndm) & (pdm > 0), pdm, 0), index=data.index)
        ndm = pd.Series(np.where((ndm > pdm) & (ndm > 0), ndm, 0), index=data.index)
        
        # Smooth +DM and -DM
        pdi = 100 * pdm.rolling(window=period).mean() / atr
        ndi = 100 * ndm.rolling(window=period).mean() / atr
        
        # Calculate ADX
        adx = 100 * abs(pdi - ndi) / (pdi + ndi)
        adx = adx.rolling(window=period).mean()
        
        # Add to dataframe
        data['adx'] = adx
        data['+di'] = pdi
        data['-di'] = ndi
        
        return data
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands added
        """
        # Calculate middle band (SMA)
        data['bb_middle'] = data['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        data['bb_std'] = data['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)
        
        # Calculate Bollinger Band Width (BBW)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Add flags for band touches
        data['touched_upper'] = (data['high'] >= data['bb_upper']).astype(int)
        data['touched_lower'] = (data['low'] <= data['bb_lower']).astype(int)
        
        # Add flags for closes within bands after touches
        data['close_within_after_touch'] = np.zeros(len(data))
        
        for i in range(2, len(data)):
            # Check for upper band touch followed by 2 closes within bands
            if (data['touched_upper'].iloc[i-2] == 1 and 
                data['close'].iloc[i-1] < data['bb_upper'].iloc[i-1] and
                data['close'].iloc[i] < data['bb_upper'].iloc[i]):
                data.loc[data.index[i], 'close_within_after_touch'] = 1
            
            # Check for lower band touch followed by 2 closes within bands
            if (data['touched_lower'].iloc[i-2] == 1 and 
                data['close'].iloc[i-1] > data['bb_lower'].iloc[i-1] and
                data['close'].iloc[i] > data['bb_lower'].iloc[i]):
                data.loc[data.index[i], 'close_within_after_touch'] = 1
        
        return data
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with RSI added
        """
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate RS
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Flag for RSI > 60 and price reducing
        data['rsi_high_price_down'] = ((data['rsi'] > 60) & 
                                      (data['close'] < data['close'].shift(1))).astype(int)
        
        # Flag for RSI < 40 and price increasing
        data['rsi_low_price_up'] = ((data['rsi'] < 40) & 
                                   (data['close'] > data['close'].shift(1))).astype(int)
        
        return data
    
    def _calculate_hv_percentile(self, data: pd.DataFrame, period: int = 20, lookback: int = 252) -> pd.DataFrame:
        """
        Calculate Historical Volatility Percentile as a proxy for IV Rank.
        
        Args:
            data: DataFrame with price data
            period: Period for volatility calculation
            lookback: Lookback period for percentile calculation
            
        Returns:
            DataFrame with HV percentile added
        """
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Calculate rolling standard deviation (annualized)
        data['hv'] = data['returns'].rolling(window=period).std() * np.sqrt(252 * 25)  # Approx. 25 15-min bars per day
        
        # Calculate percentile rank
        data['hv_percentile'] = 0.0
        
        # Need enough data for the lookback period
        if len(data) > lookback:
            for i in range(lookback, len(data)):
                # Get historical volatility values for lookback period
                hv_values = data['hv'].iloc[i-lookback:i].dropna().values
                
                if len(hv_values) > 0:
                    current_hv = data['hv'].iloc[i]
                    
                    # Calculate percentile
                    percentile = percentileofscore(hv_values, current_hv)
                    data.loc[data.index[i], 'hv_percentile'] = percentile
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLC data
            period: Period for ATR calculation
            
        Returns:
            DataFrame with ATR added
        """
        # Calculate True Range
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR
        data['atr'] = tr.rolling(window=period).mean()
        
        return data
    
    def _calculate_volume_indicators(self, data: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """
        Calculate volume indicators.
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for volume moving average
            
        Returns:
            DataFrame with volume indicators added
        """
        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(window=period).mean()
        
        # Flag for reducing volume (3-bar)
        data['reducing_volume'] = ((data['volume'] < data['volume'].shift(1)) & 
                                  (data['volume'].shift(1) < data['volume'].shift(2))).astype(int)
        
        return data
    
    def _calculate_gap_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap indicators and flag large gaps.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with gap indicators added
        """
        # Identify day changes
        data['date'] = data.index.date
        data['day_change'] = data['date'] != data['date'].shift(1)
        
        # Calculate gap size at day changes
        data['gap'] = 0.0
        data.loc[data['day_change'], 'gap'] = data.loc[data['day_change'], 'open'] - data['close'].shift(1)
        
        # Normalize gap by ATR
        data['gap_atr'] = data['gap'] / data['atr']
        
        # Flag for large gaps (> 2x ATR)
        data['large_gap_up'] = ((data['gap_atr'] > 2) & data['day_change']).astype(int)
        data['large_gap_down'] = ((data['gap_atr'] < -2) & data['day_change']).astype(int)
        
        # Flag for large gap and stall (no continuation in same direction after 3 bars)
        # For up gaps
        data['gap_up_stall'] = 0
        for i in range(3, len(data)):
            if data['large_gap_up'].iloc[i-3] == 1:
                # Check if price stalled (did not continue higher significantly)
                if data['high'].iloc[i] <= data['high'].iloc[i-3] * 1.005:  # Less than 0.5% higher
                    data.loc[data.index[i], 'gap_up_stall'] = 1
        
        # For down gaps
        data['gap_down_stall'] = 0
        for i in range(3, len(data)):
            if data['large_gap_down'].iloc[i-3] == 1:
                # Check if price stalled (did not continue lower significantly)
                if data['low'].iloc[i] >= data['low'].iloc[i-3] * 0.995:  # Less than 0.5% lower
                    data.loc[data.index[i], 'gap_down_stall'] = 1
        
        # Any gap stall flag
        data['gap_stall'] = np.maximum(data['gap_up_stall'], data['gap_down_stall'])
        
        return data
    
    def is_within_trading_hours(self, timestamp, start_time="10:15", end_time="14:30"):
        """
        Check if a timestamp is within specified trading hours.
        
        Args:
            timestamp: Timestamp to check
            start_time: Start time as "HH:MM"
            end_time: End time as "HH:MM"
            
        Returns:
            True if timestamp is within trading hours, False otherwise
        """
        # Parse start and end times
        start_hour, start_min = map(int, start_time.split(':'))
        end_hour, end_min = map(int, end_time.split(':'))
        
        # Convert to minutes since midnight for easier comparison
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        timestamp_minutes = timestamp.hour * 60 + timestamp.minute
        
        return start_minutes <= timestamp_minutes <= end_minutes
    
    def get_data(self, symbol: str) -> pd.DataFrame:
        """
        Get stored data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            DataFrame with data for the symbol
        """
        return self.data.get(symbol, pd.DataFrame())