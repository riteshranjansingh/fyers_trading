"""
Utility functions for data manipulation and preprocessing.

This module provides helper functions for:
- Calculating technical indicators
- Resampling time series data
- Data preprocessing and cleaning
- Data export and storage
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, List, Dict, Tuple

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to a price DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    result = df.copy()
    
    # Simple Moving Averages
    result['SMA_5'] = result['close'].rolling(window=5).mean()
    result['SMA_20'] = result['close'].rolling(window=20).mean()
    result['SMA_50'] = result['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    result['EMA_5'] = result['close'].ewm(span=5, adjust=False).mean()
    result['EMA_20'] = result['close'].ewm(span=20, adjust=False).mean()
    result['EMA_50'] = result['close'].ewm(span=50, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = result['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    result['BB_middle'] = result['close'].rolling(window=20).mean()
    result['BB_std'] = result['close'].rolling(window=20).std()
    result['BB_upper'] = result['BB_middle'] + (result['BB_std'] * 2)
    result['BB_lower'] = result['BB_middle'] - (result['BB_std'] * 2)
    
    # MACD
    result['MACD'] = result['close'].ewm(span=12, adjust=False).mean() - \
                     result['close'].ewm(span=26, adjust=False).mean()
    result['MACD_signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_hist'] = result['MACD'] - result['MACD_signal']
    
    # Volume indicators
    result['Volume_SMA_5'] = result['volume'].rolling(window=5).mean()
    result['Volume_SMA_20'] = result['volume'].rolling(window=20).mean()
    
    # Calculate returns
    result['daily_return'] = result['close'].pct_change()
    result['log_return'] = np.log(result['close'] / result['close'].shift(1))
    
    return result

def resample_ohlcv(
    df: pd.DataFrame, 
    timeframe: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe for resampling (e.g., '1H', '4H', '1D', '1W')
        
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
        
    # Ensure dataframe has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for resampling")
    
    # Define resampling rules
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Get only the columns we need for resampling
    ohlcv_cols = [col for col in ohlc_dict.keys() if col in df.columns]
    
    if len(ohlcv_cols) < 5:
        logger.warning(f"Missing columns for proper OHLCV resampling. Found: {ohlcv_cols}")
    
    # Filter to only include OHLCV columns
    df_ohlcv = df[ohlcv_cols]
    
    # Resample
    resampled = df_ohlcv.resample(timeframe).agg(
        {col: ohlc_dict[col] for col in ohlcv_cols}
    )
    
    return resampled

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess a DataFrame.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Handle missing values
    for col in ['open', 'high', 'low', 'close']:
        if col in result.columns:
            # Forward fill price data
            result[col] = result[col].fillna(method='ffill')
    
    if 'volume' in result.columns:
        # Fill missing volume with 0
        result['volume'] = result['volume'].fillna(0)
    
    # Drop rows that still have NaN values in essential columns
    essential_cols = [col for col in ['open', 'high', 'low', 'close'] if col in result.columns]
    if essential_cols:
        result = result.dropna(subset=essential_cols)
    
    # Convert negative volumes to zero (if any)
    if 'volume' in result.columns:
        result.loc[result['volume'] < 0, 'volume'] = 0
    
    # Remove duplicated timestamps if any
    result = result[~result.index.duplicated(keep='first')]
    
    return result

def calculate_returns(
    df: pd.DataFrame, 
    periods: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Calculate returns over various periods.
    
    Args:
        df: DataFrame with price data
        periods: List of periods to calculate returns for
        
    Returns:
        DataFrame with added return columns
    """
    if df.empty or 'close' not in df.columns:
        return df
    
    result = df.copy()
    
    # Calculate percentage returns
    for period in periods:
        result[f'return_{period}d'] = result['close'].pct_change(period)
        
    # Calculate cumulative returns
    result['cum_return'] = (1 + result['return_1d']).cumprod() - 1
    
    return result

def save_data_to_csv(
    df: pd.DataFrame,
    symbol: str,
    path: str = "./data",
    timeframe: str = "1d",
    include_indicators: bool = False
) -> str:
    """
    Save data to CSV file with organized structure.
    
    Args:
        df: DataFrame to save
        symbol: Symbol name
        path: Base path to save data
        timeframe: Timeframe of the data
        include_indicators: Whether to add technical indicators before saving
        
    Returns:
        Path to the saved file
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Clean symbol name for filename
    clean_symbol = symbol.replace(":", "_").replace("-", "_")
    
    # Add technical indicators if requested
    if include_indicators:
        df = add_technical_indicators(df)
    
    # Construct the filename
    filename = f"{clean_symbol}_{timeframe}.csv"
    filepath = os.path.join(path, filename)
    
    # Save the file
    df.to_csv(filepath)
    logger.info(f"Data saved to {filepath}")
    
    return filepath