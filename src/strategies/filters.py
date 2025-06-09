"""
Strategy filters for the SMC backtesting system.
These filters can be applied to the base strategy signals to improve performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class VolumeFilter:
    """
    Filter signals based on volume analysis.
    Only take trades when volume confirms the signal.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the volume filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or {}
        self.volume_ma_period = self.config.get('volume_ma_period', 20)
        self.min_volume_factor = self.config.get('min_volume_factor', 1.0)  # Minimum volume relative to average
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volume filter to signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with filtered signals
        """
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # Create a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Filter buy signals - require above average volume
        filtered_df.loc[(filtered_df['signal'] == 1) & 
                      (filtered_df['volume'] < filtered_df['volume_ma'] * self.min_volume_factor), 
                      'signal'] = 0
        
        # Filter sell signals - require above average volume
        filtered_df.loc[(filtered_df['signal'] == -1) & 
                      (filtered_df['volume'] < filtered_df['volume_ma'] * self.min_volume_factor), 
                      'signal'] = 0
        
        return filtered_df


class MultiTimeframeFilter:
    """
    Filter signals based on higher timeframe confirmation.
    Only take trades when higher timeframe trend aligns with the signal.
    """
    
    def __init__(self, config: Optional[Dict] = None, higher_tf_data: Optional[pd.DataFrame] = None):
        """
        Initialize the multi-timeframe filter.
        
        Args:
            config: Filter configuration
            higher_tf_data: Higher timeframe data (if available)
        """
        self.config = config or {}
        self.higher_tf_data = higher_tf_data
        self.higher_tf_ma_period = self.config.get('higher_tf_ma_period', 50)
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply multi-timeframe filter to signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with filtered signals
        """
        filtered_df = df.copy()
        
        # If we have actual higher TF data, use it
        if self.higher_tf_data is not None:
            # Map higher TF trend to the original timeframe
            # (Implementation depends on how you want to align the timeframes)
            pass
        else:
            # Approximate higher TF trend using longer moving averages
            filtered_df['higher_tf_ma'] = filtered_df['close'].rolling(window=self.higher_tf_ma_period).mean()
            
            # Determine trend from MA slope
            filtered_df['higher_tf_trend'] = 0
            filtered_df.loc[filtered_df['higher_tf_ma'] > filtered_df['higher_tf_ma'].shift(5), 'higher_tf_trend'] = 1
            filtered_df.loc[filtered_df['higher_tf_ma'] < filtered_df['higher_tf_ma'].shift(5), 'higher_tf_trend'] = -1
            
            # Filter buy signals - require uptrend on higher TF
            filtered_df.loc[(filtered_df['signal'] == 1) & 
                          (filtered_df['higher_tf_trend'] != 1), 
                          'signal'] = 0
            
            # Filter sell signals - require downtrend on higher TF
            filtered_df.loc[(filtered_df['signal'] == -1) & 
                          (filtered_df['higher_tf_trend'] != -1), 
                          'signal'] = 0
        
        return filtered_df


class CombinedFilter:
    """
    Apply multiple filters in sequence.
    """
    
    def __init__(self, filters: List):
        """
        Initialize with a list of filters.
        
        Args:
            filters: List of filter objects
        """
        self.filters = filters
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters in sequence.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with filtered signals
        """
        filtered_df = df.copy()
        
        for filter_obj in self.filters:
            filtered_df = filter_obj.apply(filtered_df)
        
        return filtered_df


def create_filter(config: Dict, higher_tf_data: Optional[pd.DataFrame] = None):
    """
    Factory function to create filter objects based on configuration.
    
    Args:
        config: Filter configuration
        higher_tf_data: Higher timeframe data if available
        
    Returns:
        Filter object
    """
    filters = []
    
    if config.get('use_volume_filter', False):
        filters.append(VolumeFilter(config))
    
    if config.get('use_multi_timeframe_filter', False):
        filters.append(MultiTimeframeFilter(config, higher_tf_data))
    
    if config.get('use_combined_filters', False) and len(filters) > 1:
        return CombinedFilter(filters)
    elif len(filters) == 1:
        return filters[0]
    else:
        # Return a pass-through filter that doesn't change anything
        return type('PassThroughFilter', (), {'apply': lambda self, df: df})()