"""
Smart Money Concepts (SMC) Strategy Implementation
Translated from PineScript to Python for backtesting with FYERS API
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class SMCStrategy:
    """
    Implementation of the Smart Money Concepts (SMC) trading strategy.
    This strategy focuses on market structure, order blocks, and fair value gaps
    to identify potential entry and exit points.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SMC strategy with configuration parameters.
        
        Args:
            config: Dictionary with strategy configuration
        """
        self.config = config
        self.ms_length = 5  # Market structure pivot length
        self.ob_length = 5  # Order block length
        
        # State variables for market structure
        self.trend = 0  # 1 for uptrend, -1 for downtrend, 0 for undefined
        self.last_pivot_high = 0.0
        self.last_pivot_low = 0.0
        self.last_bos = 0.0  # Break of structure
        self.last_choch = 0.0  # Change of character
        self.pivot_highs = []
        self.pivot_lows = []
        
        # Order blocks
        self.bullish_order_blocks = []  # Demand zones
        self.bearish_order_blocks = []  # Supply zones
        
        # Fair value gaps
        self.bullish_fvgs = []
        self.bearish_fvgs = []
        
        # Signals and positions
        self.signals = []
        self.positions = []
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the strategy by calculating required indicators.
        
        Args:
            data: OHLCV dataframe
            
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Calculate ATR for price ranges
        df['atr'] = self._calculate_atr(df, 200)
        
        # Calculate pivots for market structure
        df['pivot_high'] = self._calculate_pivot_highs(df, self.ms_length)
        df['pivot_low'] = self._calculate_pivot_lows(df, self.ms_length)
        
        # Prepare columns for market structure
        df['trend'] = 0
        df['bos'] = np.nan
        df['choch'] = np.nan
        
        # Prepare columns for order blocks
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        df['bullish_ob_top'] = np.nan
        df['bullish_ob_bottom'] = np.nan
        df['bearish_ob_top'] = np.nan
        df['bearish_ob_bottom'] = np.nan
        
        # Prepare columns for fair value gaps
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        df['bullish_fvg_top'] = np.nan
        df['bullish_fvg_bottom'] = np.nan
        df['bearish_fvg_top'] = np.nan
        df['bearish_fvg_bottom'] = np.nan
        
        return df
    
    def analyze_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze market structure to identify trends, BOS, and CHoCH points.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            DataFrame with market structure analysis
        """
        # Initialize variables for tracking market structure
        up_swing = None
        down_swing = None
        trend = 0
        
        for i in range(self.ms_length * 2, len(df)):
            # Skip if we don't have enough data
            if i < self.ms_length * 2:
                continue
                
            current_bar = df.iloc[i]
            prev_bars = df.iloc[i-self.ms_length*2:i]
            
            # Check for pivot high
            if current_bar['pivot_high']:
                # Potential change in structure if we have a pivot high
                if trend == -1 and current_bar['high'] > up_swing:
                    df.loc[df.index[i], 'bos'] = current_bar['high']
                    df.loc[df.index[i], 'trend'] = 1
                    trend = 1
                
                up_swing = current_bar['high']
            
            # Check for pivot low
            if current_bar['pivot_low']:
                # Potential change in structure if we have a pivot low
                if trend == 1 and current_bar['low'] < down_swing:
                    df.loc[df.index[i], 'bos'] = current_bar['low']
                    df.loc[df.index[i], 'trend'] = -1
                    trend = -1
                
                down_swing = current_bar['low']
            
            # Check for Change of Character (CHoCH)
            if trend == 1:
                # In uptrend, look for price making a lower low than the last pivot low
                if current_bar['low'] < down_swing and not pd.isna(down_swing):
                    df.loc[df.index[i], 'choch'] = current_bar['low']
                    df.loc[df.index[i], 'trend'] = -1
                    trend = -1
            
            elif trend == -1:
                # In downtrend, look for price making a higher high than the last pivot high
                if current_bar['high'] > up_swing and not pd.isna(up_swing):
                    df.loc[df.index[i], 'choch'] = current_bar['high']
                    df.loc[df.index[i], 'trend'] = 1
                    trend = 1
            
            # Store trend
            df.loc[df.index[i], 'trend'] = trend
        
        return df
        
    def identify_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify bullish and bearish order blocks based on market structure.
        
        Args:
            df: Dataframe with market structure analysis
            
        Returns:
            DataFrame with order blocks identified
        """
        for i in range(self.ms_length * 2, len(df)):
            current_bar = df.iloc[i]
            
            # Check for BOS or CHoCH events to identify potential order blocks
            if not pd.isna(current_bar['bos']) or not pd.isna(current_bar['choch']):
                # Look for order block for a bullish move (trend changing from down to up)
                if current_bar['trend'] == 1 and df.iloc[i-1]['trend'] == -1:
                    # Look back to find the last swing high
                    for j in range(i-1, i-10, -1):
                        if j >= 0 and df.iloc[j]['pivot_high']:
                            # Bullish order block found - typically a bearish candle before the BOS/CHoCH
                            ob_idx = j - 1
                            if ob_idx >= 0:
                                ob_bar = df.iloc[ob_idx]
                                ob_top = ob_bar['high']
                                
                                # Calculate bottom - can use ATR to adjust
                                if self.config.get('use_atr_for_ob', False):
                                    ob_bottom = ob_bar['low'] - self.ob_length * ob_bar['atr']
                                    ob_bottom = max(ob_bottom, ob_bar['low'])  # Don't go below the actual low
                                else:
                                    ob_bottom = ob_bar['low']
                                    
                                df.loc[df.index[ob_idx], 'bullish_ob'] = True
                                df.loc[df.index[ob_idx], 'bullish_ob_top'] = ob_top
                                df.loc[df.index[ob_idx], 'bullish_ob_bottom'] = ob_bottom
                            break
                
                # Look for order block for a bearish move (trend changing from up to down)
                elif current_bar['trend'] == -1 and df.iloc[i-1]['trend'] == 1:
                    # Look back to find the last swing low
                    for j in range(i-1, i-10, -1):
                        if j >= 0 and df.iloc[j]['pivot_low']:
                            # Bearish order block found - typically a bullish candle before the BOS/CHoCH
                            ob_idx = j - 1
                            if ob_idx >= 0:
                                ob_bar = df.iloc[ob_idx]
                                ob_bottom = ob_bar['low']
                                
                                # Calculate top - can use ATR to adjust
                                if self.config.get('use_atr_for_ob', False):
                                    ob_top = ob_bar['high'] + self.ob_length * ob_bar['atr']
                                    ob_top = min(ob_top, ob_bar['high'])  # Don't go above the actual high
                                else:
                                    ob_top = ob_bar['high']
                                    
                                df.loc[df.index[ob_idx], 'bearish_ob'] = True
                                df.loc[df.index[ob_idx], 'bearish_ob_top'] = ob_top
                                df.loc[df.index[ob_idx], 'bearish_ob_bottom'] = ob_bottom
                            break
        
        return df
    
    def identify_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify bullish and bearish fair value gaps.
        
        Args:
            df: Dataframe with market structure and order blocks
            
        Returns:
            DataFrame with fair value gaps identified
        """
        for i in range(2, len(df)):
            # Skip if we don't have enough data
            if i < 2:
                continue
                
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            prev_prev_bar = df.iloc[i-2]
            
            # Bullish FVG: Current bar's low is higher than previous-previous bar's high
            if current_bar['low'] > prev_prev_bar['high']:
                fvg_top = current_bar['low']
                fvg_bottom = prev_prev_bar['high']
                
                # Only consider significant gaps
                if self.config.get('fvg_threshold', 0) > 0:
                    threshold = current_bar['atr'] * self.config['fvg_threshold']
                    if (fvg_top - fvg_bottom) >= threshold:
                        df.loc[df.index[i], 'bullish_fvg'] = True
                        df.loc[df.index[i], 'bullish_fvg_top'] = fvg_top
                        df.loc[df.index[i], 'bullish_fvg_bottom'] = fvg_bottom
                else:
                    df.loc[df.index[i], 'bullish_fvg'] = True
                    df.loc[df.index[i], 'bullish_fvg_top'] = fvg_top
                    df.loc[df.index[i], 'bullish_fvg_bottom'] = fvg_bottom
            
            # Bearish FVG: Current bar's high is lower than previous-previous bar's low
            if current_bar['high'] < prev_prev_bar['low']:
                fvg_top = prev_prev_bar['low']
                fvg_bottom = current_bar['high']
                
                # Only consider significant gaps
                if self.config.get('fvg_threshold', 0) > 0:
                    threshold = current_bar['atr'] * self.config['fvg_threshold']
                    if (fvg_top - fvg_bottom) >= threshold:
                        df.loc[df.index[i], 'bearish_fvg'] = True
                        df.loc[df.index[i], 'bearish_fvg_top'] = fvg_top
                        df.loc[df.index[i], 'bearish_fvg_bottom'] = fvg_bottom
                else:
                    df.loc[df.index[i], 'bearish_fvg'] = True
                    df.loc[df.index[i], 'bearish_fvg_top'] = fvg_top
                    df.loc[df.index[i], 'bearish_fvg_bottom'] = fvg_bottom
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on SMC analysis.
        
        Args:
            df: Dataframe with SMC analysis
            
        Returns:
            DataFrame with trading signals
        """
        # Add signal columns
        df['signal'] = 0  # 1 for buy, -1 for sell, 0 for no signal
        df['signal_type'] = None
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Track active order blocks and FVGs
        active_bullish_obs = []
        active_bearish_obs = []
        active_bullish_fvgs = []
        active_bearish_fvgs = []
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            
            # Skip if we don't have enough data
            if i < self.ms_length * 2:
                continue
            
            # Add new order blocks and FVGs to active lists
            if current_bar['bullish_ob']:
                active_bullish_obs.append({
                    'index': i,
                    'top': current_bar['bullish_ob_top'],
                    'bottom': current_bar['bullish_ob_bottom'],
                    'mitigated': False
                })
            
            if current_bar['bearish_ob']:
                active_bearish_obs.append({
                    'index': i,
                    'top': current_bar['bearish_ob_top'],
                    'bottom': current_bar['bearish_ob_bottom'],
                    'mitigated': False
                })
            
            if current_bar['bullish_fvg']:
                active_bullish_fvgs.append({
                    'index': i,
                    'top': current_bar['bullish_fvg_top'],
                    'bottom': current_bar['bullish_fvg_bottom'],
                    'mitigated': False
                })
            
            if current_bar['bearish_fvg']:
                active_bearish_fvgs.append({
                    'index': i,
                    'top': current_bar['bearish_fvg_top'],
                    'bottom': current_bar['bearish_fvg_bottom'],
                    'mitigated': False
                })
            
            # Check for order block interactions
            for ob in active_bullish_obs:
                if not ob['mitigated']:
                    # Price interacting with bullish order block (demand zone)
                    if current_bar['low'] <= ob['top'] and current_bar['low'] >= ob['bottom']:
                        # Potential buy signal
                        if current_bar['trend'] == 1:  # If in uptrend
                            # Generate buy signal
                            df.loc[df.index[i], 'signal'] = 1
                            df.loc[df.index[i], 'signal_type'] = 'Bullish Order Block'
                            df.loc[df.index[i], 'stop_loss'] = ob['bottom'] - current_bar['atr'] * 0.5
                            
                            # Calculate take profit based on risk-reward ratio
                            risk = current_bar['close'] - df.loc[df.index[i], 'stop_loss']
                            reward = risk * self.config.get('risk_reward_ratio', 2.0)
                            df.loc[df.index[i], 'take_profit'] = current_bar['close'] + reward
                            
                            # Mark order block as mitigated
                            ob['mitigated'] = True
            
            for ob in active_bearish_obs:
                if not ob['mitigated']:
                    # Price interacting with bearish order block (supply zone)
                    if current_bar['high'] >= ob['bottom'] and current_bar['high'] <= ob['top']:
                        # Potential sell signal
                        if current_bar['trend'] == -1:  # If in downtrend
                            # Generate sell signal
                            df.loc[df.index[i], 'signal'] = -1
                            df.loc[df.index[i], 'signal_type'] = 'Bearish Order Block'
                            df.loc[df.index[i], 'stop_loss'] = ob['top'] + current_bar['atr'] * 0.5
                            
                            # Calculate take profit based on risk-reward ratio
                            risk = df.loc[df.index[i], 'stop_loss'] - current_bar['close']
                            reward = risk * self.config.get('risk_reward_ratio', 2.0)
                            df.loc[df.index[i], 'take_profit'] = current_bar['close'] - reward
                            
                            # Mark order block as mitigated
                            ob['mitigated'] = True
            
            # Check for FVG interactions (optional trading signals)
            if self.config.get('trade_fvg', False):
                for fvg in active_bullish_fvgs:
                    if not fvg['mitigated']:
                        # Price returning to fill bullish FVG
                        if current_bar['low'] <= fvg['top'] and current_bar['high'] >= fvg['bottom']:
                            # Potential buy signal if price reacts from the FVG
                            if current_bar['close'] > current_bar['open'] and current_bar['trend'] == 1:
                                df.loc[df.index[i], 'signal'] = 1
                                df.loc[df.index[i], 'signal_type'] = 'Bullish FVG'
                                df.loc[df.index[i], 'stop_loss'] = fvg['bottom'] - current_bar['atr'] * 0.5
                                
                                # Calculate take profit
                                risk = current_bar['close'] - df.loc[df.index[i], 'stop_loss']
                                reward = risk * self.config.get('risk_reward_ratio', 2.0)
                                df.loc[df.index[i], 'take_profit'] = current_bar['close'] + reward
                                
                                # Mark FVG as mitigated
                                fvg['mitigated'] = True
                
                for fvg in active_bearish_fvgs:
                    if not fvg['mitigated']:
                        # Price returning to fill bearish FVG
                        if current_bar['high'] >= fvg['bottom'] and current_bar['low'] <= fvg['top']:
                            # Potential sell signal if price reacts from the FVG
                            if current_bar['close'] < current_bar['open'] and current_bar['trend'] == -1:
                                df.loc[df.index[i], 'signal'] = -1
                                df.loc[df.index[i], 'signal_type'] = 'Bearish FVG'
                                df.loc[df.index[i], 'stop_loss'] = fvg['top'] + current_bar['atr'] * 0.5
                                
                                # Calculate take profit
                                risk = df.loc[df.index[i], 'stop_loss'] - current_bar['close']
                                reward = risk * self.config.get('risk_reward_ratio', 2.0)
                                df.loc[df.index[i], 'take_profit'] = current_bar['close'] - reward
                                
                                # Mark FVG as mitigated
                                fvg['mitigated'] = True
            
            # Check for BOS or CHoCH trading signals (optional)
            if self.config.get('trade_bos_choch', True):
                # BOS Buy Signal
                if not pd.isna(current_bar['bos']) and current_bar['trend'] == 1:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_type'] = 'BOS Buy'
                    # Find recent low for stop loss
                    stop_level = current_bar['low'] - current_bar['atr'] * 0.5
                    for j in range(i-1, max(0, i-10), -1):
                        if df.iloc[j]['pivot_low']:
                            stop_level = min(stop_level, df.iloc[j]['low'] - current_bar['atr'] * 0.5)
                            break
                    df.loc[df.index[i], 'stop_loss'] = stop_level
                    
                    # Calculate take profit
                    risk = current_bar['close'] - stop_level
                    reward = risk * self.config.get('risk_reward_ratio', 2.0)
                    df.loc[df.index[i], 'take_profit'] = current_bar['close'] + reward
                
                # BOS Sell Signal
                elif not pd.isna(current_bar['bos']) and current_bar['trend'] == -1:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_type'] = 'BOS Sell'
                    # Find recent high for stop loss
                    stop_level = current_bar['high'] + current_bar['atr'] * 0.5
                    for j in range(i-1, max(0, i-10), -1):
                        if df.iloc[j]['pivot_high']:
                            stop_level = max(stop_level, df.iloc[j]['high'] + current_bar['atr'] * 0.5)
                            break
                    df.loc[df.index[i], 'stop_loss'] = stop_level
                    
                    # Calculate take profit
                    risk = stop_level - current_bar['close']
                    reward = risk * self.config.get('risk_reward_ratio', 2.0)
                    df.loc[df.index[i], 'take_profit'] = current_bar['close'] - reward
                
                # CHoCH Buy Signal
                elif not pd.isna(current_bar['choch']) and current_bar['trend'] == 1:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_type'] = 'CHoCH Buy'
                    # Use the CHoCH level as stop loss
                    df.loc[df.index[i], 'stop_loss'] = current_bar['choch'] - current_bar['atr'] * 0.5
                    
                    # Calculate take profit
                    risk = current_bar['close'] - df.loc[df.index[i], 'stop_loss']
                    reward = risk * self.config.get('risk_reward_ratio', 2.0)
                    df.loc[df.index[i], 'take_profit'] = current_bar['close'] + reward
                
                # CHoCH Sell Signal
                elif not pd.isna(current_bar['choch']) and current_bar['trend'] == -1:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_type'] = 'CHoCH Sell'
                    # Use the CHoCH level as stop loss
                    df.loc[df.index[i], 'stop_loss'] = current_bar['choch'] + current_bar['atr'] * 0.5
                    
                    # Calculate take profit
                    risk = df.loc[df.index[i], 'stop_loss'] - current_bar['close']
                    reward = risk * self.config.get('risk_reward_ratio', 2.0)
                    df.loc[df.index[i], 'take_profit'] = current_bar['close'] - reward
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply strategy filters to the signals.
        
        Args:
            df: DataFrame with base signals
            
        Returns:
            DataFrame with filtered signals
        """
        if self.config['filters'].get('use_volume_filter', False):
            df = self._apply_volume_filter(df)
            
        if self.config['filters'].get('use_multi_timeframe_filter', False):
            df = self._apply_multi_timeframe_filter(df)
            
        return df
    
    def _apply_volume_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply volume filter to the signals.
        Only take trades when volume is above average.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with volume-filtered signals
        """
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Only keep signals where volume is above average
        filtered_df = df.copy()
        
        # Require volume to be above average for signal to be valid
        filtered_df.loc[filtered_df['volume'] <= filtered_df['volume_ma'], 'signal'] = 0
        
        return filtered_df
    
    def _apply_multi_timeframe_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply multi-timeframe filter to the signals.
        Only take trades when higher timeframe trend confirms.
        
        Note: This is a simplified implementation. In a real system,
        you would need to fetch and analyze data from higher timeframes.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with multi-timeframe filtered signals
        """
        # Calculate higher timeframe trend (simplified by using a longer MA)
        df['higher_tf_ma'] = df['close'].rolling(window=50).mean()
        
        # Only keep signals that align with higher timeframe trend
        filtered_df = df.copy()
        
        # For buy signals, require price to be above higher TF MA
        filtered_df.loc[(filtered_df['signal'] == 1) & 
                      (filtered_df['close'] <= filtered_df['higher_tf_ma']), 'signal'] = 0
        
        # For sell signals, require price to be below higher TF MA
        filtered_df.loc[(filtered_df['signal'] == -1) & 
                      (filtered_df['close'] >= filtered_df['higher_tf_ma']), 'signal'] = 0
        
        return filtered_df
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete SMC strategy analysis.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with analysis and signals
        """
        # Preprocess data
        df = self.preprocess_data(data)
        
        # Analyze market structure
        df = self.analyze_market_structure(df)
        
        # Identify order blocks
        df = self.identify_order_blocks(df)
        
        # Identify fair value gaps
        df = self.identify_fair_value_gaps(df)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Apply filters if enabled
        if any(self.config['filters'].values()):
            df = self.apply_filters(df)
        
        return df
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: OHLCV DataFrame
            period: ATR period
            
        Returns:
            ATR Series
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_pivot_highs(self, data: pd.DataFrame, length: int) -> pd.Series:
        """
        Calculate pivot highs for identifying swings.
        
        Args:
            data: OHLCV DataFrame
            length: Pivot length
            
        Returns:
            Series of boolean values indicating pivot highs
        """
        series = data['high']
        pivot_highs = pd.Series(False, index=series.index)
        
        for i in range(length, len(series) - length):
            if all(series.iloc[i] > series.iloc[i-length:i]) and all(series.iloc[i] > series.iloc[i+1:i+length+1]):
                pivot_highs.iloc[i] = True
                
        return pivot_highs
    
    def _calculate_pivot_lows(self, data: pd.DataFrame, length: int) -> pd.Series:
        """
        Calculate pivot lows for identifying swings.
        
        Args:
            data: OHLCV DataFrame
            length: Pivot length
            
        Returns:
            Series of boolean values indicating pivot lows
        """
        series = data['low']
        pivot_lows = pd.Series(False, index=series.index)
        
        for i in range(length, len(series) - length):
            if all(series.iloc[i] < series.iloc[i-length:i]) and all(series.iloc[i] < series.iloc[i+1:i+length+1]):
                pivot_lows.iloc[i] = True
                
        return pivot_lows