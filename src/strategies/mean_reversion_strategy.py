"""
Mean Reversion Strategy implementation.

This module defines two variants of the Mean Reversion Strategy:
1. Strategy 1: Using only "Must" conditions
2. Strategy 2: Using both "Must" and "Good to have" conditions
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Import the base Strategy class
from ..backtest.strategy import Strategy
from ..backtest.engine import BarEvent, EventType

# Configure logging
logger = logging.getLogger(__name__)

class MeanReversionStrategy(Strategy):
    """
    Base class for Mean Reversion strategies.
    """
    
    def __init__(
        self,
        symbol: str,
        parameters: Optional[Dict] = None,
        name: Optional[str] = None,
        include_good_to_have: bool = False
    ):
        """
        Initialize the Mean Reversion strategy.
        
        Args:
            symbol: Symbol to trade
            parameters: Strategy parameters
            name: Strategy name
            include_good_to_have: Whether to include "Good to have" conditions
        """
        # Set default parameters
        default_params = {
            "adx_threshold": 20,
            "rsi_high_threshold": 60,
            "rsi_low_threshold": 40,
            "position_size": 100,
            "hv_percentile_threshold": 50,
            "atr_gap_multiple": 2.0,
            "trailing_stop_atr": 1.5,  # Trailing stop as multiple of ATR
            "profit_target_atr": 3.0    # Profit target as multiple of ATR
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize base class
        super().__init__([symbol], default_params, name or f"MeanReversion_{symbol}")
        
        # Strategy settings
        self.symbol = symbol
        self.include_good_to_have = include_good_to_have
        
        # Extract parameters
        self.adx_threshold = self.parameters["adx_threshold"]
        self.rsi_high_threshold = self.parameters["rsi_high_threshold"]
        self.rsi_low_threshold = self.parameters["rsi_low_threshold"]
        self.position_size = self.parameters["position_size"]
        self.hv_percentile_threshold = self.parameters["hv_percentile_threshold"]
        self.atr_gap_multiple = self.parameters["atr_gap_multiple"]
        self.trailing_stop_atr = self.parameters["trailing_stop_atr"]
        self.profit_target_atr = self.parameters["profit_target_atr"]
        
        # Data tracking
        self.bars_data = pd.DataFrame()
        
        # Track entry and exit points for analysis
        self.entries = []
        self.exits = []
        
        # Track the current position's stop loss and profit target
        self.current_stop_loss = None
        self.current_profit_target = None
        
        # Trading window (10:15 AM to 2:30 PM)
        self.start_time = "10:15"
        self.end_time = "14:30"
        
        logger.info(f"Initialized {self.name} strategy with ADX threshold: {self.adx_threshold}, "
                   f"RSI thresholds: {self.rsi_low_threshold}/{self.rsi_high_threshold}")
    
    def initialize(self) -> None:
        """Initialize the strategy after linking to the engine."""
        pass
    
    def on_bar(self, event: BarEvent) -> None:
        """
        Process a new bar event.
        
        Args:
            event: BarEvent with new price data
        """
        if event.symbol != self.symbol:
            return
        
        # Store this bar
        self._update_bars_data(event)
        
        # Check for entry/exit signals
        self._check_signals(event.timestamp)
    
    def _update_bars_data(self, event: BarEvent) -> None:
        """
        Update the stored bars data with a new bar.
        
        Args:
            event: BarEvent with new price data
        """
        # Create a new row with bar data
        new_row = pd.Series({
            'open': event.open,
            'high': event.high,
            'low': event.low,
            'close': event.close,
            'volume': event.volume
        }, name=event.timestamp)
        
        # Append to existing data
        if self.bars_data.empty:
            self.bars_data = pd.DataFrame([new_row])
        else:
            self.bars_data = pd.concat([
                self.bars_data,
                pd.DataFrame([new_row])
            ])
    
    def _check_signals(self, timestamp: datetime) -> None:
        """
        Check for entry/exit signals.
        
        Args:
            timestamp: Current timestamp
        """
        # Not enough data
        if len(self.bars_data) < 20:  # Need at least 20 bars for indicators
            return
        
        # Get current position
        current_position = self.get_position(self.symbol)
        
        # Get current price and ATR
        current_close = self.bars_data['close'].iloc[-1]
        current_atr = self.bars_data['atr'].iloc[-1] if 'atr' in self.bars_data.columns else 0
        
        # Check if timestamp is within trading hours
        in_trading_window = self._is_within_trading_hours(timestamp)
        
        # Position management - first check if we're already in a position
        if current_position > 0:
            # Update stop loss based on trailing stop
            if self.current_stop_loss is not None:
                # Raise stop loss if price increases (trailing stop)
                new_stop = current_close - (current_atr * self.trailing_stop_atr)
                if new_stop > self.current_stop_loss:
                    self.current_stop_loss = new_stop
                    logger.info(f"{self.name}: Updated trailing stop to {self.current_stop_loss:.2f}")
            
            # Check if stop loss is hit
            if self.current_stop_loss is not None and current_close < self.current_stop_loss:
                self._exit_position(timestamp, "Stop Loss")
                return
            
            # Check if profit target is hit
            if self.current_profit_target is not None and current_close > self.current_profit_target:
                self._exit_position(timestamp, "Profit Target")
                return
        
        # Check for entry condition (if we have no position)
        if current_position == 0 and in_trading_window:
            entry_signal = self._entry_signal()
            
            if entry_signal:
                self._enter_position(timestamp)
    
    def _entry_signal(self) -> bool:
        """
        Check if entry conditions are met.
        Subclasses should override this method.
        
        Returns:
            True if entry conditions are met, False otherwise
        """
        raise NotImplementedError("Subclasses must implement _entry_signal()")
    
    def _enter_position(self, timestamp: datetime) -> None:
        """
        Enter a position.
        
        Args:
            timestamp: Current timestamp
        """
        # Place a market order to enter the position
        self.place_market_order(
            symbol=self.symbol,
            quantity=self.position_size,
            direction=1,  # Buy
            timestamp=timestamp
        )
        
        # Record entry for analysis
        entry_price = self.bars_data['close'].iloc[-1]
        current_atr = self.bars_data['atr'].iloc[-1] if 'atr' in self.bars_data.columns else 0
        
        entry_data = {
            "timestamp": timestamp,
            "price": entry_price,
            "quantity": self.position_size,
            "adx": self.bars_data['adx'].iloc[-1] if 'adx' in self.bars_data.columns else 0,
            "rsi": self.bars_data['rsi'].iloc[-1] if 'rsi' in self.bars_data.columns else 0
        }
        
        self.entries.append(entry_data)
        
        # Set stop loss and profit target
        self.current_stop_loss = entry_price - (current_atr * self.trailing_stop_atr)
        self.current_profit_target = entry_price + (current_atr * self.profit_target_atr)
        
        logger.info(f"{self.name}: Entered position at {entry_price:.2f}, stop: {self.current_stop_loss:.2f}, "
                   f"target: {self.current_profit_target:.2f}")
    
    def _exit_position(self, timestamp: datetime, reason: str) -> None:
        """
        Exit the current position.
        
        Args:
            timestamp: Current timestamp
            reason: Reason for exit
        """
        current_position = self.get_position(self.symbol)
        
        if current_position <= 0:
            return
        
        # Place a market order to exit the position
        self.place_market_order(
            symbol=self.symbol,
            quantity=current_position,
            direction=-1,  # Sell
            timestamp=timestamp
        )
        
        # Record exit for analysis
        exit_price = self.bars_data['close'].iloc[-1]
        
        exit_data = {
            "timestamp": timestamp,
            "price": exit_price,
            "quantity": current_position,
            "reason": reason
        }
        
        self.exits.append(exit_data)
        
        # Reset stop loss and profit target
        self.current_stop_loss = None
        self.current_profit_target = None
        
        logger.info(f"{self.name}: Exited position at {exit_price:.2f}, reason: {reason}")
    
    def _is_within_trading_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within trading hours.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if timestamp is within trading hours, False otherwise
        """
        # Parse start and end times
        start_hour, start_min = map(int, self.start_time.split(':'))
        end_hour, end_min = map(int, self.end_time.split(':'))
        
        # Convert to minutes since midnight for easier comparison
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        timestamp_minutes = timestamp.hour * 60 + timestamp.minute
        
        return start_minutes <= timestamp_minutes <= end_minutes
    
    def get_trade_summary(self) -> Dict:
        """
        Get a summary of trades.
        
        Returns:
            Dictionary with trade summary statistics
        """
        if not self.entries or not self.exits:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_winning_profit": 0,
                "avg_losing_loss": 0,
                "profit_factor": 0,
                "total_profit": 0
            }
        
        # Match entries with exits
        trades = []
        
        for entry in self.entries:
            # Find the next exit after this entry
            next_exits = [ex for ex in self.exits if ex["timestamp"] > entry["timestamp"]]
            
            if next_exits:
                exit_data = next_exits[0]
                
                # Calculate profit
                profit = (exit_data["price"] - entry["price"]) * entry["quantity"]
                
                # Calculate holding period
                holding_period = (exit_data["timestamp"] - entry["timestamp"]).total_seconds() / 3600  # in hours
                
                trades.append({
                    "entry_time": entry["timestamp"],
                    "exit_time": exit_data["timestamp"],
                    "entry_price": entry["price"],
                    "exit_price": exit_data["price"],
                    "quantity": entry["quantity"],
                    "profit": profit,
                    "holding_period": holding_period,
                    "exit_reason": exit_data.get("reason", "Unknown")
                })
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["profit"] > 0]
        losing_trades = [t for t in trades if t["profit"] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = sum(t["profit"] for t in trades) / total_trades if total_trades > 0 else 0
        avg_winning_profit = sum(t["profit"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_losing_loss = sum(t["profit"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        total_winning_profit = sum(t["profit"] for t in winning_trades)
        total_losing_loss = abs(sum(t["profit"] for t in losing_trades))
        
        profit_factor = total_winning_profit / total_losing_loss if total_losing_loss > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_winning_profit": avg_winning_profit,
            "avg_losing_loss": avg_losing_loss,
            "profit_factor": profit_factor,
            "total_profit": sum(t["profit"] for t in trades)
        }


class MeanReversionStrategy1(MeanReversionStrategy):
    """
    Mean Reversion Strategy 1: Uses only "Must" conditions.
    
    Must conditions:
    1. ADX < 20 and weakening
    2. Bollinger Band touch & 2 closes within band
    3. RSI > 60 and price reducing / RSI < 40 and price increasing
    4. Time window: 10:15 AM to 2:30 PM
    """
    
    def __init__(self, symbol: str, parameters: Optional[Dict] = None, name: Optional[str] = None):
        """Initialize the strategy with must-only conditions."""
        super().__init__(
            symbol=symbol,
            parameters=parameters,
            name=name or f"MeanReversion1_{symbol}",
            include_good_to_have=False
        )
    
    def _entry_signal(self) -> bool:
        """
        Check if the "Must" entry conditions are met.
        
        Returns:
            True if entry conditions are met, False otherwise
        """
        # Not enough data
        if len(self.bars_data) < 20:
            return False
        
        # Extract the latest indicators
        latest_idx = -1
        
        # 1. ADX < 20 and weakening
        adx_condition = False
        if 'adx' in self.bars_data.columns:
            current_adx = self.bars_data['adx'].iloc[latest_idx]
            prev_adx = self.bars_data['adx'].iloc[latest_idx - 1]
            adx_condition = current_adx < self.adx_threshold and current_adx < prev_adx
        
        # 2. Bollinger Band touch & 2 closes within band
        bb_condition = False
        if 'close_within_after_touch' in self.bars_data.columns:
            bb_condition = self.bars_data['close_within_after_touch'].iloc[latest_idx] == 1
        
        # 3. RSI divergence
        rsi_condition = False
        if 'rsi_high_price_down' in self.bars_data.columns and 'rsi_low_price_up' in self.bars_data.columns:
            rsi_condition = (self.bars_data['rsi_high_price_down'].iloc[latest_idx] == 1 or 
                            self.bars_data['rsi_low_price_up'].iloc[latest_idx] == 1)
        
        # All conditions must be met
        return adx_condition and bb_condition and rsi_condition
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "Mean Reversion Strategy 1 (Must conditions only)"


class MeanReversionStrategy2(MeanReversionStrategy):
    """
    Mean Reversion Strategy 2: Uses both "Must" and "Good to have" conditions.
    
    Must conditions:
    1. ADX < 20 and weakening
    2. Bollinger Band touch & 2 closes within band
    3. RSI > 60 and price reducing / RSI < 40 and price increasing
    4. Time window: 10:15 AM to 2:30 PM
    
    Good to have conditions:
    1. IV Rank > 50% (using HV percentile as proxy)
    2. Reducing volumes
    3. Gap up/down over 2x ATR and stall
    """
    
    def __init__(self, symbol: str, parameters: Optional[Dict] = None, name: Optional[str] = None):
        """Initialize the strategy with both must and good-to-have conditions."""
        super().__init__(
            symbol=symbol,
            parameters=parameters,
            name=name or f"MeanReversion2_{symbol}",
            include_good_to_have=True
        )
    
    def _entry_signal(self) -> bool:
        """
        Check if the entry conditions (Must + Good to have) are met.
        
        Returns:
            True if entry conditions are met, False otherwise
        """
        # First check if must conditions are met
        must_conditions_met = self._must_conditions_met()
        
        if not must_conditions_met:
            return False
        
        # Now check good-to-have conditions
        good_conditions_count = self._good_conditions_count()
        
        # At least 2 out of 3 good-to-have conditions should be met
        return good_conditions_count >= 2
    
    def _must_conditions_met(self) -> bool:
        """
        Check if the "Must" conditions are met.
        
        Returns:
            True if all must conditions are met, False otherwise
        """
        # Not enough data
        if len(self.bars_data) < 20:
            return False
        
        # Extract the latest indicators
        latest_idx = -1
        
        # 1. ADX < 20 and weakening
        adx_condition = False
        if 'adx' in self.bars_data.columns:
            current_adx = self.bars_data['adx'].iloc[latest_idx]
            prev_adx = self.bars_data['adx'].iloc[latest_idx - 1]
            adx_condition = current_adx < self.adx_threshold and current_adx < prev_adx
        
        # 2. Bollinger Band touch & 2 closes within band
        bb_condition = False
        if 'close_within_after_touch' in self.bars_data.columns:
            bb_condition = self.bars_data['close_within_after_touch'].iloc[latest_idx] == 1
        
        # 3. RSI divergence
        rsi_condition = False
        if 'rsi_high_price_down' in self.bars_data.columns and 'rsi_low_price_up' in self.bars_data.columns:
            rsi_condition = (self.bars_data['rsi_high_price_down'].iloc[latest_idx] == 1 or 
                            self.bars_data['rsi_low_price_up'].iloc[latest_idx] == 1)
        
        # All conditions must be met
        return adx_condition and bb_condition and rsi_condition
    
    def _good_conditions_count(self) -> int:
        """
        Count how many "Good to have" conditions are met.
        
        Returns:
            Number of good-to-have conditions met (0-3)
        """
        # Extract the latest indicators
        latest_idx = -1
        count = 0
        
        # 1. IV Rank > 50% (using HV percentile as proxy)
        if 'hv_percentile' in self.bars_data.columns:
            if self.bars_data['hv_percentile'].iloc[latest_idx] > self.hv_percentile_threshold:
                count += 1
        
        # 2. Reducing volumes
        if 'reducing_volume' in self.bars_data.columns:
            if self.bars_data['reducing_volume'].iloc[latest_idx] == 1:
                count += 1
        
        # 3. Gap up/down over 2x ATR and stall
        if 'gap_stall' in self.bars_data.columns:
            if self.bars_data['gap_stall'].iloc[latest_idx] == 1:
                count += 1
        
        return count
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return "Mean Reversion Strategy 2 (Must + Good to have conditions)"