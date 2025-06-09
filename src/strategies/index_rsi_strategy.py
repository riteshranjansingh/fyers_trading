"""
Index RSI Strategy implementation.

This strategy buys NIFTYBEES when the Nifty 50 index RSI drops below a threshold
and sells all accumulated positions when the RSI rises above another threshold.
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

class IndexRsiStrategy(Strategy):
    """
    Index RSI Strategy.
    
    This strategy monitors the RSI of an index (e.g., Nifty 50) and trades
    an ETF (e.g., NIFTYBEES) based on RSI thresholds.
    
    Parameters:
    - index_symbol: Symbol for the index to monitor RSI (e.g., "NSE:NIFTY-INDEX")
    - trading_symbol: Symbol for the ETF to trade (e.g., "NSE:NIFTYBEES-EQ")
    - rsi_period: Period for RSI calculation
    - oversold_threshold: RSI threshold for buying (buy when RSI < threshold)
    - overbought_threshold: RSI threshold for selling (sell when RSI > threshold)
    - position_size: Number of units to buy on each signal
    """
    
    def __init__(
        self,
        index_symbol: str,
        trading_symbol: str,
        parameters: Optional[Dict] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the Index RSI strategy.
        
        Args:
            index_symbol: Symbol for the index to monitor RSI
            trading_symbol: Symbol for the ETF to trade
            parameters: Strategy parameters
            name: Strategy name
        """
        # Set default parameters
        default_params = {
            "rsi_period": 14,
            "oversold_threshold": 35,
            "overbought_threshold": 70,
            "position_size": 100
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Create a symbols list with both the index and trading symbols
        symbols = [index_symbol, trading_symbol]
        
        # Initialize base class
        super().__init__(symbols, default_params, name or "Index_RSI_Strategy")
        
        # Save the specific symbols
        self.index_symbol = index_symbol
        self.trading_symbol = trading_symbol
        
        # Initialize strategy-specific attributes
        self.rsi_period = self.parameters["rsi_period"]
        self.oversold_threshold = self.parameters["oversold_threshold"]
        self.overbought_threshold = self.parameters["overbought_threshold"]
        self.position_size = self.parameters["position_size"]
        
        # Data storage for each symbol
        self.symbol_data = {symbol: {"bars": pd.DataFrame()} for symbol in symbols}
        
        # Trade tracking for analysis
        self.trade_entries = []  # Store entry details
        self.trade_exits = []    # Store exit details
        self.holding_periods = []  # Store holding periods
        self.holding_returns = []  # Store returns for each holding period
        
        # Flag to track if we're in a buy zone (RSI < oversold)
        self.in_buy_zone = False
        
        logger.info(f"Initialized {self.name} strategy with RSI period: {self.rsi_period}, "
                    f"oversold: {self.oversold_threshold}, overbought: {self.overbought_threshold}")
    
    def initialize(self) -> None:
        """Initialize the strategy after linking to the engine."""
        # Preload historical data for both symbols if available
        for symbol in self.symbols:
            hist_data = self.get_historical_bars(symbol, self.rsi_period * 3)  # Need more data for RSI calc
            if not hist_data.empty:
                self.symbol_data[symbol]["bars"] = hist_data
        
        # Calculate initial indicators if we have enough data
        if not self.symbol_data[self.index_symbol]["bars"].empty:
            self._calculate_indicators()
    
    def _calculate_rsi(self, close_prices: pd.Series) -> pd.Series:
        """
        Calculate the RSI indicator.
        
        Args:
            close_prices: Series of close prices
            
        Returns:
            Series of RSI values
        """
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate the rolling averages
        avg_gain = gains.rolling(window=self.rsi_period).mean()
        avg_loss = losses.rolling(window=self.rsi_period).mean()
        
        # Calculate the relative strength (RS)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_indicators(self) -> None:
        """Calculate strategy indicators."""
        # Get index data
        index_bars = self.symbol_data[self.index_symbol]["bars"]
        
        if len(index_bars) < self.rsi_period:
            return  # Not enough data
        
        # Calculate RSI
        index_bars["rsi"] = self._calculate_rsi(index_bars["close"])
        
        # Store the latest RSI value
        latest_rsi = index_bars["rsi"].iloc[-1]
        logger.debug(f"Latest RSI for {self.index_symbol}: {latest_rsi:.2f}")
    
    def on_bar(self, event: BarEvent) -> None:
        """
        Process a new bar event.
        
        Args:
            event: BarEvent with new price data
        """
        symbol = event.symbol
        
        # Skip if this is not one of our symbols
        if symbol not in self.symbols:
            return
        
        # Update our data with this new bar
        new_bar = pd.Series({
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume
        }, name=event.timestamp)
        
        # Add the bar to our data storage
        if self.symbol_data[symbol]["bars"].empty:
            self.symbol_data[symbol]["bars"] = pd.DataFrame([new_bar])
        else:
            self.symbol_data[symbol]["bars"] = pd.concat([
                self.symbol_data[symbol]["bars"],
                pd.DataFrame([new_bar])
            ])
        
        # If this is the index symbol, recalculate indicators
        if symbol == self.index_symbol:
            self._calculate_indicators()
            
            # Check for trading signals
            self._check_signals(event.timestamp)
    
    def _check_signals(self, timestamp: datetime) -> None:
        """
        Check for trading signals and execute trades.
        
        Args:
            timestamp: Current timestamp
        """
        # Get index data
        index_bars = self.symbol_data[self.index_symbol]["bars"]
        
        if "rsi" not in index_bars.columns or len(index_bars) < self.rsi_period:
            return  # Not enough data or RSI not calculated
        
        # Get the latest RSI
        latest_rsi = index_bars["rsi"].iloc[-1]
        
        # Get current position
        current_position = self.get_position(self.trading_symbol)
        
        # Check buy condition: RSI < oversold threshold
        if latest_rsi < self.oversold_threshold:
            # We're in the buy zone
            self.in_buy_zone = True
            
            # Buy a batch of the trading symbol
            self.place_market_order(
                symbol=self.trading_symbol,
                quantity=self.position_size,
                direction=1,  # Buy
                timestamp=timestamp
            )
            
            # Track entry for analysis
            self.trade_entries.append({
                "timestamp": timestamp,
                "symbol": self.trading_symbol,
                "quantity": self.position_size,
                "price": self.symbol_data[self.trading_symbol]["bars"]["close"].iloc[-1],
                "rsi": latest_rsi
            })
            
            logger.info(f"{self.name}: BUY signal for {self.trading_symbol} at {timestamp}, "
                       f"RSI: {latest_rsi:.2f}, buying {self.position_size} units")
            
        # Check sell condition: RSI > overbought threshold
        elif latest_rsi > self.overbought_threshold and current_position > 0:
            # We're out of the buy zone
            self.in_buy_zone = False
            
            # Sell all of the trading symbol
            self.place_market_order(
                symbol=self.trading_symbol,
                quantity=current_position,
                direction=-1,  # Sell
                timestamp=timestamp
            )
            
            # Calculate holding periods and returns for analysis
            exit_price = self.symbol_data[self.trading_symbol]["bars"]["close"].iloc[-1]
            
            for entry in self.trade_entries:
                # Calculate holding period in days
                holding_days = (timestamp - entry["timestamp"]).days
                self.holding_periods.append(holding_days)
                
                # Calculate return for this entry
                entry_return = (exit_price / entry["price"]) - 1
                self.holding_returns.append(entry_return)
            
            # Track exit for analysis
            self.trade_exits.append({
                "timestamp": timestamp,
                "symbol": self.trading_symbol,
                "quantity": current_position,
                "price": exit_price,
                "rsi": latest_rsi,
                "avg_holding_days": np.mean(self.holding_periods) if self.holding_periods else 0,
                "avg_return": np.mean(self.holding_returns) if self.holding_returns else 0
            })
            
            logger.info(f"{self.name}: SELL signal for {self.trading_symbol} at {timestamp}, "
                       f"RSI: {latest_rsi:.2f}, selling {current_position} units")
            
            # Clear entry tracking now that we've exited
            self.trade_entries = []
            self.holding_periods = []
            self.holding_returns = []
    
    def get_strategy_summary(self) -> Dict:
        """
        Get a summary of the strategy's performance.
        
        Returns:
            Dictionary with summary statistics
        """
        # Calculate average holding period across all trades
        all_holding_periods = []
        for exit_data in self.trade_exits:
            if "avg_holding_days" in exit_data:
                all_holding_periods.append(exit_data["avg_holding_days"])
        
        avg_holding_period = np.mean(all_holding_periods) if all_holding_periods else 0
        
        # Calculate average return across all trades
        all_returns = []
        for exit_data in self.trade_exits:
            if "avg_return" in exit_data:
                all_returns.append(exit_data["avg_return"])
        
        avg_return = np.mean(all_returns) if all_returns else 0
        
        # Count trades
        num_entries = len(self.trade_entries) + sum(len(exit["entries"]) for exit in self.trade_exits if "entries" in exit)
        num_exits = len(self.trade_exits)
        
        return {
            "avg_holding_period_days": avg_holding_period,
            "avg_trade_return": avg_return,
            "avg_trade_return_pct": avg_return * 100 if avg_return else 0,
            "num_entries": num_entries,
            "num_exits": num_exits,
            "total_trades": num_entries + num_exits
        }