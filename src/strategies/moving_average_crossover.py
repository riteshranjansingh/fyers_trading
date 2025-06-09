"""
Moving Average Crossover strategy implementation.

This strategy generates buy signals when a fast moving average crosses above
a slow moving average, and sell signals when the fast MA crosses below the slow MA.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

# Import the base Strategy class
from ..backtest.strategy import Strategy
from ..backtest.engine import BarEvent, EventType

# Configure logging
logger = logging.getLogger(__name__)

class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Parameters:
    - fast_ma_period: Period for the fast moving average
    - slow_ma_period: Period for the slow moving average
    - ma_type: Type of moving average ('simple' or 'exponential')
    - position_size: Size of each position (in currency or percentage)
    """
    
    def __init__(
        self,
        symbols: List[str],
        parameters: Optional[Dict] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the MA Crossover strategy.
        
        Args:
            symbols: List of symbols to trade
            parameters: Strategy parameters
            name: Strategy name
        """
        # Set default parameters
        default_params = {
            "fast_ma_period": 10,
            "slow_ma_period": 30,
            "ma_type": "simple",  # 'simple' or 'exponential'
            "position_size": 100,  # Number of units or percentage
            "trail_stop_pct": 0.02  # Trailing stop percentage (if enabled)
        }
        
        # Override defaults with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Initialize base class
        super().__init__(symbols, default_params, name or "MA_Crossover")
        
        # Initialize strategy-specific attributes
        self.fast_ma_period = self.parameters["fast_ma_period"]
        self.slow_ma_period = self.parameters["slow_ma_period"]
        self.ma_type = self.parameters["ma_type"]
        self.position_size = self.parameters["position_size"]
        
        # Data storage for each symbol
        self.symbol_data = {symbol: {"bars": pd.DataFrame()} for symbol in symbols}
        
        # Track last signal for each symbol to avoid duplicate signals
        self.last_signal = {symbol: 0 for symbol in symbols}  # 0 for no position, 1 for long, -1 for short
        
        logger.info(f"Initialized {self.name} strategy with fast MA: {self.fast_ma_period}, "
                   f"slow MA: {self.slow_ma_period}, type: {self.ma_type}")
    
    def initialize(self) -> None:
        """Initialize the strategy after linking to the engine."""
        for symbol in self.symbols:
            # Pre-load historical data if available
            hist_data = self.get_historical_bars(symbol, self.slow_ma_period + 10)
            if not hist_data.empty:
                self.symbol_data[symbol]["bars"] = hist_data
                self._calculate_indicators(symbol)
    
    def _calculate_indicators(self, symbol: str) -> None:
        """
        Calculate strategy indicators for a symbol.
        
        Args:
            symbol: Symbol to calculate indicators for
        """
        bars = self.symbol_data[symbol]["bars"]
        
        if len(bars) < self.slow_ma_period:
            return  # Not enough data
        
        # Calculate fast and slow moving averages
        if self.ma_type.lower() == "simple":
            bars["fast_ma"] = bars["close"].rolling(window=self.fast_ma_period).mean()
            bars["slow_ma"] = bars["close"].rolling(window=self.slow_ma_period).mean()
        else:  # exponential
            bars["fast_ma"] = bars["close"].ewm(span=self.fast_ma_period, adjust=False).mean()
            bars["slow_ma"] = bars["close"].ewm(span=self.slow_ma_period, adjust=False).mean()
        
        # Calculate crossover signal
        bars["signal"] = 0
        
        # Crossover detection
        bars["signal"] = np.where(
            (bars["fast_ma"] > bars["slow_ma"]) & 
            (bars["fast_ma"].shift(1) <= bars["slow_ma"].shift(1)),
            1,  # Buy signal
            np.where(
                (bars["fast_ma"] < bars["slow_ma"]) & 
                (bars["fast_ma"].shift(1) >= bars["slow_ma"].shift(1)),
                -1,  # Sell signal
                0  # No signal
            )
        )
    
    def on_bar(self, event: BarEvent) -> None:
        """
        Process a new bar event.
        
        Args:
            event: BarEvent with new price data
        """
        symbol = event.symbol
        
        # Skip if this is not a symbol we're trading
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
        
        # Calculate indicators
        self._calculate_indicators(symbol)
        
        # Check for trading signals
        self._check_signals(symbol, event.timestamp)
    
    def _check_signals(self, symbol: str, timestamp: datetime) -> None:
        """
        Check for trading signals and execute trades.
        
        Args:
            symbol: Symbol to check signals for
            timestamp: Current timestamp
        """
        bars = self.symbol_data[symbol]["bars"]
        
        if len(bars) < self.slow_ma_period:
            return  # Not enough data
        
        # Get the latest signal
        latest_signal = bars["signal"].iloc[-1]
        current_position = self.get_position(symbol)
        
        # If we have a buy signal and no current position
        if latest_signal == 1 and current_position <= 0:
            # Calculate position size
            quantity = self.position_size
            
            # Close any existing short position first
            if current_position < 0:
                self.place_market_order(
                    symbol=symbol,
                    quantity=abs(current_position),
                    direction=1,  # Buy to cover
                    timestamp=timestamp
                )
            
            # Then open a new long position
            self.place_market_order(
                symbol=symbol,
                quantity=quantity,
                direction=1,  # Buy
                timestamp=timestamp
            )
            
            logger.info(f"{self.name}: BUY signal for {symbol} at {timestamp}, "
                       f"opening position of {quantity} units")
            
            # Update last signal
            self.last_signal[symbol] = 1
            
        # If we have a sell signal and no current short position
        elif latest_signal == -1 and current_position >= 0:
            # Calculate position size
            quantity = self.position_size if current_position == 0 else current_position
            
            # Close any existing long position
            if current_position > 0:
                self.place_market_order(
                    symbol=symbol,
                    quantity=current_position,
                    direction=-1,  # Sell to close
                    timestamp=timestamp
                )
                
                logger.info(f"{self.name}: SELL signal for {symbol} at {timestamp}, "
                           f"closing long position of {current_position} units")
            
            # Optionally open a new short position if strategy allows it
            # Uncomment the following to add short selling:
            """
            self.place_market_order(
                symbol=symbol,
                quantity=quantity,
                direction=-1,  # Sell short
                timestamp=timestamp
            )
            
            logger.info(f"{self.name}: SELL signal for {symbol} at {timestamp}, "
                       f"opening short position of {quantity} units")
            """
            
            # Update last signal
            self.last_signal[symbol] = -1