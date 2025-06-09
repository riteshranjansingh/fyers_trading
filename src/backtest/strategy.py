"""
Base strategy module for implementing trading strategies.

This module provides the base Strategy class that all strategies should inherit from.
It defines the interface and common functionality for strategies used in backtesting.
"""

import uuid
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from abc import ABC, abstractmethod

# Import engine components
from .engine import (
    Event, BarEvent, SignalEvent, OrderEvent, FillEvent, MarketEvent,
    EventType, OrderType
)

# Configure logging
logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement
    to be compatible with the backtesting engine.
    """
    
    def __init__(
        self,
        symbols: List[str],
        parameters: Optional[Dict] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the strategy.
        
        Args:
            symbols: List of symbols this strategy will trade
            parameters: Dictionary of strategy parameters
            name: Name of the strategy (defaults to class name)
        """
        self.symbols = symbols
        self.parameters = parameters or {}
        self.name = name or self.__class__.__name__
        self.id = str(uuid.uuid4())  # Unique identifier for this strategy instance
        
        # Will be set when registered with the engine
        self.engine = None
        
        # Containers for strategy state
        self.positions = {}  # Current positions by symbol
        self.orders = {}  # Orders by ID
        self.data = {}  # Data by symbol
        
        # Performance tracking
        self.trades = []
        self.signals = []
        
        logger.info(f"Initialized strategy {self.name} with ID {self.id}")
    
    def set_engine(self, engine) -> None:
        """
        Set the backtest engine for this strategy.
        
        Args:
            engine: BacktestEngine instance
        """
        self.engine = engine
        self.initialize()
        logger.info(f"Strategy {self.name} linked to backtesting engine")
    
    def initialize(self) -> None:
        """
        Initialize the strategy after it's linked to the engine.
        
        This method is called after the strategy is registered with the engine,
        but before the backtest begins. Override this to set up any initial state.
        """
        pass
    
    @abstractmethod
    def on_bar(self, event: BarEvent) -> None:
        """
        Handle a new bar event.
        
        This is the main method where strategy logic should be implemented.
        It's called for each new bar/candle of data.
        
        Args:
            event: BarEvent with new price data
        """
        pass
    
    def on_signal(self, event: SignalEvent) -> None:
        """
        Handle a signal event.
        
        This method is called when a signal is generated (typically by this strategy).
        Override this to implement signal processing logic.
        
        Args:
            event: SignalEvent with trading signal
        """
        # Default implementation simply logs the signal
        logger.info(f"Strategy {self.name} received signal: {event}")
    
    def on_fill(self, event: FillEvent) -> None:
        """
        Handle a fill event.
        
        This method is called when an order is filled.
        Override this to implement fill processing logic.
        
        Args:
            event: FillEvent with fill information
        """
        # Default implementation updates position tracking
        symbol = event.symbol
        
        # Initialize position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = 0
            
        # Update position
        if event.direction == 1:  # Buy
            self.positions[symbol] += event.quantity
        else:  # Sell
            self.positions[symbol] -= event.quantity
            
        # Record the trade
        self.trades.append({
            "timestamp": event.timestamp,
            "symbol": symbol,
            "quantity": event.quantity,
            "direction": event.direction,
            "price": event.fill_price,
            "commission": event.commission,
            "order_id": event.order_id
        })
        
        logger.info(f"Strategy {self.name} processed fill: {event}")
    
    def on_market(self, event: MarketEvent) -> None:
        """
        Handle a market event.
        
        This method is called when new market data is available.
        Override this to implement market update logic.
        
        Args:
            event: MarketEvent with market update
        """
        # Default implementation does nothing
        pass
    
    def generate_signal(
        self,
        symbol: str,
        direction: int,
        strength: float = 1.0,
        timestamp: Optional[datetime] = None
    ) -> SignalEvent:
        """
        Generate a trading signal.
        
        Args:
            symbol: Symbol to trade
            direction: 1 for long, -1 for short, 0 for exit
            strength: Signal strength between 0 and 1
            timestamp: Signal timestamp (defaults to current time)
            
        Returns:
            SignalEvent object
        """
        timestamp = timestamp or datetime.now()
        
        signal = SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strength=strength,
            strategy_id=self.id
        )
        
        # Record the signal
        self.signals.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "direction": direction,
            "strength": strength
        })
        
        # Add the signal to the engine's event queue
        if self.engine:
            self.engine.add_event(signal)
            logger.info(f"Strategy {self.name} generated signal: {signal}")
        else:
            logger.warning(f"Strategy {self.name} not connected to engine, signal not dispatched")
            
        return signal
    
    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        direction: int,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None
    ) -> OrderEvent:
        """
        Place a market order.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of units to trade
            direction: 1 for buy, -1 for sell
            timestamp: Order timestamp (defaults to current time)
            order_id: Optional order ID
            
        Returns:
            OrderEvent object
        """
        timestamp = timestamp or datetime.now()
        
        order = OrderEvent(
            timestamp=timestamp,
            symbol=symbol,
            order_type=OrderType.MARKET,
            quantity=quantity,
            direction=direction,
            order_id=order_id
        )
        
        # Record the order
        if order.order_id:
            self.orders[order.order_id] = order
        
        # Add the order to the engine's event queue
        if self.engine:
            self.engine.add_event(order)
            logger.info(f"Strategy {self.name} placed market order: {order}")
        else:
            logger.warning(f"Strategy {self.name} not connected to engine, order not dispatched")
            
        return order
    
    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        direction: int,
        limit_price: float,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None
    ) -> OrderEvent:
        """
        Place a limit order.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of units to trade
            direction: 1 for buy, -1 for sell
            limit_price: Limit price for the order
            timestamp: Order timestamp (defaults to current time)
            order_id: Optional order ID
            
        Returns:
            OrderEvent object
        """
        timestamp = timestamp or datetime.now()
        
        order = OrderEvent(
            timestamp=timestamp,
            symbol=symbol,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            direction=direction,
            limit_price=limit_price,
            order_id=order_id
        )
        
        # Record the order
        if order.order_id:
            self.orders[order.order_id] = order
        
        # Add the order to the engine's event queue
        if self.engine:
            self.engine.add_event(order)
            logger.info(f"Strategy {self.name} placed limit order: {order}")
        else:
            logger.warning(f"Strategy {self.name} not connected to engine, order not dispatched")
            
        return order
    
    def place_stop_order(
        self,
        symbol: str,
        quantity: int,
        direction: int,
        stop_price: float,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None
    ) -> OrderEvent:
        """
        Place a stop order.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of units to trade
            direction: 1 for buy, -1 for sell
            stop_price: Stop price for the order
            timestamp: Order timestamp (defaults to current time)
            order_id: Optional order ID
            
        Returns:
            OrderEvent object
        """
        timestamp = timestamp or datetime.now()
        
        order = OrderEvent(
            timestamp=timestamp,
            symbol=symbol,
            order_type=OrderType.STOP,
            quantity=quantity,
            direction=direction,
            stop_price=stop_price,
            order_id=order_id
        )
        
        # Record the order
        if order.order_id:
            self.orders[order.order_id] = order
        
        # Add the order to the engine's event queue
        if self.engine:
            self.engine.add_event(order)
            logger.info(f"Strategy {self.name} placed stop order: {order}")
        else:
            logger.warning(f"Strategy {self.name} not connected to engine, order not dispatched")
            
        return order
    
    def get_position(self, symbol: str) -> int:
        """
        Get the current position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Current position quantity (negative for short positions)
        """
        return self.positions.get(symbol, 0)
    
    def get_historical_bars(self, symbol: str, lookback: int = 20) -> pd.DataFrame:
        """
        Get historical bar data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            lookback: Number of historical bars to return
            
        Returns:
            DataFrame with historical bars
        """
        if not self.engine:
            logger.warning(f"Strategy {self.name} not connected to engine, can't get historical data")
            return pd.DataFrame()
            
        return self.engine.get_historical_bars(symbol, lookback)
    
    def get_current_bar(self, symbol: str) -> Optional[pd.Series]:
        """
        Get the current bar data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            pandas Series with bar data or None if not available
        """
        if not self.engine:
            logger.warning(f"Strategy {self.name} not connected to engine, can't get current bar")
            return None
            
        return self.engine.get_current_bar(symbol)