"""
Backtesting engine for FYERS trading strategies.

This module provides an event-driven backtesting framework for testing trading strategies
on historical data. It simulates orders, executions, and account management.

Main components:
- BacktestEngine: Core backtesting simulation
- Event: Base class for events in the event queue
- OrderEvent, FillEvent, etc.: Specific event types
- Portfolio: Tracks positions, balance, and performance
"""

import pandas as pd
import numpy as np
import logging
import uuid
import copy
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Callable
from collections import deque, defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events in the backtesting system."""
    BAR = "BAR"  # New bar/candle data
    ORDER = "ORDER"  # Order sent to broker
    FILL = "FILL"  # Order filled
    SIGNAL = "SIGNAL"  # Trading signal from strategy
    MARKET = "MARKET"  # Market state update


class Event:
    """Base class for all events in the event queue."""
    
    def __init__(self, event_type: EventType, timestamp: Optional[datetime] = None):
        """
        Initialize a new event.
        
        Args:
            event_type: Type of event from EventType enum
            timestamp: When the event occurred (default: current time)
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"{self.event_type.value} Event at {self.timestamp}"


class MarketEvent(Event):
    """Event signaling new market data available."""
    
    def __init__(self, timestamp: datetime, symbols: Optional[List[str]] = None):
        """
        Initialize a market event.
        
        Args:
            timestamp: When the event occurred
            symbols: List of symbols with updated data
        """
        super().__init__(EventType.MARKET, timestamp)
        self.symbols = symbols or []
    
    def __str__(self) -> str:
        """String representation of the market event."""
        return f"Market Event at {self.timestamp} for {len(self.symbols)} symbols"


class BarEvent(Event):
    """Event containing bar (OHLCV) data for a symbol."""
    
    def __init__(
        self, 
        symbol: str, 
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float
    ):
        """
        Initialize a bar event.
        
        Args:
            symbol: Trading symbol this bar belongs to
            timestamp: Bar timestamp
            open_price: Opening price
            high_price: Highest price
            low_price: Lowest price
            close_price: Closing price
            volume: Trading volume
        """
        super().__init__(EventType.BAR, timestamp)
        self.symbol = symbol
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume
    
    def __str__(self) -> str:
        """String representation of the bar event."""
        return (f"Bar Event: {self.symbol} at {self.timestamp} "
                f"OHLC: [{self.open}, {self.high}, {self.low}, {self.close}] "
                f"Volume: {self.volume}")


class SignalEvent(Event):
    """Event containing a trading signal from a strategy."""
    
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        direction: int,  # 1 for long, -1 for short, 0 for exit
        strength: float = 1.0,
        strategy_id: Optional[str] = None
    ):
        """
        Initialize a signal event.
        
        Args:
            timestamp: When the signal was generated
            symbol: Trading symbol this signal is for
            direction: 1 for long, -1 for short, 0 for exit
            strength: Signal strength between 0 and 1
            strategy_id: ID of the strategy generating this signal
        """
        super().__init__(EventType.SIGNAL, timestamp)
        self.symbol = symbol
        self.direction = direction
        self.strength = strength
        self.strategy_id = strategy_id
    
    def __str__(self) -> str:
        """String representation of the signal event."""
        dir_str = "LONG" if self.direction == 1 else "SHORT" if self.direction == -1 else "EXIT"
        return (f"Signal Event: {self.symbol} {dir_str} (strength: {self.strength}) "
                f"at {self.timestamp}")


class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = "MARKET"  # Execute immediately at market price
    LIMIT = "LIMIT"  # Execute at specified price or better
    STOP = "STOP"  # Convert to market order when price reaches stop price
    STOP_LIMIT = "STOP_LIMIT"  # Convert to limit order when price reaches stop price


class OrderEvent(Event):
    """Event for placing an order."""
    
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        order_type: OrderType,
        quantity: int,
        direction: int,  # 1 for buy, -1 for sell
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        """
        Initialize an order event.
        
        Args:
            timestamp: When the order was created
            symbol: Trading symbol for the order
            order_type: Type of order from OrderType enum
            quantity: Number of units to buy/sell
            direction: 1 for buy, -1 for sell
            limit_price: Price for limit orders
            stop_price: Price for stop orders
            order_id: Unique order identifier (generated if None)
        """
        super().__init__(EventType.ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.order_id = order_id or str(uuid.uuid4())
    
    def __str__(self) -> str:
        """String representation of the order event."""
        dir_str = "BUY" if self.direction == 1 else "SELL"
        return (f"Order Event: {dir_str} {self.quantity} {self.symbol} "
                f"as {self.order_type.value} at {self.timestamp}")


class FillEvent(Event):
    """Event indicating an order has been filled."""
    
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        quantity: int,
        direction: int,  # 1 for buy, -1 for sell
        fill_price: float,
        commission: float,
        order_id: str
    ):
        """
        Initialize a fill event.
        
        Args:
            timestamp: When the order was filled
            symbol: Trading symbol that was filled
            quantity: Number of units filled
            direction: 1 for buy fill, -1 for sell fill
            fill_price: Price at which the order was filled
            commission: Commission or fees paid
            order_id: ID of the order that was filled
        """
        super().__init__(EventType.FILL, timestamp)
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_price = fill_price
        self.commission = commission
        self.order_id = order_id
    
    def __str__(self) -> str:
        """String representation of the fill event."""
        dir_str = "BUY" if self.direction == 1 else "SELL"
        return (f"Fill Event: {dir_str} {self.quantity} {self.symbol} "
                f"at {self.fill_price} (commission: {self.commission}) "
                f"at {self.timestamp}")


class Position:
    """Represents a position in a single trading symbol."""
    
    def __init__(self, symbol: str):
        """
        Initialize a new position.
        
        Args:
            symbol: Trading symbol for this position
        """
        self.symbol = symbol
        self.quantity = 0
        self.avg_price = 0.0
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
        self.trades = []
        
    def update(self, fill_event: FillEvent) -> None:
        """
        Update the position based on a fill event.
        
        Args:
            fill_event: The fill event to process
        """
        if fill_event.symbol != self.symbol:
            raise ValueError(f"Fill event symbol {fill_event.symbol} doesn't match position symbol {self.symbol}")
        
        # For position sizing
        position_value = self.quantity * self.avg_price if self.quantity != 0 else 0
        fill_value = fill_event.quantity * fill_event.fill_price
        
        # Handle buy fill
        if fill_event.direction == 1:
            # Calculate the new average price
            if self.quantity >= 0:
                # Adding to a long position or opening new position
                self.avg_price = (position_value + fill_value) / (self.quantity + fill_event.quantity)
                self.quantity += fill_event.quantity
            else:
                # Reducing a short position
                if fill_event.quantity <= abs(self.quantity):
                    # Calculate realized P&L from covering part of short
                    self.realized_pnl += (self.avg_price - fill_event.fill_price) * fill_event.quantity
                    self.quantity += fill_event.quantity
                else:
                    # Closing short position and going long
                    # First calculate P&L from covering entire short
                    self.realized_pnl += (self.avg_price - fill_event.fill_price) * abs(self.quantity)
                    
                    # Then calculate new long position
                    new_quantity = fill_event.quantity - abs(self.quantity)
                    self.quantity = new_quantity
                    self.avg_price = fill_event.fill_price
        
        # Handle sell fill
        else:
            # Reducing a long position
            if self.quantity > 0:
                if fill_event.quantity <= self.quantity:
                    # Calculate realized P&L from selling part of long
                    self.realized_pnl += (fill_event.fill_price - self.avg_price) * fill_event.quantity
                    self.quantity -= fill_event.quantity
                else:
                    # Closing long position and going short
                    # First calculate P&L from selling entire long
                    self.realized_pnl += (fill_event.fill_price - self.avg_price) * self.quantity
                    
                    # Then calculate new short position
                    new_quantity = fill_event.quantity - self.quantity
                    self.quantity = -new_quantity
                    self.avg_price = fill_event.fill_price
            else:
                # Adding to a short position or opening new short
                self.avg_price = (position_value + fill_value) / (self.quantity - fill_event.quantity)
                self.quantity -= fill_event.quantity
        
        # Track trade for analysis
        self.trades.append({
            "timestamp": fill_event.timestamp,
            "quantity": fill_event.quantity,
            "direction": fill_event.direction,
            "price": fill_event.fill_price,
            "commission": fill_event.commission
        })
    
    def market_value(self, current_price: float) -> float:
        """
        Calculate the current market value of the position.
        
        Args:
            current_price: Current market price of the symbol
            
        Returns:
            Market value of the position
        """
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate the unrealized P&L of the position.
        
        Args:
            current_price: Current market price of the symbol
            
        Returns:
            Unrealized profit/loss
        """
        if self.quantity == 0:
            return 0.0
        
        if self.quantity > 0:
            # Long position
            return (current_price - self.avg_price) * self.quantity
        else:
            # Short position
            return (self.avg_price - current_price) * abs(self.quantity)
    
    def total_pnl(self, current_price: float) -> float:
        """
        Calculate the total P&L (realized + unrealized).
        
        Args:
            current_price: Current market price of the symbol
            
        Returns:
            Total profit/loss
        """
        return self.realized_pnl + self.unrealized_pnl(current_price)
    
    def __str__(self) -> str:
        """String representation of the position."""
        return (f"Position: {self.symbol}, Quantity: {self.quantity}, "
                f"Avg Price: {self.avg_price:.2f}, Realized P&L: {self.realized_pnl:.2f}")


class Portfolio:
    """
    Manages a collection of positions and tracks account performance.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Starting capital for the portfolio
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> Position
        self.equity_curve = []  # List of (timestamp, equity) tuples
        self.transactions = []  # List of all transactions
        
    def update_position(self, fill_event: FillEvent) -> None:
        """
        Update a position based on a fill event.
        
        Args:
            fill_event: Fill event to process
        """
        symbol = fill_event.symbol
        
        # Create position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        # Update position
        current_position = self.positions[symbol]
        current_position.update(fill_event)
        
        # Update cash
        transaction_value = fill_event.fill_price * fill_event.quantity
        if fill_event.direction == 1:  # Buy
            self.cash -= transaction_value + fill_event.commission
        else:  # Sell
            self.cash += transaction_value - fill_event.commission
        
        # Record transaction
        self.transactions.append({
            "timestamp": fill_event.timestamp,
            "symbol": symbol,
            "quantity": fill_event.quantity,
            "direction": fill_event.direction,
            "price": fill_event.fill_price,
            "commission": fill_event.commission,
            "cash_after": self.cash
        })
    
    def update_equity(self, timestamp: datetime, current_prices: Dict[str, float]) -> float:
        """
        Update the equity curve with current portfolio value.
        
        Args:
            timestamp: Current timestamp
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            Current total equity value
        """
        total_value = self.cash
        
        # Add value of all positions
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position.market_value(current_prices[symbol])
            else:
                logger.warning(f"No current price for {symbol}, using average price")
                total_value += position.market_value(position.avg_price)
        
        # Record point on equity curve
        self.equity_curve.append((timestamp, total_value))
        
        return total_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get a position by symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position object or None if no position exists
        """
        return self.positions.get(symbol)
    
    def get_positions_summary(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Get a summary of all current positions.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            List of position summary dictionaries
        """
        summaries = []
        
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue  # Skip closed positions
                
            current_price = current_prices.get(symbol, position.avg_price)
            
            summaries.append({
                "symbol": symbol,
                "quantity": position.quantity,
                "avg_price": position.avg_price,
                "current_price": current_price,
                "market_value": position.market_value(current_price),
                "unrealized_pnl": position.unrealized_pnl(current_price),
                "realized_pnl": position.realized_pnl,
                "total_pnl": position.total_pnl(current_price)
            })
            
        return summaries
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """
        Get the equity curve as a DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])
            
        df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        df.set_index("timestamp", inplace=True)
        
        # Add returns calculation
        df["returns"] = df["equity"].pct_change()
        df["log_returns"] = np.log(df["equity"] / df["equity"].shift(1))
        df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1
        
        return df
    
    def __str__(self) -> str:
        """String representation of the portfolio."""
        position_count = sum(1 for pos in self.positions.values() if pos.quantity != 0)
        
        if not self.equity_curve:
            equity = self.initial_capital
            change = 0
        else:
            equity = self.equity_curve[-1][1]
            change = (equity / self.initial_capital - 1) * 100
            
        return (f"Portfolio: {position_count} active positions, "
                f"Cash: ${self.cash:.2f}, "
                f"Equity: ${equity:.2f}, "
                f"Change: {change:.2f}%")


class MarketSimulator:
    """
    Simulates market behavior for backtesting, including order execution.
    """
    
    def __init__(
        self, 
        data: Dict[str, pd.DataFrame],
        slippage_model: Optional[Callable] = None,
        commission_model: Optional[Callable] = None
    ):
        """
        Initialize the market simulator.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            slippage_model: Function to calculate price slippage
            commission_model: Function to calculate trade commission
        """
        self.data = data
        self.current_bars = {}  # Current bar data for each symbol
        
        # Set default slippage and commission models if none provided
        self.slippage_model = slippage_model or self._default_slippage
        self.commission_model = commission_model or self._default_commission
    
    def _default_slippage(
        self, 
        price: float, 
        quantity: int, 
        direction: int, 
        symbol: str
    ) -> float:
        """
        Default slippage model - 0.1% of price in the direction of the trade.
        
        Args:
            price: Base price for the trade
            quantity: Number of units
            direction: Trade direction (1 for buy, -1 for sell)
            symbol: Trading symbol
            
        Returns:
            Adjusted price after slippage
        """
        slippage_pct = 0.001  # 0.1%
        slippage_amount = price * slippage_pct
        
        # For buys, price goes up; for sells, price goes down
        adjusted_price = price + (slippage_amount * direction)
        
        return adjusted_price
    
    def _default_commission(
        self, 
        price: float, 
        quantity: int, 
        symbol: str
    ) -> float:
        """
        Default commission model - 0.05% of trade value.
        
        Args:
            price: Trade price
            quantity: Number of units
            symbol: Trading symbol
            
        Returns:
            Commission amount
        """
        commission_pct = 0.0005  # 0.05%
        return price * abs(quantity) * commission_pct
    
    def process_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        """
        Process an order and generate a fill event.
        
        Args:
            order_event: Order event to process
            
        Returns:
            Fill event if order can be executed, None otherwise
        """
        symbol = order_event.symbol
        
        # Check if we have current bar data for this symbol
        if symbol not in self.current_bars:
            logger.warning(f"No current bar data for {symbol}, can't process order")
            return None
        
        current_bar = self.current_bars[symbol]
        
        # Determine execution price based on order type
        if order_event.order_type == OrderType.MARKET:
            # Market orders execute at current price with slippage
            base_price = current_bar.close
            
        elif order_event.order_type == OrderType.LIMIT:
            # Limit orders execute if price is favorable
            if order_event.direction == 1:  # Buy limit
                if current_bar.low > order_event.limit_price:
                    return None  # Price never went below limit
                base_price = min(order_event.limit_price, current_bar.close)
            else:  # Sell limit
                if current_bar.high < order_event.limit_price:
                    return None  # Price never went above limit
                base_price = max(order_event.limit_price, current_bar.close)
                
        elif order_event.order_type == OrderType.STOP:
            # Stop orders execute if price is unfavorable
            if order_event.direction == 1:  # Buy stop
                if current_bar.high < order_event.stop_price:
                    return None  # Price never reached stop
                base_price = max(order_event.stop_price, current_bar.close)
            else:  # Sell stop
                if current_bar.low > order_event.stop_price:
                    return None  # Price never reached stop
                base_price = min(order_event.stop_price, current_bar.close)
                
        else:
            logger.warning(f"Unsupported order type: {order_event.order_type}")
            return None
        
        # Apply slippage model
        fill_price = self.slippage_model(
            base_price, 
            order_event.quantity, 
            order_event.direction,
            symbol
        )
        
        # Calculate commission
        commission = self.commission_model(
            fill_price,
            order_event.quantity,
            symbol
        )
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=current_bar.name,  # Use bar timestamp as fill time
            symbol=symbol,
            quantity=order_event.quantity,
            direction=order_event.direction,
            fill_price=fill_price,
            commission=commission,
            order_id=order_event.order_id
        )
        
        return fill_event
    
    def update_bars(self, timestamp: datetime) -> Dict[str, BarEvent]:
        """
        Update the current bars for all symbols at the given timestamp.
        
        Args:
            timestamp: Current timestamp in the simulation
            
        Returns:
            Dictionary of symbol -> BarEvent
        """
        bar_events = {}
        
        for symbol, df in self.data.items():
            # Check if we have data for this timestamp
            if timestamp in df.index:
                bar = df.loc[timestamp]
                
                bar_event = BarEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=bar["open"],
                    high_price=bar["high"],
                    low_price=bar["low"],
                    close_price=bar["close"],
                    volume=bar["volume"]
                )
                
                # Update current bars
                self.current_bars[symbol] = bar
                
                # Add to return dictionary
                bar_events[symbol] = bar_event
                
        return bar_events
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get the current prices for all symbols.
        
        Returns:
            Dictionary mapping symbols to current prices
        """
        return {symbol: bar.close for symbol, bar in self.current_bars.items()}


class BacktestEngine:
    """
    Core backtesting engine that coordinates data, events, and trading logic.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        slippage_model: Optional[Callable] = None,
        commission_model: Optional[Callable] = None
    ):
        """
        Initialize the backtest engine.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            initial_capital: Starting capital
            slippage_model: Function to calculate price slippage
            commission_model: Function to calculate trade commission
        """
        self.data = data
        self.symbols = list(data.keys())
        self.events = deque()  # Event queue
        
        # Get sorted list of all timestamps across all data
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        self.timestamps = sorted(all_timestamps)
        
        if not self.timestamps:
            raise ValueError("No data timestamps found")
            
        # Create market simulator and portfolio
        self.market = MarketSimulator(
            data=data,
            slippage_model=slippage_model,
            commission_model=commission_model
        )
        self.portfolio = Portfolio(initial_capital=initial_capital)
        
        # Data storage for analysis
        self.all_holdings = []  # Snapshots of portfolio value over time
        self.all_positions = []  # Snapshots of positions over time
        self.strategy_returns = defaultdict(list)  # Strategy-specific returns
        
        # Performance tracking
        self.stats = {}  # Will hold final performance statistics
        
    def register_strategy(self, strategy) -> None:
        """
        Register a trading strategy with the backtest engine.
        
        Args:
            strategy: Strategy instance to register
        """
        strategy.set_engine(self)
        self.strategies.append(strategy)
    
    def _process_bar_events(self, timestamp: datetime) -> None:
        """
        Process new bar data for all symbols.
        
        Args:
            timestamp: Current timestamp
        """
        # Get updated bars
        bar_events = self.market.update_bars(timestamp)
        
        # Add bar events to the queue
        for symbol, bar_event in bar_events.items():
            self.events.append(bar_event)
        
        # Add a market event to signal new data available
        if bar_events:
            self.events.append(MarketEvent(timestamp, list(bar_events.keys())))
    
    def _update_portfolio_value(self, timestamp: datetime) -> None:
        """
        Update portfolio value at the current timestamp.
        
        Args:
            timestamp: Current timestamp
        """
        # Get current prices from market
        current_prices = self.market.get_current_prices()
        
        # Update equity curve
        equity = self.portfolio.update_equity(timestamp, current_prices)
        
        # Record holdings for analysis
        holdings = {
            "timestamp": timestamp,
            "cash": self.portfolio.cash,
            "total": equity,
            "returns": 0.0  # Will be calculated in post-processing
        }
        self.all_holdings.append(holdings)
        
        # Record positions for analysis
        positions_dict = {"timestamp": timestamp}
        for symbol in self.symbols:
            position = self.portfolio.get_position(symbol)
            quantity = position.quantity if position else 0
            positions_dict[symbol] = quantity
        self.all_positions.append(positions_dict)
    
    def run(self, strategies) -> Dict:
        """
        Run the backtest with the given strategies.
        
        Args:
            strategies: List of strategy instances to use
            
        Returns:
            Dictionary with backtest results
        """
        self.strategies = strategies if isinstance(strategies, list) else [strategies]
        
        # Register each strategy with the engine
        for strategy in self.strategies:
            strategy.set_engine(self)
        
        logger.info(f"Starting backtest with {len(self.strategies)} strategies on {len(self.symbols)} symbols")
        
        # Iterate through each timestamp
        for timestamp in self.timestamps:
            # Process new bar data
            self._process_bar_events(timestamp)
            
            # Process all pending events
            while self.events:
                event = self.events.popleft()
                
                # Handle different event types
                if event.event_type == EventType.BAR:
                    # Notify each strategy of the new bar
                    for strategy in self.strategies:
                        strategy.on_bar(event)
                        
                elif event.event_type == EventType.SIGNAL:
                    # Process signal by creating order(s)
                    for strategy in self.strategies:
                        if strategy.id == event.strategy_id:
                            strategy.on_signal(event)
                        
                elif event.event_type == EventType.ORDER:
                    # Process order through market simulator
                    fill_event = self.market.process_order(event)
                    if fill_event:
                        self.events.append(fill_event)
                        
                elif event.event_type == EventType.FILL:
                    # Update portfolio with fill information
                    self.portfolio.update_position(event)
                    
                    # Notify strategies of the fill
                    for strategy in self.strategies:
                        strategy.on_fill(event)
                        
                elif event.event_type == EventType.MARKET:
                    # Notify strategies of market update
                    for strategy in self.strategies:
                        strategy.on_market(event)
            
            # Update portfolio value at this timestamp
            self._update_portfolio_value(timestamp)
        
        # Calculate final performance metrics
        self._calculate_performance()
        
        return self.get_results()
    
    def _calculate_performance(self) -> None:
        """Calculate performance metrics after backtest is complete."""
        # Convert holdings to DataFrame
        holdings_df = pd.DataFrame(self.all_holdings)
        holdings_df.set_index("timestamp", inplace=True)
        
        # Calculate returns
        holdings_df["returns"] = holdings_df["total"].pct_change()
        holdings_df["log_returns"] = np.log(holdings_df["total"] / holdings_df["total"].shift(1))
        holdings_df["cumulative_returns"] = (1 + holdings_df["returns"]).cumprod() - 1
        
        # Calculate key metrics
        total_return = holdings_df["total"].iloc[-1] / holdings_df["total"].iloc[0] - 1
        
        # Annualized return
        days = (holdings_df.index[-1] - holdings_df.index[0]).days
        annual_return = (1 + total_return) ** (365 / max(1, days)) - 1
        
        # Volatility (annualized)
        daily_std = holdings_df["returns"].std()
        annual_vol = daily_std * np.sqrt(252)  # Assuming 252 trading days in a year
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        peak = holdings_df["total"].expanding(min_periods=1).max()
        drawdown = (holdings_df["total"] / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        # Store calculated statistics
        self.stats = {
            "start_date": holdings_df.index[0],
            "end_date": holdings_df.index[-1],
            "duration_days": days,
            "initial_capital": self.portfolio.initial_capital,
            "final_equity": holdings_df["total"].iloc[-1],
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annual_return": annual_return,
            "annual_return_pct": annual_return * 100,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown
        }
        
    def get_results(self) -> Dict:
        """
        Get the backtest results.
        
        Returns:
            Dictionary with backtest results and statistics
        """
        results = {
            "stats": self.stats,
            "equity_curve": self.portfolio.get_equity_curve_df(),
            "trades": self.portfolio.transactions,
            "final_positions": self.portfolio.get_positions_summary(
                self.market.get_current_prices()
            )
        }
        return results
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to the event queue.
        
        Args:
            event: Event to add
        """
        self.events.append(event)
    
    def get_current_bar(self, symbol: str) -> Optional[pd.Series]:
        """
        Get the current bar data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            pandas Series with bar data or None if not available
        """
        return self.market.current_bars.get(symbol)
    
    def get_historical_bars(self, symbol: str, lookback: int = 20) -> pd.DataFrame:
        """
        Get historical bar data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            lookback: Number of historical bars to return
            
        Returns:
            DataFrame with historical bars
        """
        if symbol not in self.data:
            return pd.DataFrame()
            
        # Find current timestamp
        current_timestamp = None
        for ts in reversed(self.timestamps):
            if ts in self.data[symbol].index:
                current_timestamp = ts
                break
                
        if current_timestamp is None:
            return pd.DataFrame()
            
        # Get historical data up to current timestamp
        symbol_data = self.data[symbol]
        current_idx = symbol_data.index.get_indexer([current_timestamp])[0]
        
        # Handle case where lookback goes beyond available data
        start_idx = max(0, current_idx - lookback + 1)
        historical_data = symbol_data.iloc[start_idx:current_idx+1]
        
        return historical_data