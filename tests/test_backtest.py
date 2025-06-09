"""
Test suite for the backtesting framework.

This module tests the main components of the backtesting engine:
- Events and event processing
- Strategy execution
- Portfolio management
- Performance analysis
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the project root to the Python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the backtesting components
from src.backtest.engine import (
    BacktestEngine, Event, BarEvent, SignalEvent, OrderEvent, FillEvent, MarketEvent,
    EventType, OrderType, Portfolio, Position, MarketSimulator
)
from src.backtest.analysis import BacktestAnalyzer
from src.strategies.moving_average_crossover import MovingAverageCrossover

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBacktestEvents(unittest.TestCase):
    """Test the event classes and event processing."""
    
    def test_event_creation(self):
        """Test that events can be created correctly."""
        # Test base Event
        event = Event(EventType.MARKET)
        self.assertEqual(event.event_type, EventType.MARKET)
        
        # Test BarEvent
        timestamp = datetime.now()
        bar_event = BarEvent(
            symbol="TEST",
            timestamp=timestamp,
            open_price=100.0,
            high_price=105.0,
            low_price=99.0,
            close_price=103.0,
            volume=1000
        )
        self.assertEqual(bar_event.event_type, EventType.BAR)
        self.assertEqual(bar_event.symbol, "TEST")
        self.assertEqual(bar_event.close, 103.0)
        
        # Test SignalEvent
        signal_event = SignalEvent(
            timestamp=timestamp,
            symbol="TEST",
            direction=1,  # Buy
            strength=0.5,
            strategy_id="test_strategy"
        )
        self.assertEqual(signal_event.event_type, EventType.SIGNAL)
        self.assertEqual(signal_event.direction, 1)
        
        # Test OrderEvent
        order_event = OrderEvent(
            timestamp=timestamp,
            symbol="TEST",
            order_type=OrderType.MARKET,
            quantity=10,
            direction=1  # Buy
        )
        self.assertEqual(order_event.event_type, EventType.ORDER)
        self.assertEqual(order_event.order_type, OrderType.MARKET)
        
        # Test FillEvent
        fill_event = FillEvent(
            timestamp=timestamp,
            symbol="TEST",
            quantity=10,
            direction=1,  # Buy
            fill_price=102.5,
            commission=1.25,
            order_id=order_event.order_id
        )
        self.assertEqual(fill_event.event_type, EventType.FILL)
        self.assertEqual(fill_event.fill_price, 102.5)


class TestPosition(unittest.TestCase):
    """Test the Position class for tracking positions."""
    
    def test_position_updates(self):
        """Test that positions are updated correctly."""
        position = Position("TEST")
        
        # Initial state
        self.assertEqual(position.quantity, 0)
        self.assertEqual(position.avg_price, 0.0)
        
        # Buy 10 shares at $100
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            quantity=10,
            direction=1,  # Buy
            fill_price=100.0,
            commission=0.0,
            order_id="order1"
        )
        position.update(buy_fill)
        
        # Check position after buy
        self.assertEqual(position.quantity, 10)
        self.assertEqual(position.avg_price, 100.0)
        
        # Buy 5 more shares at $105
        buy_fill2 = FillEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            quantity=5,
            direction=1,  # Buy
            fill_price=105.0,
            commission=0.0,
            order_id="order2"
        )
        position.update(buy_fill2)
        
        # Check position after second buy
        self.assertEqual(position.quantity, 15)
        self.assertAlmostEqual(position.avg_price, 101.67, places=2)
        
        # Sell 8 shares at $110
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            quantity=8,
            direction=-1,  # Sell
            fill_price=110.0,
            commission=0.0,
            order_id="order3"
        )
        position.update(sell_fill)
        
        # Check position after sell
        self.assertEqual(position.quantity, 7)
        self.assertAlmostEqual(position.avg_price, 101.67, places=2)  # Avg price doesn't change on sell
        
        # Check P&L calculation
        self.assertAlmostEqual(position.realized_pnl, 8 * (110.0 - 101.67), places=2)
        self.assertAlmostEqual(position.unrealized_pnl(110.0), 7 * (110.0 - 101.67), places=2)
        self.assertAlmostEqual(position.total_pnl(110.0), 
                              position.realized_pnl + position.unrealized_pnl(110.0), places=2)


class TestPortfolio(unittest.TestCase):
    """Test the Portfolio class for managing positions and account balance."""
    
    def test_portfolio_updates(self):
        """Test that portfolio is updated correctly."""
        portfolio = Portfolio(initial_capital=10000.0)
        
        # Initial state
        self.assertEqual(portfolio.cash, 10000.0)
        self.assertEqual(len(portfolio.positions), 0)
        
        # Buy 10 shares of TEST at $100
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            quantity=10,
            direction=1,  # Buy
            fill_price=100.0,
            commission=5.0,
            order_id="order1"
        )
        portfolio.update_position(buy_fill)
        
        # Check portfolio after buy
        self.assertEqual(portfolio.cash, 10000.0 - (10 * 100.0) - 5.0)
        self.assertEqual(len(portfolio.positions), 1)
        self.assertEqual(portfolio.positions["TEST"].quantity, 10)
        
        # Update equity with current prices
        current_prices = {"TEST": 105.0}
        equity = portfolio.update_equity(datetime.now(), current_prices)
        
        # Check equity calculation
        expected_equity = portfolio.cash + (10 * 105.0)
        self.assertEqual(equity, expected_equity)
        
        # Sell 5 shares of TEST at $110
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            quantity=5,
            direction=-1,  # Sell
            fill_price=110.0,
            commission=5.0,
            order_id="order2"
        )
        portfolio.update_position(sell_fill)
        
        # Check portfolio after sell
        self.assertEqual(portfolio.cash, 10000.0 - (10 * 100.0) - 5.0 + (5 * 110.0) - 5.0)
        self.assertEqual(portfolio.positions["TEST"].quantity, 5)
        
        # Check position summary
        summaries = portfolio.get_positions_summary(current_prices)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["symbol"], "TEST")
        self.assertEqual(summaries[0]["quantity"], 5)


class TestBacktestEngine(unittest.TestCase):
    """Test the BacktestEngine class."""
    
    def setUp(self):
        """Set up test data for backtesting."""
        # Create test data for a single symbol
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
        
        # Create price series with a trend (for testing MA crossover)
        prices = pd.Series(np.linspace(100, 120, len(date_range)) + 
                          np.sin(np.linspace(0, 8 * np.pi, len(date_range))) * 10,
                          index=date_range)
        
        # Create OHLCV DataFrame
        data = pd.DataFrame({
            "open": prices,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices * 1.01,
            "volume": np.random.randint(1000, 10000, len(date_range))
        }, index=date_range)
        
        self.test_data = {"TEST": data}
        
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        engine = BacktestEngine(data=self.test_data, initial_capital=10000.0)
        
        # Check engine attributes
        self.assertEqual(engine.symbols, ["TEST"])
        self.assertEqual(len(engine.timestamps), len(self.test_data["TEST"]))
        
    def test_backtest_execution(self):
        """Test that a backtest can be executed."""
        engine = BacktestEngine(data=self.test_data, initial_capital=10000.0)
        
        # Create a simple strategy for testing
        strategy = MovingAverageCrossover(
            symbols=["TEST"],
            parameters={
                "fast_ma_period": 5,
                "slow_ma_period": 15,
                "ma_type": "simple",
                "position_size": 10
            }
        )
        
        # Run the backtest
        results = engine.run(strategy)
        
        # Check that results were generated
        self.assertIn("equity_curve", results)
        self.assertIn("stats", results)
        self.assertIn("trades", results)
        
        # Check that the strategy executed some trades
        self.assertGreater(len(results["trades"]), 0)
        
        # Check final equity is different from initial
        self.assertNotEqual(results["stats"]["final_equity"], 10000.0)


class TestBacktestAnalyzer(unittest.TestCase):
    """Test the BacktestAnalyzer class."""
    
    def setUp(self):
        """Set up test data for analysis."""
        # Create a dummy equity curve
        date_range = pd.date_range(start="2023-01-01", end="2023-01-31", freq="B")
        equity = pd.Series(np.linspace(10000, 12000, len(date_range)) + 
                          np.sin(np.linspace(0, 4 * np.pi, len(date_range))) * 200,
                          index=date_range)
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            "equity": equity,
            "returns": equity.pct_change(),
            "log_returns": np.log(equity / equity.shift(1)),
            "cumulative_returns": (1 + equity.pct_change()).cumprod() - 1
        })
        equity_df = equity_df.dropna()
        
        # Create dummy trades
        trades = []
        for i in range(10):
            is_buy = i % 2 == 0
            trades.append({
                "timestamp": date_range[i * 2],
                "symbol": "TEST",
                "quantity": 10,
                "direction": 1 if is_buy else -1,
                "price": 100 + i,
                "commission": 5.0,
                "order_id": f"order{i}"
            })
        
        # Create dummy results
        self.test_results = {
            "equity_curve": equity_df,
            "trades": trades,
            "stats": {
                "initial_capital": 10000.0,
                "final_equity": equity.iloc[-1],
                "total_return": (equity.iloc[-1] / equity.iloc[0]) - 1,
                "annual_return": 0.25,
                "total_return_pct": 20.0,
                "annual_volatility": 0.15,
                "sharpe_ratio": 1.67,
                "max_drawdown": -5.0
            }
        }
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        analyzer = BacktestAnalyzer(self.test_results)
        
        # Check analyzer attributes
        self.assertFalse(analyzer.equity_curve.empty)
        self.assertEqual(len(analyzer.trades), 10)
        self.assertNotEqual(len(analyzer.stats), 0)
    
    def test_analyzer_calculations(self):
        """Test that the analyzer calculates metrics correctly."""
        analyzer = BacktestAnalyzer(self.test_results)
        
        # Calculate risk metrics
        risk_metrics = analyzer.calculate_risk_metrics()
        
        # Check that metrics were calculated
        self.assertIn("max_drawdown", risk_metrics)
        self.assertIn("value_at_risk_95", risk_metrics)
        self.assertIn("calmar_ratio", risk_metrics)
        
        # Test monthly returns calculation
        monthly_returns = analyzer.calculate_monthly_returns()
        self.assertFalse(monthly_returns.empty)
        
        # Test summary generation
        summary = analyzer.get_summary()
        self.assertIn("performance", summary)
        self.assertIn("risk_metrics", summary)


if __name__ == "__main__":
    unittest.main()