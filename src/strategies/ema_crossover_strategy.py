"""
EMA Crossover strategy based on the crossover of specified period EMAs.
Fully configurable for different EMA periods, risk parameters, and timeframes.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

class EMACrossoverStrategy:
    """
    Strategy based on EMA Crossover.
    
    Buy Signal: When fast-period EMA crosses above slow-period EMA
    Sell Signal: When fast-period EMA crosses below slow-period EMA
    
    Configurable for any combination of EMA periods.
    """
    
    def __init__(self, 
                 fast_period=10, 
                 slow_period=20,
                 risk_percent=1.5, 
                 risk_reward_ratio=2.0,
                 position_percent=15.0):
        """
        Initialize the EMA Crossover strategy.
        
        Args:
            fast_period: Period for the fast EMA (default: 10)
            slow_period: Period for the slow EMA (default: 20)
            risk_percent: Risk as percentage of entry price
            risk_reward_ratio: Risk to Reward ratio
            position_percent: Percentage of capital to allocate per trade
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.risk_percent = risk_percent
        self.risk_reward_ratio = risk_reward_ratio
        self.position_percent = position_percent
        
    def calculate_ema_signals(self, df):
        """
        Calculate EMAs and generate crossover signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with EMAs and signals added
        """
        # Make a copy to avoid modifying the original dataframe
        result = df.copy()
        
        # Standardize column names
        # Check if columns exist and map them to expected names
        column_mapping = {}
        
        # Check for common variations of column names
        if 'high' not in result.columns:
            if 'High' in result.columns:
                column_mapping['High'] = 'high'
            elif 'h' in result.columns:
                column_mapping['h'] = 'high'
        
        if 'low' not in result.columns:
            if 'Low' in result.columns:
                column_mapping['Low'] = 'low'
            elif 'l' in result.columns:
                column_mapping['l'] = 'low'
        
        if 'close' not in result.columns:
            if 'Close' in result.columns:
                column_mapping['Close'] = 'close'
            elif 'c' in result.columns:
                column_mapping['c'] = 'close'
        
        # Rename columns if necessary
        if column_mapping:
            result = result.rename(columns=column_mapping)
        
        # Check if we have the required columns
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in result.columns:
                logger.error(f"Required column '{col}' not found in data")
                raise KeyError(f"Required column '{col}' not found in data. Available columns: {result.columns.tolist()}")
        
        # Calculate EMAs
        result[f'ema_{self.fast_period}'] = result['close'].ewm(span=self.fast_period, adjust=False).mean()
        result[f'ema_{self.slow_period}'] = result['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate crossovers
        result['ema_diff'] = result[f'ema_{self.fast_period}'] - result[f'ema_{self.slow_period}']
        result['cross_up'] = (result['ema_diff'].shift(1) <= 0) & (result['ema_diff'] > 0)
        result['cross_down'] = (result['ema_diff'].shift(1) >= 0) & (result['ema_diff'] < 0)
        
        # Generate signals
        result['signal'] = 0
        result.loc[result['cross_up'], 'signal'] = 1  # Buy signal
        result.loc[result['cross_down'], 'signal'] = -1  # Sell signal
        
        return result
    
    def set_stop_and_target(self, df):
        """
        Set stop loss and take profit levels based on percentage risk.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with stop loss and take profit levels
        """
        result = df.copy()
        
        # Initialize stop loss and take profit columns
        result['stop_loss'] = np.nan
        result['take_profit'] = np.nan
        
        # Set stops and targets for buy signals based on percentage risk
        buy_signals = result[result['signal'] == 1].index
        for i in buy_signals:
            entry_price = result.loc[i, 'close']
            stop_distance = entry_price * (self.risk_percent / 100)
            result.loc[i, 'stop_loss'] = entry_price - stop_distance
            result.loc[i, 'take_profit'] = entry_price + (stop_distance * self.risk_reward_ratio)
        
        # Set stops and targets for sell signals (short positions)
        sell_signals = result[result['signal'] == -1].index
        for i in sell_signals:
            entry_price = result.loc[i, 'close']
            stop_distance = entry_price * (self.risk_percent / 100)
            result.loc[i, 'stop_loss'] = entry_price + stop_distance
            result.loc[i, 'take_profit'] = entry_price - (stop_distance * self.risk_reward_ratio)
        
        return result
    
    def calculate_position_size(self, capital, price):
        """
        Calculate position size based on percentage of capital.
        
        Args:
            capital: Current capital
            price: Current price
            
        Returns:
            Number of units to trade (rounded down to whole units)
        """
        allocation = capital * (self.position_percent / 100)
        units = int(allocation / price)  # Whole units only
        return max(1, units)  # Ensure at least 1 unit is traded
    
    def backtest(self, df, units_initial=100):
        """
        Run backtest with adaptive position sizing and percentage-based risk management.
        
        Args:
            df: DataFrame with OHLCV data
            units_initial: Initial position in units (capital = units_initial * initial price)
            
        Returns:
            Dictionary of backtest results and performance metrics
        """
        # Calculate EMAs and generate signals
        data = self.calculate_ema_signals(df)
        
        # Set stop loss and take profit levels
        data = self.set_stop_and_target(data)
        
        # Calculate initial capital based on initial price and units
        initial_price = data.iloc[0]['close']
        initial_capital = initial_price * units_initial
        
        # Initialize portfolio tracking
        current_capital = initial_capital
        equity = [initial_capital]
        trades = []
        
        # Current position
        current_position = None
        entry_price = None
        entry_time = None
        stop_loss = None
        take_profit = None
        position_size = 0  # Units currently held
        
        # Run simulation
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_row = data.iloc[i]
            current_price = current_row['close']
            
            # Check for signal
            current_signal = current_row['signal']
            
            # If no position is open and we get a buy signal
            if current_position is None and current_signal == 1:
                # Calculate position size based on current capital
                position_size = self.calculate_position_size(current_capital, current_price)
                
                # Check if we can afford at least 1 unit
                if position_size > 0:
                    # Open long position
                    current_position = 'LONG'
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = current_row['stop_loss']
                    take_profit = current_row['take_profit']
                    
                    logger.info(f"Opening LONG position at {entry_price} on {entry_time}, " +
                                f"size: {position_size} units, capital: {current_capital:.2f}")
                
            # If no position is open and we get a sell signal
            elif current_position is None and current_signal == -1:
                # Calculate position size based on current capital
                position_size = self.calculate_position_size(current_capital, current_price)
                
                # Check if we can afford at least 1 unit
                if position_size > 0:
                    # Open short position
                    current_position = 'SHORT'
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = current_row['stop_loss']
                    take_profit = current_row['take_profit']
                    
                    logger.info(f"Opening SHORT position at {entry_price} on {entry_time}, " +
                                f"size: {position_size} units, capital: {current_capital:.2f}")
            
            # Check if we need to close existing position
            if current_position == 'LONG':
                # Check stop loss
                if current_price <= stop_loss:
                    # Stopped out
                    profit = (current_price - entry_price) * position_size
                    current_capital += profit
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'size': position_size,
                        'profit': profit,
                        'exit_reason': 'STOP',
                        'duration': current_time - entry_time,
                        'capital_after': current_capital
                    })
                    
                    # Update equity
                    equity.append(current_capital)
                    
                    # Reset position
                    current_position = None
                    entry_price = None
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                    position_size = 0
                    
                    logger.info(f"Closed LONG position at {current_price} (STOP) with profit {profit:.2f}, " +
                                f"new capital: {current_capital:.2f}")
                
                # Check take profit
                elif current_price >= take_profit:
                    # Take profit hit
                    profit = (current_price - entry_price) * position_size
                    current_capital += profit
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'size': position_size,
                        'profit': profit,
                        'exit_reason': 'TARGET',
                        'duration': current_time - entry_time,
                        'capital_after': current_capital
                    })
                    
                    # Update equity
                    equity.append(current_capital)
                    
                    # Reset position
                    current_position = None
                    entry_price = None
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                    position_size = 0
                    
                    logger.info(f"Closed LONG position at {current_price} (TARGET) with profit {profit:.2f}, " +
                                f"new capital: {current_capital:.2f}")
                
                # Check for reversal signal
                elif current_signal == -1:
                    # Exit on reversal signal
                    profit = (current_price - entry_price) * position_size
                    current_capital += profit
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'size': position_size,
                        'profit': profit,
                        'exit_reason': 'REVERSAL',
                        'duration': current_time - entry_time,
                        'capital_after': current_capital
                    })
                    
                    # Update equity
                    equity.append(current_capital)
                    
                    logger.info(f"Closed LONG position at {current_price} (REVERSAL) with profit {profit:.2f}, " +
                                f"new capital: {current_capital:.2f}")
                    
                    # Open new position in opposite direction
                    position_size = self.calculate_position_size(current_capital, current_price)
                    
                    if position_size > 0:
                        current_position = 'SHORT'
                        entry_price = current_price
                        entry_time = current_time
                        stop_loss = current_row['stop_loss']
                        take_profit = current_row['take_profit']
                        
                        logger.info(f"Opening SHORT position at {entry_price} on {entry_time}, " +
                                    f"size: {position_size} units, capital: {current_capital:.2f}")
                    else:
                        # If can't afford new position, reset
                        current_position = None
                        entry_price = None
                        entry_time = None
                        stop_loss = None
                        take_profit = None
                        position_size = 0
            
            elif current_position == 'SHORT':
                # Check stop loss
                if current_price >= stop_loss:
                    # Stopped out
                    profit = (entry_price - current_price) * position_size
                    current_capital += profit
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'size': position_size,
                        'profit': profit,
                        'exit_reason': 'STOP',
                        'duration': current_time - entry_time,
                        'capital_after': current_capital
                    })
                    
                    # Update equity
                    equity.append(current_capital)
                    
                    # Reset position
                    current_position = None
                    entry_price = None
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                    position_size = 0
                    
                    logger.info(f"Closed SHORT position at {current_price} (STOP) with profit {profit:.2f}, " +
                                f"new capital: {current_capital:.2f}")
                
                # Check take profit
                elif current_price <= take_profit:
                    # Take profit hit
                    profit = (entry_price - current_price) * position_size
                    current_capital += profit
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'size': position_size,
                        'profit': profit,
                        'exit_reason': 'TARGET',
                        'duration': current_time - entry_time,
                        'capital_after': current_capital
                    })
                    
                    # Update equity
                    equity.append(current_capital)
                    
                    # Reset position
                    current_position = None
                    entry_price = None
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                    position_size = 0
                    
                    logger.info(f"Closed SHORT position at {current_price} (TARGET) with profit {profit:.2f}, " +
                                f"new capital: {current_capital:.2f}")
                
                # Check for reversal signal
                elif current_signal == 1:
                    # Exit on reversal signal
                    profit = (entry_price - current_price) * position_size
                    current_capital += profit
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position,
                        'size': position_size,
                        'profit': profit,
                        'exit_reason': 'REVERSAL',
                        'duration': current_time - entry_time,
                        'capital_after': current_capital
                    })
                    
                    # Update equity
                    equity.append(current_capital)
                    
                    logger.info(f"Closed SHORT position at {current_price} (REVERSAL) with profit {profit:.2f}, " +
                                f"new capital: {current_capital:.2f}")
                    
                    # Open new position in opposite direction
                    position_size = self.calculate_position_size(current_capital, current_price)
                    
                    if position_size > 0:
                        current_position = 'LONG'
                        entry_price = current_price
                        entry_time = current_time
                        stop_loss = current_row['stop_loss']
                        take_profit = current_row['take_profit']
                        
                        logger.info(f"Opening LONG position at {entry_price} on {entry_time}, " +
                                    f"size: {position_size} units, capital: {current_capital:.2f}")
                    else:
                        # If can't afford new position, reset
                        current_position = None
                        entry_price = None
                        entry_time = None
                        stop_loss = None
                        take_profit = None
                        position_size = 0
        
        # Close any open position at the end
        if current_position is not None:
            last_price = data.iloc[-1]['close']
            
            if current_position == 'LONG':
                profit = (last_price - entry_price) * position_size
            else:  # SHORT
                profit = (entry_price - last_price) * position_size
                
            current_capital += profit
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': last_price,
                'position': current_position,
                'size': position_size,
                'profit': profit,
                'exit_reason': 'END_OF_DATA',
                'duration': data.index[-1] - entry_time,
                'capital_after': current_capital
            })
            
            # Update equity
            equity.append(current_capital)
            
            logger.info(f"Closed {current_position} position at {last_price} (END_OF_DATA) with profit {profit:.2f}, " +
                        f"final capital: {current_capital:.2f}")
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        
        # Add more metrics to trades_df
        if not trades_df.empty:
            trades_df['return_pct'] = trades_df['profit'] / trades_df['entry_price'] / trades_df['size'] * 100
            trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
            trades_df['win'] = trades_df['profit'] > 0
            
            if 'duration' in trades_df.columns:
                trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
        
        # Prepare equity curve
        if trades:
            equity_curve = pd.Series(equity, index=[data.index[0]] + [t['exit_time'] for t in trades])
        else:
            equity_curve = pd.Series([initial_capital], index=[data.index[0]])
        
        # Calculate performance metrics
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = sum(1 for t in trades if t['profit'] > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades
            
            total_profit = sum(t['profit'] for t in trades)
            average_profit = total_profit / total_trades
            
            winning_profits = [t['profit'] for t in trades if t['profit'] > 0]
            losing_profits = [t['profit'] for t in trades if t['profit'] <= 0]
            
            average_win = sum(winning_profits) / len(winning_profits) if winning_profits else 0
            average_loss = sum(losing_profits) / len(losing_profits) if losing_profits else 0
            
            profit_factor = abs(sum(winning_profits) / sum(losing_profits)) if sum(losing_profits) != 0 else float('inf')
            
            # Calculate monthly returns
            if not trades_df.empty and 'exit_time' in trades_df.columns:
                # Group by month and calculate returns
                trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
                monthly_returns = trades_df.groupby('month')['profit'].sum()
                
                # Calculate drawdown
                equity_curve_df = pd.DataFrame(equity_curve, columns=['equity'])
                equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
                equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['peak']) / equity_curve_df['peak'] * 100
                max_drawdown = equity_curve_df['drawdown'].min()
            else:
                monthly_returns = pd.Series()
                max_drawdown = 0
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            total_profit = 0
            average_profit = 0
            average_win = 0
            average_loss = 0
            profit_factor = 0
            monthly_returns = pd.Series()
            max_drawdown = 0
        
        # Return results
        results = {
            'equity_curve': equity_curve,
            'trades': trades_df,
            'data': data,
            'monthly_returns': monthly_returns,
            'metrics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': average_profit,
                'average_win': average_win,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'initial_capital': initial_capital,
                'final_capital': current_capital,
                'return_pct': (current_capital - initial_capital) / initial_capital * 100,
                'max_drawdown': max_drawdown
            }
        }
        
        return results