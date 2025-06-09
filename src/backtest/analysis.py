"""
Performance analysis module for backtesting results.

This module provides functions for:
- Calculating performance metrics
- Analyzing trade statistics
- Evaluating risk metrics
- Generating visualizations of backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """
    Analyzes the results of a backtest and calculates performance metrics.
    """
    
    def __init__(self, results: Dict):
        """
        Initialize the analyzer with backtest results.
        
        Args:
            results: Dictionary with backtest results from BacktestEngine
        """
        self.results = results
        self.equity_curve = results.get("equity_curve", pd.DataFrame())
        self.trades = results.get("trades", [])
        self.stats = results.get("stats", {})
        
        # Convert trades list to DataFrame for easier analysis
        if self.trades:
            self.trades_df = pd.DataFrame(self.trades)
            if "timestamp" in self.trades_df.columns:
                self.trades_df["timestamp"] = pd.to_datetime(self.trades_df["timestamp"])
                self.trades_df.set_index("timestamp", inplace=True)
        else:
            self.trades_df = pd.DataFrame()
            
        # Process trades to generate trade-level statistics
        self._process_trades()
    
    def _process_trades(self):
        """Process trades to calculate trade-level metrics."""
        if self.trades_df.empty:
            logger.warning("No trades to analyze")
            self.trade_stats = {}
            return
        
        # Group trades by symbol for analysis
        try:
            # Calculate trade P&L
            self.trades_df["trade_value"] = self.trades_df["price"] * self.trades_df["quantity"]
            self.trades_df["is_buy"] = self.trades_df["direction"] == 1
            self.trades_df["is_sell"] = self.trades_df["direction"] == -1
            
            # Calculate more metrics in a separate method
            self._calculate_trade_metrics()
        
        except Exception as e:
            logger.error(f"Error processing trades: {str(e)}")
            self.trade_stats = {}
    
    def _calculate_trade_metrics(self):
        """Calculate detailed trade metrics."""
        try:
            # Convert trades to round-trip trades
            round_trips = self._extract_round_trips()
            
            if round_trips.empty:
                logger.warning("No round-trip trades to analyze")
                self.trade_stats = {
                    "total_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "avg_loss": 0,
                    "profit_factor": 0,
                    "avg_trade_pnl": 0,
                    "max_win": 0,
                    "max_loss": 0,
                    "avg_trade_duration": 0
                }
                return
                
            # Calculate win/loss statistics
            round_trips["is_win"] = round_trips["pnl"] > 0
            
            total_trades = len(round_trips)
            winning_trades = round_trips[round_trips["is_win"]].shape[0]
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average profit, loss, and profit factor
            total_profit = round_trips[round_trips["is_win"]]["pnl"].sum() if winning_trades > 0 else 0
            total_loss = abs(round_trips[~round_trips["is_win"]]["pnl"].sum()) if losing_trades > 0 else 0
            
            avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Find maximum win and loss
            max_win = round_trips["pnl"].max() if not round_trips.empty else 0
            max_loss = round_trips["pnl"].min() if not round_trips.empty else 0
            
            # Calculate average trade P&L
            avg_trade_pnl = round_trips["pnl"].mean() if not round_trips.empty else 0
            
            # Calculate average trade duration
            if "duration" in round_trips.columns:
                avg_duration = round_trips["duration"].mean()
                avg_duration_str = str(avg_duration)
            else:
                avg_duration = None
                avg_duration_str = "N/A"
            
            # Store calculated statistics
            self.trade_stats = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "win_rate_pct": win_rate * 100,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_trade_pnl": avg_trade_pnl,
                "max_win": max_win,
                "max_loss": max_loss,
                "avg_trade_duration": avg_duration_str
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {str(e)}")
            self.trade_stats = {}
    
    def _extract_round_trips(self) -> pd.DataFrame:
        """
        Extract round-trip trades from the trades list.
        
        Returns:
            DataFrame with round-trip trades
        """
        if self.trades_df.empty:
            return pd.DataFrame()
            
        try:
            # This is a simplified approach - a more sophisticated implementation
            # would track positions over time and match entry/exit trades
            
            # Group trades by symbol
            symbols = self.trades_df["symbol"].unique()
            round_trips = []
            
            for symbol in symbols:
                # Get trades for this symbol
                symbol_trades = self.trades_df[self.trades_df["symbol"] == symbol].copy()
                
                # Sort by timestamp
                symbol_trades = symbol_trades.sort_index()
                
                # Track position
                position = 0
                entry_price = 0
                entry_time = None
                entry_cost = 0
                
                for idx, trade in symbol_trades.iterrows():
                    # For buy trades
                    if trade["direction"] == 1:
                        # If no position or adding to long, update entry
                        if position >= 0:
                            # Update entry price (weighted average)
                            total_cost = (position * entry_price) + (trade["quantity"] * trade["price"])
                            total_quantity = position + trade["quantity"]
                            entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                            
                            # Update or set entry time
                            if entry_time is None:
                                entry_time = idx
                                
                            # Update position
                            position += trade["quantity"]
                            
                        # If reducing short position, calculate P&L
                        else:
                            # Calculate P&L for the covered part
                            quantity_to_cover = min(trade["quantity"], abs(position))
                            pnl = (entry_price - trade["price"]) * quantity_to_cover
                            
                            # Create round-trip record
                            round_trips.append({
                                "symbol": symbol,
                                "entry_time": entry_time,
                                "exit_time": idx,
                                "duration": idx - entry_time if entry_time else pd.Timedelta(0),
                                "quantity": quantity_to_cover,
                                "entry_price": entry_price,
                                "exit_price": trade["price"],
                                "pnl": pnl,
                                "type": "short"
                            })
                            
                            # Update position
                            position += trade["quantity"]
                            
                            # If position becomes positive, reset entry
                            if position > 0:
                                entry_price = trade["price"]
                                entry_time = idx
                    
                    # For sell trades
                    else:
                        # If reducing long position, calculate P&L
                        if position > 0:
                            # Calculate P&L for the sold part
                            quantity_to_sell = min(trade["quantity"], position)
                            pnl = (trade["price"] - entry_price) * quantity_to_sell
                            
                            # Create round-trip record
                            round_trips.append({
                                "symbol": symbol,
                                "entry_time": entry_time,
                                "exit_time": idx,
                                "duration": idx - entry_time if entry_time else pd.Timedelta(0),
                                "quantity": quantity_to_sell,
                                "entry_price": entry_price,
                                "exit_price": trade["price"],
                                "pnl": pnl,
                                "type": "long"
                            })
                            
                            # Update position
                            position -= trade["quantity"]
                            
                            # If position becomes negative, reset entry
                            if position < 0:
                                entry_price = trade["price"]
                                entry_time = idx
                                
                        # If no position or adding to short, update entry
                        else:
                            # Update entry price (weighted average)
                            if position < 0:
                                total_cost = (abs(position) * entry_price) + (trade["quantity"] * trade["price"])
                                total_quantity = abs(position) + trade["quantity"]
                                entry_price = total_cost / total_quantity if total_quantity > 0 else 0
                            else:
                                entry_price = trade["price"]
                            
                            # Update or set entry time
                            if entry_time is None or position == 0:
                                entry_time = idx
                                
                            # Update position
                            position -= trade["quantity"]
            
            # Convert to DataFrame
            if round_trips:
                round_trips_df = pd.DataFrame(round_trips)
                return round_trips_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting round-trip trades: {str(e)}")
            return pd.DataFrame()
    
    def calculate_risk_metrics(self) -> Dict:
        """
        Calculate risk-related metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        if self.equity_curve.empty:
            return {
                "max_drawdown": 0,
                "max_drawdown_duration": 0,
                "value_at_risk_95": 0,
                "value_at_risk_99": 0,
                "expected_shortfall": 0,
                "calmar_ratio": 0,
                "sortino_ratio": 0
            }
            
        try:
            # Calculate drawdown
            peak = self.equity_curve["equity"].expanding(min_periods=1).max()
            drawdown = (self.equity_curve["equity"] / peak - 1) * 100
            
            # Max drawdown
            max_drawdown = drawdown.min()
            max_dd_idx = drawdown.idxmin()
            
            # Drawdown duration
            # Find the last peak before the max drawdown
            peak_before_dd = drawdown[:max_dd_idx][drawdown[:max_dd_idx] == 0].index[-1] if not drawdown[:max_dd_idx][drawdown[:max_dd_idx] == 0].empty else self.equity_curve.index[0]
            
            # Find the next recovery after the max drawdown
            try:
                recovery_after_dd = drawdown[max_dd_idx:][drawdown[max_dd_idx:] == 0].index[0] if not drawdown[max_dd_idx:][drawdown[max_dd_idx:] == 0].empty else self.equity_curve.index[-1]
            except:
                recovery_after_dd = self.equity_curve.index[-1]
                
            max_dd_duration = recovery_after_dd - peak_before_dd
            
            # Value at Risk (VaR)
            returns = self.equity_curve["returns"].dropna()
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (ES) / Conditional VaR
            es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            
            # Calmar Ratio (annualized return / max drawdown)
            annual_return = self.stats.get("annual_return", 0)
            calmar = annual_return / abs(max_drawdown / 100) if max_drawdown != 0 else float('inf')
            
            # Sortino Ratio (annualized return / downside deviation)
            # Using 0 as the required minimum return for simplicity
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = annual_return / downside_deviation if downside_deviation > 0 else float('inf')
            
            return {
                "max_drawdown": max_drawdown,
                "max_drawdown_duration": str(max_dd_duration),
                "value_at_risk_95": var_95,
                "value_at_risk_99": var_99,
                "expected_shortfall": es_95,
                "calmar_ratio": calmar,
                "sortino_ratio": sortino
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                "max_drawdown": 0,
                "max_drawdown_duration": 0,
                "value_at_risk_95": 0,
                "value_at_risk_99": 0,
                "expected_shortfall": 0,
                "calmar_ratio": 0,
                "sortino_ratio": 0,
                "error": str(e)
            }
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns from the equity curve.
        
        Returns:
            DataFrame with monthly returns
        """
        if self.equity_curve.empty:
            return pd.DataFrame()
            
        try:
            # Resample to month-end and calculate returns
            monthly_equity = self.equity_curve["equity"].resample("M").last()
            monthly_returns = monthly_equity.pct_change().dropna()
            
            # Create DataFrame with year and month columns
            monthly_df = pd.DataFrame({
                "return": monthly_returns,
                "year": monthly_returns.index.year,
                "month": monthly_returns.index.month
            })
            
            return monthly_df
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {str(e)}")
            return pd.DataFrame()
    
    def get_summary(self) -> Dict:
        """
        Get a comprehensive summary of the backtest results.
        
        Returns:
            Dictionary with all backtest metrics
        """
        # General performance metrics
        performance = self.stats.copy()
        
        # Trade statistics
        trade_stats = self.trade_stats.copy() if hasattr(self, "trade_stats") else {}
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics()
        
        # Combine all metrics
        summary = {
            "performance": performance,
            "trade_stats": trade_stats,
            "risk_metrics": risk_metrics
        }
        
        return summary
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)
        """
        if self.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
            
        plt.figure(figsize=figsize)
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve.index, self.equity_curve["equity"], label="Portfolio Value")
        plt.title("Equity Curve")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.legend()
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        peak = self.equity_curve["equity"].expanding(min_periods=1).max()
        drawdown = (self.equity_curve["equity"] / peak - 1) * 100
        plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
        plt.plot(drawdown.index, drawdown, color="red", label="Drawdown")
        plt.title("Drawdown")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_returns_distribution(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of returns.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)
        """
        if self.equity_curve.empty:
            logger.warning("No returns data to plot")
            return
            
        returns = self.equity_curve["returns"].dropna()
        
        plt.figure(figsize=figsize)
        
        # Plot returns distribution
        sns.histplot(returns, kde=True)
        plt.axvline(x=0, color="red", linestyle="--")
        plt.title("Distribution of Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Plot a heatmap of monthly returns.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)
        """
        monthly_returns = self.calculate_monthly_returns()
        
        if monthly_returns.empty:
            logger.warning("No monthly returns data to plot")
            return
            
        # Create pivot table for heatmap
        pivot_table = monthly_returns.pivot_table(index="year", columns="month", values="return")
        
        # Convert to percentage for display
        pivot_table_pct = pivot_table * 100
        
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_table_pct, annot=True, cmap="RdYlGn", center=0, fmt=".2f")
        plt.title("Monthly Returns (%)")
        plt.xlabel("Month")
        plt.ylabel("Year")
        
        # Update month labels
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plt.xticks(np.arange(12) + 0.5, month_labels)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Plot trade analysis charts.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)
        """
        if not hasattr(self, "trade_stats") or not self.trade_stats:
            logger.warning("No trade statistics to plot")
            return
            
        round_trips = self._extract_round_trips()
        
        if round_trips.empty:
            logger.warning("No round-trip trades to analyze")
            return
            
        plt.figure(figsize=figsize)
        
        # Plot P&L per trade
        plt.subplot(2, 2, 1)
        round_trips["pnl"].plot(kind="bar", color=round_trips["pnl"].apply(lambda x: "green" if x > 0 else "red"))
        plt.title("P&L per Trade")
        plt.xlabel("Trade #")
        plt.ylabel("P&L ($)")
        plt.grid(True)
        
        # Plot P&L distribution
        plt.subplot(2, 2, 2)
        sns.histplot(round_trips["pnl"], kde=True)
        plt.axvline(x=0, color="red", linestyle="--")
        plt.title("P&L Distribution")
        plt.xlabel("P&L ($)")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        # Plot cumulative P&L
        plt.subplot(2, 2, 3)
        cum_pnl = round_trips["pnl"].cumsum()
        plt.plot(range(len(cum_pnl)), cum_pnl)
        plt.title("Cumulative P&L")
        plt.xlabel("Trade #")
        plt.ylabel("Cumulative P&L ($)")
        plt.grid(True)
        
        # Plot win/loss by trade type
        plt.subplot(2, 2, 4)
        if "type" in round_trips.columns:
            win_loss_by_type = pd.crosstab(round_trips["type"], round_trips["pnl"] > 0)
            win_loss_by_type.columns = ["Loss", "Win"]
            win_loss_by_type.plot(kind="bar", stacked=True)
            plt.title("Win/Loss by Trade Type")
            plt.xlabel("Trade Type")
            plt.ylabel("Count")
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No trade type data available", horizontalalignment="center", verticalalignment="center")
            plt.title("Win/Loss by Trade Type")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()