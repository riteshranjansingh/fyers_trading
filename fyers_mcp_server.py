#!/usr/bin/env python3
"""
FYERS Trading MCP Server

This MCP server provides tools for interacting with FYERS trading API,
making complex trading operations accessible through natural language.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import pandas as pd

# Add the project root to Python path - fixed for your project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# MCP imports
try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server.stdio import stdio_server
except ImportError as e:
    print(f"MCP import error: {e}")
    print("Please install MCP: pip install mcp")
    sys.exit(1)

# Import your existing FYERS modules
try:
    from src.api.connection import FyersConnection
    from src.api.data import FyersDataFetcher  
    from src.api.symbol_manager import SymbolManager
    from src.api.data_utils import add_technical_indicators, calculate_returns
    from src.utils.logging_config import setup_logging
except ImportError as e:
    print(f"FYERS module import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Set up logging
logger = setup_logging()

class FyersMCPServer:
    """MCP Server for FYERS Trading API integration."""
    
    def __init__(self):
        self.connection = None
        self.data_fetcher = None
        self.symbol_manager = None
        self.server = Server("fyers-trading")
        self._setup_tools()
    
    async def initialize(self):
        """Initialize FYERS connection and components."""
        try:
            logger.info("Initializing FYERS MCP Server...")
            
            # Initialize connection
            self.connection = FyersConnection()
            if not self.connection.authenticate():
                raise Exception("Failed to authenticate with FYERS API")
            
            # Initialize symbol manager
            self.symbol_manager = SymbolManager()
            
            # Initialize data fetcher
            self.data_fetcher = FyersDataFetcher(self.connection, self.symbol_manager)
            
            logger.info("FYERS MCP Server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FYERS MCP Server: {str(e)}")
            raise
    
    def _setup_tools(self):
        """Set up MCP tools for FYERS operations."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available FYERS trading tools."""
            return [
                Tool(
                    name="search_symbols",
                    description="Search for trading symbols by name, exchange, or type",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Symbol name to search for (e.g., 'SBIN', 'RELIANCE')"
                            },
                            "exchange": {
                                "type": "string",
                                "description": "Exchange filter (NSE, BSE, MCX)",
                                "enum": ["NSE", "BSE", "MCX"]
                            },
                            "symbol_type": {
                                "type": "string", 
                                "description": "Symbol type filter (EQ, FUT, OPT, etc.)",
                                "enum": ["EQ", "FUT", "OPT", "CE", "PE"]
                            },
                            "exact_match": {
                                "type": "boolean",
                                "description": "Whether to search for exact matches only",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                ),
                
                Tool(
                    name="get_historical_data",
                    description="Fetch historical price data for a symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol (e.g., 'NSE:SBIN-EQ' or just 'SBIN')"
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Time resolution for data",
                                "enum": ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mo"],
                                "default": "1d"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to look back from today",
                                "default": 30
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format (optional, overrides days_back)"
                            },
                            "date_to": {
                                "type": "string", 
                                "description": "End date in YYYY-MM-DD format (optional, defaults to today)"
                            },
                            "add_indicators": {
                                "type": "boolean",
                                "description": "Whether to add technical indicators",
                                "default": False
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                
                Tool(
                    name="get_market_quotes",
                    description="Get current market quotes for symbols",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols to get quotes for"
                            }
                        },
                        "required": ["symbols"]
                    }
                ),
                
                Tool(
                    name="get_market_depth", 
                    description="Get market depth (order book) for a symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to get depth for"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                
                Tool(
                    name="analyze_data",
                    description="Perform technical analysis on historical data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol to analyze"
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Time resolution",
                                "enum": ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1mo"],
                                "default": "1d"
                            },
                            "days_back": {
                                "type": "integer", 
                                "description": "Number of days to analyze",
                                "default": 100
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis to perform",
                                "enum": ["basic_stats", "technical_indicators", "returns_analysis", "volatility"],
                                "default": "basic_stats"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                
                Tool(
                    name="backtest_strategy",
                    description="Run a simple backtest on historical data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol to backtest on"
                            },
                            "strategy": {
                                "type": "string", 
                                "description": "Strategy type to test",
                                "enum": ["sma_crossover", "rsi_mean_reversion", "bollinger_bands"],
                                "default": "sma_crossover"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to backtest",
                                "default": 365
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Strategy-specific parameters",
                                "additionalProperties": True
                            }
                        },
                        "required": ["symbol"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if not self.data_fetcher:
                    await self.initialize()
                
                if name == "search_symbols":
                    return await self._search_symbols(**arguments)
                elif name == "get_historical_data":
                    return await self._get_historical_data(**arguments)
                elif name == "get_market_quotes":
                    return await self._get_market_quotes(**arguments)
                elif name == "get_market_depth":
                    return await self._get_market_depth(**arguments)
                elif name == "analyze_data":
                    return await self._analyze_data(**arguments)
                elif name == "backtest_strategy":
                    return await self._backtest_strategy(**arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error in tool call {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _search_symbols(self, query: str, exchange: str = None, 
                            symbol_type: str = None, exact_match: bool = False) -> List[TextContent]:
        """Search for symbols based on criteria."""
        try:
            results = self.symbol_manager.search(
                name=query,
                exchange=exchange,
                symbol_type=symbol_type,
                exact_match=exact_match
            )
            
            if not results:
                return [TextContent(type="text", text=f"No symbols found for query: {query}")]
            
            # Format results
            output = [f"Found {len(results)} symbols matching '{query}':\n"]
            
            for i, symbol_info in enumerate(results[:10]):  # Limit to top 10
                symbol_str = self.symbol_manager.format_symbol(symbol_info)
                description = symbol_info.get('description', 'N/A')
                output.append(f"{i+1}. {symbol_str} - {description}")
            
            if len(results) > 10:
                output.append(f"\n... and {len(results) - 10} more results")
            
            return [TextContent(type="text", text="\n".join(output))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching symbols: {str(e)}")]
    
    async def _get_historical_data(self, symbol: str, resolution: str = "1d", 
                                 days_back: int = 30, date_from: str = None,
                                 date_to: str = None, add_indicators: bool = False) -> List[TextContent]:
        """Fetch historical data for a symbol."""
        try:
            # Calculate dates if not provided
            if not date_from:
                end_date = datetime.now() if not date_to else datetime.strptime(date_to, "%Y-%m-%d")
                start_date = end_date - timedelta(days=days_back)
            else:
                start_date = datetime.strptime(date_from, "%Y-%m-%d")
                end_date = datetime.strptime(date_to, "%Y-%m-%d") if date_to else datetime.now()
            
            # Fetch data using extended method for long periods
            if (end_date - start_date).days > 100:
                df = self.data_fetcher.get_extended_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    date_from=start_date,
                    date_to=end_date
                )
            else:
                df = self.data_fetcher.get_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    date_from=start_date,
                    date_to=end_date
                )
            
            if df.empty:
                return [TextContent(type="text", text=f"No data found for {symbol}")]
            
            # Add technical indicators if requested
            if add_indicators:
                df = add_technical_indicators(df)
            
            # Create summary
            summary = [
                f"Historical data for {symbol} ({resolution})",
                f"Period: {df.index.min().date()} to {df.index.max().date()}",
                f"Records: {len(df)}",
                "",
                "Latest 5 records:",
                df.tail().round(2).to_string(),
                "",
                "Basic statistics:",
                df[['open', 'high', 'low', 'close', 'volume']].describe().round(2).to_string()
            ]
            
            return [TextContent(type="text", text="\n".join(summary))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error fetching historical data: {str(e)}")]
    
    async def _get_market_quotes(self, symbols: List[str]) -> List[TextContent]:
        """Get current market quotes."""
        try:
            quotes = self.data_fetcher.get_market_quotes(symbols)
            
            if not quotes:
                return [TextContent(type="text", text="No quotes data received")]
            
            output = ["Current Market Quotes:\n"]
            
            for quote_data in quotes:
                if quote_data.get('s') == 'ok' and 'v' in quote_data:
                    v = quote_data['v']
                    symbol = quote_data.get('n', 'Unknown')
                    ltp = v.get('ltp', 'N/A')
                    chp = v.get('chp', 'N/A')
                    volume = v.get('volume', 'N/A')
                    
                    output.append(f"• {symbol}: ₹{ltp} ({chp:+.2f}%) Vol: {volume:,}")
            
            return [TextContent(type="text", text="\n".join(output))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error fetching quotes: {str(e)}")]
    
    async def _get_market_depth(self, symbol: str) -> List[TextContent]:
        """Get market depth for a symbol."""
        try:
            depth = self.data_fetcher.get_market_depth(symbol)
            
            if not depth:
                return [TextContent(type="text", text=f"No market depth data for {symbol}")]
            
            # Extract depth data for the symbol
            symbol_depth = depth.get(symbol, {})
            
            if not symbol_depth:
                return [TextContent(type="text", text=f"No depth data found for {symbol}")]
            
            output = [f"Market Depth for {symbol}:\n"]
            
            # Basic info
            ltp = symbol_depth.get('ltp', 'N/A')
            volume = symbol_depth.get('v', 'N/A')
            output.append(f"LTP: ₹{ltp}, Volume: {volume:,}")
            output.append("")
            
            # Bid/Ask data
            bids = symbol_depth.get('bids', [])
            asks = symbol_depth.get('ask', [])
            
            output.append("Top 5 Bids and Asks:")
            output.append("Bids (Price/Qty/Orders) | Asks (Price/Qty/Orders)")
            output.append("-" * 50)
            
            for i in range(min(5, max(len(bids), len(asks)))):
                bid_str = ""
                ask_str = ""
                
                if i < len(bids):
                    bid = bids[i]
                    bid_str = f"₹{bid.get('price', 0):.2f}/{bid.get('volume', 0)}/{bid.get('ord', 0)}"
                
                if i < len(asks):
                    ask = asks[i] 
                    ask_str = f"₹{ask.get('price', 0):.2f}/{ask.get('volume', 0)}/{ask.get('ord', 0)}"
                
                output.append(f"{bid_str:20} | {ask_str}")
            
            return [TextContent(type="text", text="\n".join(output))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error fetching market depth: {str(e)}")]
    
    async def _analyze_data(self, symbol: str, resolution: str = "1d", 
                          days_back: int = 100, analysis_type: str = "basic_stats") -> List[TextContent]:
        """Perform analysis on historical data."""
        try:
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            df = self.data_fetcher.get_extended_historical_data(
                symbol=symbol,
                resolution=resolution,
                date_from=start_date,
                date_to=end_date
            )
            
            if df.empty:
                return [TextContent(type="text", text=f"No data available for {symbol}")]
            
            output = [f"Analysis for {symbol} ({analysis_type}):\n"]
            
            if analysis_type == "basic_stats":
                stats = df[['open', 'high', 'low', 'close', 'volume']].describe()
                output.append("Basic Statistics:")
                output.append(stats.round(2).to_string())
                
            elif analysis_type == "technical_indicators":
                df_with_indicators = add_technical_indicators(df)
                latest = df_with_indicators.iloc[-1]
                
                output.append("Latest Technical Indicators:")
                output.append(f"SMA(20): ₹{latest.get('SMA_20', 'N/A'):.2f}")
                output.append(f"RSI(14): {latest.get('RSI', 'N/A'):.2f}")
                output.append(f"MACD: {latest.get('MACD', 'N/A'):.4f}")
                
            elif analysis_type == "returns_analysis":
                df_with_returns = calculate_returns(df)
                
                daily_return = df_with_returns['return_1d'].dropna()
                output.append("Returns Analysis:")
                output.append(f"Average Daily Return: {daily_return.mean()*100:.3f}%")
                output.append(f"Daily Volatility: {daily_return.std()*100:.3f}%")
                output.append(f"Best Day: {daily_return.max()*100:.2f}%")
                output.append(f"Worst Day: {daily_return.min()*100:.2f}%")
                
            elif analysis_type == "volatility":
                returns = df['close'].pct_change().dropna()
                rolling_vol = returns.rolling(20).std() * (252**0.5) * 100  # Annualized
                
                output.append("Volatility Analysis:")
                output.append(f"Current 20-day Volatility: {rolling_vol.iloc[-1]:.2f}%")
                output.append(f"Average Volatility: {rolling_vol.mean():.2f}%")
                output.append(f"Max Volatility: {rolling_vol.max():.2f}%")
                output.append(f"Min Volatility: {rolling_vol.min():.2f}%")
            
            return [TextContent(type="text", text="\n".join(output))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error in analysis: {str(e)}")]
    
    async def _backtest_strategy(self, symbol: str, strategy: str = "sma_crossover",
                               days_back: int = 365, parameters: Dict = None) -> List[TextContent]:
        """Run a simple backtest."""
        try:
            # This is a simplified backtest - you can expand this significantly
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            df = self.data_fetcher.get_extended_historical_data(
                symbol=symbol,
                resolution="1d",
                date_from=start_date,
                date_to=end_date
            )
            
            if df.empty or len(df) < 50:
                return [TextContent(type="text", text=f"Insufficient data for backtesting {symbol}")]
            
            # Add technical indicators
            df = add_technical_indicators(df)
            
            output = [f"Backtest Results for {symbol} ({strategy}):\n"]
            
            if strategy == "sma_crossover":
                # Simple SMA crossover strategy
                fast_period = parameters.get('fast_period', 20) if parameters else 20
                slow_period = parameters.get('slow_period', 50) if parameters else 50
                
                # Generate signals
                df['signal'] = 0
                df.loc[df['SMA_20'] > df['SMA_50'], 'signal'] = 1
                df.loc[df['SMA_20'] <= df['SMA_50'], 'signal'] = -1
                
                # Calculate returns
                df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change()
                df['buy_hold_return'] = df['close'].pct_change()
                
                # Performance metrics
                strategy_total = (1 + df['strategy_return'].fillna(0)).cumprod().iloc[-1] - 1
                buy_hold_total = (1 + df['buy_hold_return'].fillna(0)).cumprod().iloc[-1] - 1
                
                output.append(f"Strategy: SMA({fast_period}) vs SMA({slow_period}) Crossover")
                output.append(f"Period: {df.index.min().date()} to {df.index.max().date()}")
                output.append(f"Total Strategy Return: {strategy_total*100:.2f}%")
                output.append(f"Total Buy & Hold Return: {buy_hold_total*100:.2f}%")
                output.append(f"Outperformance: {(strategy_total - buy_hold_total)*100:.2f}%")
                
                # Trade statistics
                signals = df['signal'].diff()
                entries = len(signals[signals != 0]) // 2
                output.append(f"Number of Trades: {entries}")
                
            return [TextContent(type="text", text="\n".join(output))]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error in backtesting: {str(e)}")]


async def main():
    """Main function to run the FYERS MCP server."""
    fyers_server = FyersMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await fyers_server.server.run(
            read_stream, 
            write_stream, 
            InitializationOptions(
                server_name="fyers-trading",
                server_version="1.0.0",
                capabilities=fyers_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())