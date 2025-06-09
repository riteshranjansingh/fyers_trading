#!/usr/bin/env python3
"""
FYERS MCP Server - Main server file
Integrates FYERS API with Claude AI using Model Context Protocol (MCP)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

# MCP imports
import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Add the src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our existing FYERS modules
from src.api.connection import FyersConnection
from src.api.data import FyersDataFetcher
from src.api.symbol_manager import SymbolManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fyers-mcp")

class FyersMCPServer:
    """Main FYERS MCP Server class"""
    
    def __init__(self):
        self.connection = None
        self.data_fetcher = None
        self.symbol_manager = None
        self.server = Server("fyers-mcp")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up all MCP tool handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List all available FYERS MCP tools"""
            return [
                types.Tool(
                    name="check_connection",
                    description="Check FYERS API connection status and authenticate if needed",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="get_profile",
                    description="Get FYERS account profile information",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                types.Tool(
                    name="get_market_quote",
                    description="Get current market quotes for specified symbols",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols (e.g., ['SBIN', 'RELIANCE'] or ['NSE:SBIN-EQ'])"
                            }
                        },
                        "required": ["symbols"]
                    }
                ),
                types.Tool(
                    name="get_historical_data",
                    description="Fetch historical price data for a symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol name (e.g., 'SBIN' or 'NSE:SBIN-EQ')"
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Time resolution (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1mo)",
                                "default": "1d"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to fetch data for",
                                "default": 30
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                types.Tool(
                    name="get_market_depth",
                    description="Get market depth (order book) for a symbol",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol name (e.g., 'SBIN' or 'NSE:SBIN-EQ')"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                types.Tool(
                    name="search_symbols",
                    description="Search for symbols by name, exchange, or type",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Symbol name to search for (e.g., 'SBIN', 'RELIANCE')"
                            },
                            "exchange": {
                                "type": "string",
                                "description": "Exchange filter (NSE, BSE)",
                                "default": None
                            },
                            "symbol_type": {
                                "type": "string", 
                                "description": "Symbol type filter (EQ, FUT, OPT)",
                                "default": None
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10
                            }
                        },
                        "required": ["name"]
                    }
                ),
                types.Tool(
                    name="analyze_symbol_performance",
                    description="Analyze symbol performance with technical indicators",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol name to analyze"
                            },
                            "period": {
                                "type": "integer",
                                "description": "Analysis period in days",
                                "default": 30
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                types.Tool(
                    name="compare_symbols",
                    description="Compare performance of multiple symbols",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols to compare"
                            },
                            "period": {
                                "type": "integer",
                                "description": "Comparison period in days",
                                "default": 30
                            }
                        },
                        "required": ["symbols"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls"""
            try:
                if name == "check_connection":
                    return await self._check_connection()
                elif name == "get_profile":
                    return await self._get_profile()
                elif name == "get_market_quote":
                    return await self._get_market_quote(arguments.get("symbols", []))
                elif name == "get_historical_data":
                    return await self._get_historical_data(
                        arguments.get("symbol"),
                        arguments.get("resolution", "1d"),
                        arguments.get("days_back", 30)
                    )
                elif name == "get_market_depth":
                    return await self._get_market_depth(arguments.get("symbol"))
                elif name == "search_symbols":
                    return await self._search_symbols(
                        arguments.get("name"),
                        arguments.get("exchange"),
                        arguments.get("symbol_type"),
                        arguments.get("limit", 10)
                    )
                elif name == "analyze_symbol_performance":
                    return await self._analyze_symbol_performance(
                        arguments.get("symbol"),
                        arguments.get("period", 30)
                    )
                elif name == "compare_symbols":
                    return await self._compare_symbols(
                        arguments.get("symbols", []),
                        arguments.get("period", 30)
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [types.TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]
    
    async def _ensure_connection(self):
        """Ensure FYERS connection is established"""
        if not self.connection:
            self.connection = FyersConnection()
            if not self.connection.authenticate():
                raise Exception("Failed to authenticate with FYERS API")
            
            self.data_fetcher = FyersDataFetcher(self.connection)
            self.symbol_manager = SymbolManager()
    
    async def _check_connection(self) -> list[types.TextContent]:
        """Check FYERS connection status"""
        try:
            await self._ensure_connection()
            if self.connection.test_connection():
                return [types.TextContent(
                    type="text", 
                    text="✅ FYERS API connection is active and working properly."
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text="❌ FYERS API connection test failed. Please check your credentials."
                )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Connection error: {str(e)}"
            )]
    
    async def _get_profile(self) -> list[types.TextContent]:
        """Get FYERS profile information"""
        try:
            await self._ensure_connection()
            session = self.connection.get_session()
            profile = session.get_profile()
            
            if profile.get("s") == "ok":
                profile_data = profile.get("data", {})
                response = f"""📊 **FYERS Account Profile**
                
**User Details:**
- Name: {profile_data.get('name', 'N/A')}
- Email: {profile_data.get('email', 'N/A')}
- User ID: {profile_data.get('user_id', 'N/A')}
- Mobile: {profile_data.get('mobile', 'N/A')}

**Account Status:**
- Status: {profile_data.get('user_type', 'N/A')}
- Exchange Access: {', '.join(profile_data.get('exchanges', []))}
"""
                return [types.TextContent(type="text", text=response)]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Failed to fetch profile: {profile.get('message', 'Unknown error')}"
                )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error fetching profile: {str(e)}"
            )]
    
    async def _get_market_quote(self, symbols: List[str]) -> list[types.TextContent]:
        """Get market quotes for symbols"""
        try:
            await self._ensure_connection()
            quotes = self.data_fetcher.get_market_quotes(symbols)
            
            if not quotes:
                return [types.TextContent(
                    type="text",
                    text="❌ No quote data received"
                )]
            
            response = "📈 **Market Quotes**\n\n"
            for quote_data in quotes:
                if quote_data.get("s") == "ok":
                    data = quote_data.get("v", {})
                    symbol = data.get("symbol", "Unknown")
                    ltp = data.get("lp", 0)
                    change = data.get("ch", 0)
                    change_pct = data.get("chp", 0)
                    volume = data.get("volume", 0)
                    
                    change_indicator = "🟢" if change >= 0 else "🔴"
                    response += f"""{change_indicator} **{symbol}**
- Last Price: ₹{ltp:,.2f}
- Change: ₹{change:+,.2f} ({change_pct:+.2f}%)
- Volume: {volume:,}
- High: ₹{data.get('high_price', 0):,.2f} | Low: ₹{data.get('low_price', 0):,.2f}

"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error fetching quotes: {str(e)}"
            )]
    
    async def _get_historical_data(self, symbol: str, resolution: str, days_back: int) -> list[types.TextContent]:
        """Get historical data for a symbol"""
        try:
            await self._ensure_connection()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            df = self.data_fetcher.get_historical_data(
                symbol=symbol,
                resolution=resolution,
                date_from=start_date,
                date_to=end_date
            )
            
            if df.empty:
                return [types.TextContent(
                    type="text",
                    text=f"❌ No historical data found for {symbol}"
                )]
            
            # Basic analysis
            latest = df.iloc[-1]
            first = df.iloc[0]
            
            total_return = ((latest['close'] - first['close']) / first['close']) * 100
            high_52w = df['high'].max()
            low_52w = df['low'].min()
            avg_volume = df['volume'].mean()
            
            response = f"""📊 **Historical Data Analysis for {symbol}**
            
**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({resolution})
**Total Records:** {len(df)}

**Price Summary:**
- Current Price: ₹{latest['close']:,.2f}
- Period Return: {total_return:+.2f}%
- High: ₹{high_52w:,.2f} | Low: ₹{low_52w:,.2f}
- Average Volume: {avg_volume:,.0f}

**Recent 5 Days:**
```
Date        Open    High    Low     Close   Volume
{df.tail().to_string(columns=['open', 'high', 'low', 'close', 'volume'], float_format='%.2f')}
```
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error fetching historical data: {str(e)}"
            )]
    
    async def _get_market_depth(self, symbol: str) -> list[types.TextContent]:
        """Get market depth for a symbol"""
        try:
            await self._ensure_connection()
            depth = self.data_fetcher.get_market_depth(symbol)
            
            if not depth:
                return [types.TextContent(
                    type="text",
                    text=f"❌ No market depth data found for {symbol}"
                )]
            
            symbol_data = list(depth.values())[0] if depth else {}
            
            response = f"""📋 **Market Depth for {symbol}**

**Current Price:** ₹{symbol_data.get('ltp', 0):,.2f}
**Total Buy Qty:** {symbol_data.get('totalbuyqty', 0):,}
**Total Sell Qty:** {symbol_data.get('totalsellqty', 0):,}

**Top 5 Bids:**
"""
            bids = symbol_data.get('bids', [])[:5]
            for i, bid in enumerate(bids, 1):
                response += f"{i}. ₹{bid.get('price', 0):,.2f} × {bid.get('volume', 0):,} ({bid.get('ord', 0)} orders)\n"
            
            response += "\n**Top 5 Asks:**\n"
            asks = symbol_data.get('ask', [])[:5]
            for i, ask in enumerate(asks, 1):
                response += f"{i}. ₹{ask.get('price', 0):,.2f} × {ask.get('volume', 0):,} ({ask.get('ord', 0)} orders)\n"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error fetching market depth: {str(e)}"
            )]
    
    async def _search_symbols(self, name: str, exchange: str, symbol_type: str, limit: int) -> list[types.TextContent]:
        """Search for symbols"""
        try:
            await self._ensure_connection()
            
            matches = self.symbol_manager.search(
                name=name,
                exchange=exchange,
                symbol_type=symbol_type
            )
            
            if not matches:
                return [types.TextContent(
                    type="text",
                    text=f"❌ No symbols found matching '{name}'"
                )]
            
            response = f"🔍 **Symbol Search Results for '{name}'**\n\n"
            
            for i, match in enumerate(matches[:limit], 1):
                symbol_str = self.symbol_manager.format_symbol(match)
                description = match.get('description', 'N/A')
                response += f"{i}. **{symbol_str}**\n   Description: {description}\n\n"
            
            if len(matches) > limit:
                response += f"... and {len(matches) - limit} more results"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error searching symbols: {str(e)}"
            )]
    
    async def _analyze_symbol_performance(self, symbol: str, period: int) -> list[types.TextContent]:
        """Analyze symbol performance with technical indicators"""
        try:
            await self._ensure_connection()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
            
            df = self.data_fetcher.get_historical_data(
                symbol=symbol,
                resolution="1d",
                date_from=start_date,
                date_to=end_date
            )
            
            if df.empty:
                return [types.TextContent(
                    type="text",
                    text=f"❌ No data found for {symbol}"
                )]
            
            # Import data utils for technical analysis
            from src.api.data_utils import add_technical_indicators
            df_with_indicators = add_technical_indicators(df)
            
            latest = df_with_indicators.iloc[-1]
            first = df_with_indicators.iloc[0]
            
            total_return = ((latest['close'] - first['close']) / first['close']) * 100
            volatility = df_with_indicators['daily_return'].std() * (252 ** 0.5) * 100  # Annualized
            
            response = f"""📈 **Technical Analysis for {symbol}**

**Performance Summary ({period} days):**
- Total Return: {total_return:+.2f}%
- Annualized Volatility: {volatility:.2f}%
- Current Price: ₹{latest['close']:,.2f}

**Technical Indicators:**
- RSI: {latest['RSI']:.2f} {'(Overbought)' if latest['RSI'] > 70 else '(Oversold)' if latest['RSI'] < 30 else '(Neutral)'}
- SMA 20: ₹{latest['SMA_20']:,.2f}
- EMA 20: ₹{latest['EMA_20']:,.2f}
- Position vs SMA20: {'Above' if latest['close'] > latest['SMA_20'] else 'Below'} ({((latest['close'] / latest['SMA_20'] - 1) * 100):+.2f}%)

**Bollinger Bands:**
- Upper: ₹{latest['BB_upper']:,.2f}
- Middle: ₹{latest['BB_middle']:,.2f}
- Lower: ₹{latest['BB_lower']:,.2f}

**MACD:**
- MACD: {latest['MACD']:.4f}
- Signal: {latest['MACD_signal']:.4f}
- Histogram: {latest['MACD_hist']:+.4f}
"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error analyzing symbol: {str(e)}"
            )]
    
    async def _compare_symbols(self, symbols: List[str], period: int) -> list[types.TextContent]:
        """Compare performance of multiple symbols"""
        try:
            await self._ensure_connection()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period)
            
            comparison_data = []
            
            for symbol in symbols:
                try:
                    df = self.data_fetcher.get_historical_data(
                        symbol=symbol,
                        resolution="1d",
                        date_from=start_date,
                        date_to=end_date
                    )
                    
                    if not df.empty:
                        first_price = df.iloc[0]['close']
                        last_price = df.iloc[-1]['close']
                        total_return = ((last_price - first_price) / first_price) * 100
                        volatility = df['close'].pct_change().std() * (252 ** 0.5) * 100
                        
                        comparison_data.append({
                            'symbol': symbol,
                            'return': total_return,
                            'volatility': volatility,
                            'current_price': last_price
                        })
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
            
            if not comparison_data:
                return [types.TextContent(
                    type="text",
                    text="❌ No data available for comparison"
                )]
            
            # Sort by return
            comparison_data.sort(key=lambda x: x['return'], reverse=True)
            
            response = f"📊 **Symbol Comparison ({period} days)**\n\n"
            response += "```\nSymbol          Return    Volatility  Current Price\n"
            response += "-" * 50 + "\n"
            
            for data in comparison_data:
                response += f"{data['symbol']:<12} {data['return']:+7.2f}%  {data['volatility']:7.2f}%  ₹{data['current_price']:>9,.2f}\n"
            
            response += "```\n\n"
            
            best_performer = comparison_data[0]
            worst_performer = comparison_data[-1]
            
            response += f"🏆 **Best Performer:** {best_performer['symbol']} ({best_performer['return']:+.2f}%)\n"
            response += f"📉 **Worst Performer:** {worst_performer['symbol']} ({worst_performer['return']:+.2f}%)"
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ Error comparing symbols: {str(e)}"
            )]

async def main():
    """Main function to run the FYERS MCP server"""
    server_instance = FyersMCPServer()
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="fyers-mcp",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())