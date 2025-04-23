"""
Symbol management module for FYERS API.

This module provides tools for loading, searching, and formatting symbols
from FYERS symbol master files.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Union, Set, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SymbolManager:
    """
    Manages symbol data for trading applications.
    
    This class handles:
    - Loading symbol data from FYERS symbol master files
    - Searching for symbols by name, exchange, segment, etc.
    - Formatting symbols for API calls
    - Caching symbol data for efficient lookups
    """
    
    # FYERS symbol master URLs - can be updated as needed
    SYMBOL_URLS = {
        "NSE_CM": "https://public.fyers.in/sym_details/NSE_CM_sym_master.json",
        "NSE_FO": "https://public.fyers.in/sym_details/NSE_FO_sym_master.json",
        "BSE_CM": "https://public.fyers.in/sym_details/BSE_CM_sym_master.json",
        "MCX_COM": "https://public.fyers.in/sym_details/MCX_COM_sym_master.json"
    }
    
    def __init__(self, 
                 symbol_files: Optional[Dict[str, str]] = None, 
                 download_missing: bool = True,
                 cache_dir: str = None):
        """
        Initialize the SymbolManager.
        
        Args:
            symbol_files: Dictionary mapping segment names to file paths
                          Example: {"NSE_CM": "path/to/nse_cm.json"}
            download_missing: Whether to download missing files from FYERS
            cache_dir: Directory to store downloaded symbol files
        """
        self.symbols = {}  # Will hold all symbol data by segment
        self.lookup_cache = {}  # For quick lookups by criteria
        
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "data", 
            "symbols"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load symbols from provided files or download
        if symbol_files:
            for segment, file_path in symbol_files.items():
                self.load_symbols(segment, file_path)
        elif download_missing:
            self.download_all_symbol_files()
    
    def download_symbol_file(self, segment: str, force: bool = False) -> bool:
        """
        Download a symbol master file from FYERS.
        
        Args:
            segment: Segment name (e.g., "NSE_CM")
            force: Whether to download even if file exists
            
        Returns:
            bool: True if download was successful
        """
        if segment not in self.SYMBOL_URLS:
            logger.error(f"Unknown segment: {segment}")
            return False
        
        url = self.SYMBOL_URLS[segment]
        file_path = os.path.join(self.cache_dir, f"{segment}_sym_master.json")
        
        # Skip if file exists and not forced to download
        if os.path.exists(file_path) and not force:
            logger.info(f"Symbol file for {segment} already exists at {file_path}")
            self.load_symbols(segment, file_path)
            return True
        
        try:
            logger.info(f"Downloading {segment} symbol master file from {url}")
            response = requests.get(url)
            response.raise_for_status()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(response.json(), f)
            
            logger.info(f"Successfully downloaded {segment} symbol file to {file_path}")
            
            # Load the symbols from the downloaded file
            self.load_symbols(segment, file_path)
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {segment} symbol file: {str(e)}")
            return False
    
    def download_all_symbol_files(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all symbol master files from FYERS.
        
        Args:
            force: Whether to download even if files exist
            
        Returns:
            Dict mapping segment names to download success (True/False)
        """
        results = {}
        for segment in self.SYMBOL_URLS.keys():
            results[segment] = self.download_symbol_file(segment, force)
        return results
    
    def load_symbols(self, segment: str, file_path: str) -> int:
        """
        Load symbols from a symbol master file.
        
        Args:
            segment: Segment name (e.g., "NSE_CM")
            file_path: Path to the symbol master JSON file
            
        Returns:
            int: Number of symbols loaded
        """
        try:
            logger.info(f"Loading {segment} symbols from {file_path}")
            
            with open(file_path, 'r') as f:
                symbols_data = json.load(f)
            
            # Store the symbols for this segment
            self.symbols[segment] = symbols_data
            
            # Clear lookup cache when loading new symbols
            self.lookup_cache = {}
            
            logger.info(f"Successfully loaded {len(symbols_data)} {segment} symbols")
            return len(symbols_data)
            
        except Exception as e:
            logger.error(f"Failed to load {segment} symbols: {str(e)}")
            return 0
    
    def search(self, 
              name: str = None, 
              exchange: str = None,
              symbol_type: str = None,
              segment: str = None,
              exact_match: bool = False) -> List[Dict]:
        """
        Search for symbols matching the specified criteria.
        
        Args:
            name: Symbol name (e.g., "SBIN", "RELIANCE")
            exchange: Exchange name (e.g., "NSE", "BSE")
            symbol_type: Symbol type (e.g., "EQ", "FUT", "OPT")
            segment: Segment to search in (e.g., "NSE_CM")
            exact_match: Whether to require exact matches for name
            
        Returns:
            List of matching symbol dictionaries
        """
        # Create a cache key for this search
        cache_key = f"{name}:{exchange}:{symbol_type}:{segment}:{exact_match}"
        
        # Return cached results if available
        if cache_key in self.lookup_cache:
            return self.lookup_cache[cache_key]
        
        results = []
        
        # Determine which segments to search
        segments_to_search = [segment] if segment and segment in self.symbols else self.symbols.keys()
        
        for seg in segments_to_search:
            if seg not in self.symbols:
                continue
                
            # Handle the case where symbols is a dictionary with keys as symbol strings
            symbol_dict = self.symbols[seg]
            
            for symbol_str, symbol_data in symbol_dict.items():
                # Apply filters
                match = True
                
                if name:
                    # Check if name is in the symbol string (e.g., "NSE:SBIN-EQ")
                    if exact_match:
                        # For exact match, check if the name appears as a full word
                        symbol_parts = symbol_str.split(':')
                        if len(symbol_parts) > 1:
                            symbol_name = symbol_parts[1].split('-')[0]
                            if symbol_name != name:
                                match = False
                        else:
                            match = False
                    else:
                        # For partial match, just check if name is a substring
                        if name.upper() not in symbol_str.upper():
                            match = False
                
                if exchange:
                    # Extract exchange from symbol string (e.g., "NSE" from "NSE:SBIN-EQ")
                    symbol_parts = symbol_str.split(':')
                    if len(symbol_parts) > 0:
                        symbol_exchange = symbol_parts[0]
                        if symbol_exchange != exchange:
                            match = False
                    else:
                        match = False
                    
                if symbol_type:
                    # Extract symbol type from symbol string (e.g., "EQ" from "NSE:SBIN-EQ")
                    symbol_parts = symbol_str.split('-')
                    if len(symbol_parts) > 1:
                        symbol_type_value = symbol_parts[1]
                        if symbol_type_value != symbol_type:
                            match = False
                    else:
                        match = False
                
                if match:
                    # Add the symbol string as a key in the data for easy reference
                    result_item = {"symbol_str": symbol_str}
                    if isinstance(symbol_data, dict):
                        result_item.update(symbol_data)
                    results.append(result_item)
        
        # Cache the results
        self.lookup_cache[cache_key] = results
        return results
    
    def get_symbol_info(self, symbol_str: str) -> Optional[Dict]:
        """
        Get detailed information for a specific symbol string.
        
        Args:
            symbol_str: Symbol string (e.g., "NSE:SBIN-EQ")
            
        Returns:
            Dictionary with symbol information or None if not found
        """
        try:
            # First, determine which segment might contain this symbol
            for segment, symbols in self.symbols.items():
                if symbol_str in symbols:
                    result = {"symbol_str": symbol_str}
                    symbol_data = symbols[symbol_str]
                    if isinstance(symbol_data, dict):
                        result.update(symbol_data)
                    return result
            
            # If not found directly, try to search
            exchange = None
            name = None
            symbol_type = None
            
            # Parse the symbol string
            parts = symbol_str.split(':')
            if len(parts) == 2:
                exchange = parts[0]
                remaining = parts[1]
                
                # Check for symbol type
                type_parts = remaining.split('-')
                if len(type_parts) > 1:
                    name = type_parts[0]
                    symbol_type = type_parts[1]
                else:
                    name = remaining
            
            # Search with the parsed components
            matches = self.search(
                name=name,
                exchange=exchange,
                symbol_type=symbol_type,
                exact_match=True
            )
            
            if matches:
                return matches[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol_str}: {str(e)}")
            return None
    
    def format_symbol(self, symbol_info: Dict) -> str:
        """
        Format a symbol dictionary into the FYERS API format.
        
        Args:
            symbol_info: Symbol dictionary from the symbol master
            
        Returns:
            Formatted symbol string (e.g., "NSE:SBIN-EQ")
        """
        # If the symbol_str key is present, use it directly
        if "symbol_str" in symbol_info:
            return symbol_info["symbol_str"]
        
        # Otherwise, try to construct it from component parts
        exchange = symbol_info.get('exchange', '')
        name = symbol_info.get('name', '')
        symbol_type = symbol_info.get('symbol_type', '')
        
        if symbol_type:
            return f"{exchange}:{name}-{symbol_type}"
        else:
            return f"{exchange}:{name}"
    
    def get_all_symbols(self, 
                       exchange: str = None, 
                       symbol_type: str = None,
                       segment: str = None) -> List[str]:
        """
        Get all symbol strings matching specified criteria.
        
        Args:
            exchange: Filter by exchange (e.g., "NSE", "BSE")
            symbol_type: Filter by symbol type (e.g., "EQ", "FUT")
            segment: Filter by segment (e.g., "NSE_CM")
            
        Returns:
            List of formatted symbol strings
        """
        symbol_infos = self.search(
            exchange=exchange,
            symbol_type=symbol_type,
            segment=segment
        )
        
        return [self.format_symbol(info) for info in symbol_infos]
    
    def get_fytoken(self, symbol_str: str) -> Optional[str]:
        """
        Get the fyToken for a symbol string.
        
        Args:
            symbol_str: Symbol string (e.g., "NSE:SBIN-EQ")
            
        Returns:
            fyToken string or None if not found
        """
        symbol_info = self.get_symbol_info(symbol_str)
        if symbol_info:
            return symbol_info.get('fyToken')
        return None
    
    def get_symbol_by_fytoken(self, fytoken: str) -> Optional[Dict]:
        """
        Find a symbol by its fyToken.
        
        Args:
            fytoken: FYERS token for the symbol
            
        Returns:
            Symbol dictionary or None if not found
        """
        # Check cache first
        cache_key = f"fytoken:{fytoken}"
        if cache_key in self.lookup_cache:
            return self.lookup_cache[cache_key]
        
        # Search across all segments
        for segment, symbols in self.symbols.items():
            for symbol_str, symbol_data in symbols.items():
                if isinstance(symbol_data, dict) and symbol_data.get('fyToken') == fytoken:
                    # Create result with symbol string
                    result = {"symbol_str": symbol_str}
                    result.update(symbol_data)
                    
                    # Cache the result
                    self.lookup_cache[cache_key] = result
                    return result
        
        return None
    
    def refresh_symbols(self, force: bool = True) -> Dict[str, bool]:
        """
        Refresh symbol data by downloading the latest files.
        
        Args:
            force: Whether to force download even if files exist
            
        Returns:
            Dict mapping segment names to refresh success (True/False)
        """
        # Clear caches
        self.lookup_cache = {}
        
        # Download fresh data
        return self.download_all_symbol_files(force=force)
    
    def get_segments(self) -> List[str]:
        """
        Get list of available segments.
        
        Returns:
            List of segment names
        """
        return list(self.symbols.keys())
    
    def get_exchanges(self) -> Set[str]:
        """
        Get set of all available exchanges.
        
        Returns:
            Set of exchange names
        """
        exchanges = set()
        for segment, symbols in self.symbols.items():
            for symbol_str in symbols.keys():
                parts = symbol_str.split(':')
                if len(parts) > 0:
                    exchanges.add(parts[0])
        return exchanges
    
    def get_symbol_types(self) -> Set[str]:
        """
        Get set of all available symbol types.
        
        Returns:
            Set of symbol types
        """
        types = set()
        for segment, symbols in self.symbols.items():
            for symbol_str in symbols.keys():
                parts = symbol_str.split('-')
                if len(parts) > 1:
                    types.add(parts[1])
        return types
    
    def __str__(self) -> str:
        """String representation showing loaded segments and counts."""
        parts = ["SymbolManager:"]
        for segment, symbols in self.symbols.items():
            parts.append(f"  {segment}: {len(symbols)} symbols")
        return "\n".join(parts)