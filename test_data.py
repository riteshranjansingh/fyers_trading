"""
Test script for FYERS API data fetching
"""
from src.api.connection import FyersConnection
from src.api.data import FyersData
import matplotlib.pyplot as plt

def main():
    # Initialize connection
    print("Connecting to FYERS API...")
    conn = FyersConnection()
    session = conn.connect()
    
    if not session:
        print("Failed to connect to FYERS API. Exiting.")
        return
    
    # Initialize data handler
    data_handler = FyersData(session)
    
    # Fetch historical data
    symbol = "NSE:SBIN-EQ"  # State Bank of India
    print(f"Fetching historical data for {symbol}...")
    
    df = data_handler.get_historical_data(symbol, resolution="D", days_back=30)
    
    if df is not None and not df.empty:
        print(f"Successfully fetched {len(df)} records")
        print("\nFirst 5 records:")
        print(df.head())
        
        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['close'])
        plt.title(f"{symbol} - Close Price (Last 30 Days)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{symbol.replace(':', '_')}_chart.png")
        print(f"Chart saved as {symbol.replace(':', '_')}_chart.png")
        
        # Try to get current quote
        print(f"\nFetching current quote for {symbol}...")
        quote = data_handler.get_quote(symbol)
        if quote:
            print(f"Current quote: {quote}")
        else:
            print("Failed to fetch current quote")
    else:
        print("Failed to fetch historical data")

if __name__ == "__main__":
    main()