# main.py
from data_fetcher import AlphaVantageAPI
import config
import logging
import os
from pathlib import Path

def setup_data_directory():
    # Get the current file's directory
    current_dir = Path(__file__).parent.parent
    # Create data directory path
    data_dir = current_dir / 'data'
    # Create the directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    return data_dir

def main():
    # Setup data directory
    data_dir = setup_data_directory()
    
    # Initialize API client
    api = AlphaVantageAPI(config.API_KEY)
    
    # Example: Fetch stock data
    stock_data = api.fetch_stock_data('AAPL')
    if stock_data is not None:
        print("\nStock Data Sample:")
        print(stock_data.head())
        stock_data.to_csv(data_dir / 'aapl_stock_data.csv')
        print(f"Stock data saved to {data_dir / 'aapl_stock_data.csv'}")
    
    # Example: Fetch crypto data
    crypto_data = api.fetch_crypto_data('BTC')
    if crypto_data is not None:
        print("\nCrypto Data Sample:")
        print(crypto_data.head())
        crypto_data.to_csv(data_dir / 'btc_crypto_data.csv')
        print(f"Crypto data saved to {data_dir / 'btc_crypto_data.csv'}")

if __name__ == "__main__":
    main()