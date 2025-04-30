import requests
import pandas as pd
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_stock_data(self, symbol: str, interval: str = 'daily') -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Alpha Vantage
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with stock data or None if request fails
        """
        try:
            function = f"TIME_SERIES_{interval.upper()}"
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            time_series_key = f"Time Series ({interval.title()})"
            
            if time_series_key not in data:
                logger.error(f"No time series data found for {symbol}")
                return None
                
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            
            # Convert string values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(',',''), errors='coerce')
            
            logger.info(f"Successfully fetched {interval} data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
    
    def fetch_crypto_data(self, symbol: str, market: str = 'USD') -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data from Alpha Vantage
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            market: Market currency (e.g., 'USD')
            
        Returns:
            DataFrame with crypto data or None if request fails
        """
        try:
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': symbol,
                'market': market,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "Time Series (Digital Currency Daily)" not in data:
                logger.error(f"No time series data found for {symbol}")
                return None
                
            df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient='index')
            
            # Rename columns to remove market currency
            df.columns = [col.split(' ')[1] for col in df.columns]
            df.index = pd.to_datetime(df.index)
            
            # Convert string values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(',',''), errors='coerce')
            
            logger.info(f"Successfully fetched data for {symbol}/{market}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching crypto data: {str(e)}")
            return None