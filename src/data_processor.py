import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to process stock and cryptocurrency data and generate technical indicators"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to price data
        
        Args:
            df: DataFrame with OHLCV data (must have columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        # Create a copy to avoid modifying the original DataFrame
        result = df.copy()
        
        # Simple Moving Averages
        result['sma_5'] = result['close'].rolling(window=5).mean()
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # RSI (Relative Strength Index)
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        result['bb_std'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        
        # Average True Range (ATR)
        high_low = result['high'] - result['low']
        high_close = (result['high'] - result['close'].shift()).abs()
        low_close = (result['low'] - result['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result['atr_14'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        result['volume_sma_20'] = result['volume'].rolling(window=20).mean()
        
        # Price change features
        result['returns_1d'] = result['close'].pct_change()
        result['returns_5d'] = result['close'].pct_change(periods=5)
        
        # Log returns (better for statistical analysis)
        result['log_returns_1d'] = np.log(result['close'] / result['close'].shift(1))
        
        # Volatility (standard deviation of returns)
        result['volatility_14d'] = result['returns_1d'].rolling(window=14).std()
        
        # Fill NaN values that may have been introduced by the rolling calculations
        result = result.dropna()
        
        logger.info(f"Added technical indicators to dataframe with shape: {result.shape}")
        return result
    
    @staticmethod
    def add_target_variable(df: pd.DataFrame, forward_periods: int = 1) -> pd.DataFrame:
        """
        Add target variable for price movement prediction
        
        Args:
            df: DataFrame with price data
            forward_periods: Number of periods ahead to predict
            
        Returns:
            DataFrame with target variables added
        """
        result = df.copy()
        
        # Price n periods in the future
        result[f'future_price_{forward_periods}d'] = result['close'].shift(-forward_periods)
        
        # Binary classification target (1 if price goes up, 0 if it goes down)
        result[f'target_direction_{forward_periods}d'] = (
            result[f'future_price_{forward_periods}d'] > result['close']).astype(int)
        
        # Regression target (percentage change)
        result[f'target_return_{forward_periods}d'] = (
            result[f'future_price_{forward_periods}d'] / result['close'] - 1)
        
        # Remove rows with NaN target values
        result = result.dropna(subset=[f'future_price_{forward_periods}d'])
        
        logger.info(f"Added target variables for {forward_periods} periods ahead prediction")
        return result
    
    @staticmethod
    def prepare_features_targets(df: pd.DataFrame, 
                               target_col: str,
                               feature_cols: Optional[List[str]] = None) -> tuple:
        """
        Prepare feature and target arrays for machine learning
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            feature_cols: List of feature column names (if None, use all except targets)
            
        Returns:
            Tuple of (X, y) arrays for machine learning
        """
        # If no feature columns specified, use all except target columns
        if feature_cols is None:
            # Exclude columns that contain 'target' or 'future' in their names
            feature_cols = [col for col in df.columns if 'target' not in col and 'future' not in col]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        logger.info(f"Prepared features with shape {X.shape} and targets with shape {y.shape}")
        return X, y
    
    @staticmethod
    def split_data(df: pd.DataFrame, train_size: float = 0.8) -> tuple:
        """
        Split data into training and testing sets chronologically
        
        Args:
            df: DataFrame to split
            train_size: Proportion of data to use for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_size)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Split data into train set with {len(train_df)} samples and test set with {len(test_df)} samples")
        return train_df, test_df