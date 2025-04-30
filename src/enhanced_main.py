import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
import argparse

# Import project modules
from data_fetcher import AlphaVantageAPI
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from trading_strategy import MLStrategy, TechnicalStrategy
from visualizer import Visualizer
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Set up project directories"""
    # Get project root directory
    root_dir = Path(__file__).parent.parent
    
    # Create necessary directories
    data_dir = root_dir / 'data'
    models_dir = root_dir / 'models'
    figures_dir = root_dir / 'figures'
    
    # Create directories if they don't exist
    try:
        data_dir.mkdir(exist_ok=True)
    except FileNotFoundError:
        if not data_dir.is_dir():
            data_dir = root_dir / 'data_new'
            data_dir.mkdir(exist_ok=True)
            logger.warning(f"'data' exists but is not a directory. Using {data_dir} instead.")
    
    try:
        models_dir.mkdir(exist_ok=True)
    except FileNotFoundError:
        if not models_dir.is_dir():
            models_dir = root_dir / 'models_new'
            models_dir.mkdir(exist_ok=True)
            logger.warning(f"'models' exists but is not a directory. Using {models_dir} instead.")
    
    try:
        figures_dir.mkdir(exist_ok=True)
    except FileExistsError:
        if not figures_dir.is_dir():
            figures_dir = root_dir / 'figures_new'
            figures_dir.mkdir(exist_ok=True)
            logger.warning(f"'figures' exists but is not a directory. Using {figures_dir} instead.")
    
    return {
        'root': root_dir,
        'data': data_dir,
        'models': models_dir,
        'figures': figures_dir
    }

def fetch_data(symbol, data_type, interval='daily'):
    """Fetch data from Alpha Vantage API"""
    api = AlphaVantageAPI(config.API_KEY)
    
    if data_type.lower() == 'stock':
        data = api.fetch_stock_data(symbol, interval)
    elif data_type.lower() == 'crypto':
        data = api.fetch_crypto_data(symbol)
    else:
        logger.error(f"Invalid data type: {data_type}")
        return None
    
    # Check if we have enough data
    if data is not None and len(data) < 100:
        logger.warning(f"Retrieved only {len(data)} rows for {symbol}. This may not be enough for reliable analysis.")
    
    return data

def load_data(data_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {data_path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        return None

def preprocess_data(data, prediction_horizon=5):
    """Preprocess data and add technical indicators"""
    processor = DataProcessor()
    
    # Check if we have enough data
    if len(data) < 100:
        logger.warning(f"Dataset is very small ({len(data)} rows). Results may not be reliable.")
        
    # Add technical indicators
    data_with_indicators = processor.add_technical_indicators(data)
    
    # Check how many rows we have after adding indicators
    logger.info(f"Data shape after adding indicators: {data_with_indicators.shape}")
    
    # Add target variables for prediction
    data_with_targets = processor.add_target_variable(
        data_with_indicators, forward_periods=prediction_horizon)
    
    # Check how many rows we have after adding targets
    logger.info(f"Data shape after adding target variables: {data_with_targets.shape}")
    
    # Use a larger portion for training when dataset is small
    train_size = 0.7 if len(data_with_targets) >= 100 else 0.8
    
    # Split data into training and testing sets
    train_data, test_data = processor.split_data(data_with_targets, train_size=train_size)
    
    return train_data, test_data

def train_model(train_data, model_name, target_col, prediction_type='classification'):
    """Train a machine learning model"""
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Select feature columns (exclude target variables)
    feature_cols = [col for col in train_data.columns 
                   if not (col.startswith('target') or col.startswith('future'))]
    
    # Log the training data size
    logger.info(f"Training data shape: {train_data.shape}")
    
    # Skip a smaller number of rows to avoid NaN values from indicators
    # Use a percentage of the data instead of a fixed number
    skip_rows = min(20, int(len(train_data) * 0.2))  # Skip the minimum of 20 rows or 20% of the data
    logger.info(f"Skipping first {skip_rows} rows to avoid NaN values")
    
    # Prepare features and targets
    X_train, y_train = processor.prepare_features_targets(
        train_data.iloc[skip_rows:], target_col, feature_cols)
    
    # Check if we have enough data to train
    if len(X_train) == 0:
        logger.error("No training samples available. Try using more data or reducing the prediction horizon.")
        raise ValueError("No training samples available.")
    
    logger.info(f"Final training set shape: X={X_train.shape}, y={y_train.shape}")
    
    # Train model
    if prediction_type == 'classification':
        trainer.train_classification_model(
            X_train, y_train, feature_cols, model_type='random_forest')
    
    # Save model
    trainer.save_model(model_name)
    
    return trainer

def evaluate_model(model, test_data, target_col):
    """Evaluate model performance"""
    processor = DataProcessor()
    
    # Select feature columns
    feature_cols = model.feature_names
    
    # Dynamically determine the number of rows to exclude based on NaN values
    first_valid_index = test_data.dropna().index[0]
    X_test, y_test = processor.prepare_features_targets(
        test_data.loc[first_valid_index:], target_col, feature_cols)
    
    # Evaluate model
    metrics = model.evaluate_classification_model(X_test, y_test)
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    
    return metrics, importance_df

def backtest_strategy(strategy, data, commission=0.001, position_size=0.95,
                     stop_loss=None, take_profit=None):
    """Backtest trading strategy"""
    # Generate signals and perform backtest
    performance = strategy.backtest(
        data, 
        commission=commission,
        position_size_pct=position_size,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    return performance, strategy.trades, strategy.equity_curve

def visualize_results(data, test_data, importance_df, equity_curve, trades):
    """Visualize data and results"""
    viz = Visualizer()
    
    # Plot price chart with moving averages
    viz.plot_price_chart(
        data.iloc[-200:],
        title="Price Chart with Moving Averages",
        moving_averages=['sma_20', 'sma_50'],
        save_path="price_chart.png"
    )
    
    # Plot technical indicators
    indicators = {
        "Oscillators": ["rsi_14", "macd", "macd_signal"],
        "Volatility": ["bb_upper", "bb_middle", "bb_lower", "atr_14"]
    }
    viz.plot_technical_indicators(
        data.iloc[-200:],
        indicators,
        title="Technical Indicators",
        save_path="technical_indicators.png"
    )
    
    # Plot feature importance
    if importance_df is not None:
        viz.plot_feature_importance(
            importance_df,
            title="Feature Importance",
            top_n=15,
            save_path="feature_importance.png"
        )
    
    # Plot backtest results
    if equity_curve is not None:
        viz.plot_backtest_results(
            equity_curve,
            trades,
            title="Backtest Results",
            save_path="backtest_results.png"
        )
        
        # Plot returns distribution
        if 'returns' in equity_curve.columns:
            viz.plot_returns_distribution(
                equity_curve['returns'].dropna(),
                title="Returns Distribution",
                save_path="returns_distribution.png"
            )
    
    # Plot correlation matrix of features
    feature_cols = [col for col in data.columns 
                   if not (col.startswith('target') or col.startswith('future'))]
    viz.plot_correlation_matrix(
        data[feature_cols].iloc[-100:],
        title="Feature Correlation Matrix",
        save_path="correlation_matrix.png"
    )

def main(args):
    """Main function to run the trading system"""
    # Set up directories
    dirs = setup_directories()
    
    # Load or fetch data
    if args.data_path:
        data = load_data(args.data_path)
    else:
        data = fetch_data(args.symbol, args.data_type, args.interval)
        
        # Save data to CSV
        if data is not None:
            file_name = f"{args.symbol.lower()}_{args.data_type.lower()}_data.csv"
            data_path = dirs['data'] / file_name
            data.to_csv(data_path)
            logger.info(f"Data saved to {data_path}")
    
    if data is None:
        logger.error("Failed to load or fetch data. Exiting.")
        return
    
    # Preprocess data
    train_data, test_data = preprocess_data(data, args.prediction_horizon)
    
    # Define target column for prediction
    target_col = f"target_direction_{args.prediction_horizon}d"
    
    # Train or load model
    if args.load_model:
        trainer = ModelTrainer()
        try:
            trainer.load_model(args.model_name)
            logger.info(f"Loaded model {args.model_name}")
        except FileNotFoundError:
            logger.warning(f"Model {args.model_name} not found. Training new model.")
            trainer = train_model(train_data, args.model_name, target_col)
    else:
        trainer = train_model(train_data, args.model_name, target_col)
    
    # Evaluate model
    metrics, importance_df = evaluate_model(trainer, test_data, target_col)
    logger.info(f"Model evaluation metrics: {metrics}")
    
    # Initialize trading strategy
    if args.strategy_type == 'ml':
        strategy = MLStrategy(
            model=trainer, 
            threshold=args.threshold,
            initial_balance=10000.0  # Using default value
        )
    else:
        strategy = TechnicalStrategy()
    
    # Backtest strategy
    performance, trades, equity_curve = backtest_strategy(
        strategy, test_data, 
        commission=args.commission,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit
    )
    
    logger.info(f"Strategy performance: {performance}")
    
    # Visualize results
    visualize_results(data, test_data, importance_df, equity_curve, trades)
    
    logger.info("Trading system run completed!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="AI Trading System")
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data-path', type=str, help='Path to data CSV file')
    data_group.add_argument('--symbol', type=str, default='AAPL', help='Stock or crypto symbol')
    data_group.add_argument('--data-type', type=str, default='stock', choices=['stock', 'crypto'], help='Type of data')
    data_group.add_argument('--interval', type=str, default='daily', choices=['daily', 'weekly', 'monthly'], help='Data interval')
    
    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--load-model', action='store_true', help='Load existing model instead of training')
    model_group.add_argument('--model-name', type=str, default='stock_predictor', help='Model name to save or load')
    model_group.add_argument('--prediction-horizon', type=int, default=5, help='Number of days to predict ahead')
    
    # Strategy options
    strategy_group = parser.add_argument_group('Strategy Options')
    strategy_group.add_argument('--strategy-type', type=str, default='ml', choices=['ml', 'technical'], help='Trading strategy type')
    strategy_group.add_argument('--threshold', type=float, default=0.6, help='Probability threshold for ML strategy')
    strategy_group.add_argument('--commission', type=float, default=0.001, help='Trading commission rate')
    strategy_group.add_argument('--position-size', type=float, default=0.95, help='Position size as fraction of account')
    strategy_group.add_argument('--stop-loss', type=float, default=None, help='Stop loss percentage')
    strategy_group.add_argument('--take-profit', type=float, default=None, help='Take profit percentage')
    
    args = parser.parse_args()
    
    # Run main function
    main(args)