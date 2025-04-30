import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Position(Enum):
    """Enum for position types"""
    FLAT = 0
    LONG = 1
    SHORT = 2

class TradeAction(Enum):
    """Enum for trade actions"""
    HOLD = 0
    BUY = 1
    SELL = 2

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize trading strategy
        
        Args:
            initial_balance: Initial account balance
        """
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset trading account and positions"""
        self.balance = self.initial_balance
        self.position = Position.FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on strategy
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with added signal column
        """
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def backtest(self, data: pd.DataFrame, commission: float = 0.001, 
                position_size_pct: float = 0.95, stop_loss_pct: float = None,
                take_profit_pct: float = None) -> Dict:
        """
        Backtest trading strategy on historical data
        
        Args:
            data: DataFrame with price data and signals
            commission: Commission rate per trade (as a percentage)
            position_size_pct: Position size as a percentage of account balance
            stop_loss_pct: Stop loss percentage (None for no stop loss)
            take_profit_pct: Take profit percentage (None for no take profit)
            
        Returns:
            Dictionary of backtest performance metrics
        """
        # Reset trading account
        self.reset()
        
        # Generate trading signals if not already present
        if 'signal' not in data.columns:
            data = self.generate_signals(data)
        
        # Add columns for tracking positions and equity
        results = data.copy()
        results['position'] = Position.FLAT.value
        results['trade_action'] = TradeAction.HOLD.value
        results['balance'] = self.initial_balance
        results['equity'] = self.initial_balance
        results['returns'] = 0.0
        
        # Initialize tracking variables
        current_position = Position.FLAT
        entry_price = 0.0
        entry_balance = self.initial_balance
        entry_date = None
        
        # Iterate through each time step
        for i in range(1, len(results)):
            current_price = results.iloc[i]['close']
            prev_price = results.iloc[i-1]['close']
            signal = results.iloc[i]['signal']
            
            # Default is to hold the current position
            trade_action = TradeAction.HOLD
            
            # If flat and signal to buy
            if current_position == Position.FLAT and signal > 0:
                current_position = Position.LONG
                trade_action = TradeAction.BUY
                entry_price = current_price
                entry_date = results.index[i]
                entry_balance = results.iloc[i-1]['balance']
                # Calculate position size
                position_size = entry_balance * position_size_pct
                
                # Apply commission
                commission_cost = position_size * commission
                results.iloc[i, results.columns.get_loc('balance')] = entry_balance - commission_cost
            
            # If long and signal to sell
            elif current_position == Position.LONG and signal < 0:
                # Calculate profit/loss
                position_size = entry_balance * position_size_pct
                exit_value = position_size * (current_price / entry_price)
                
                # Apply commission
                commission_cost = exit_value * commission
                pnl = exit_value - position_size - commission_cost
                
                # Update balance
                new_balance = results.iloc[i-1]['balance'] + pnl
                results.iloc[i, results.columns.get_loc('balance')] = new_balance
                
                # Record trade
                self.trades.append({
                    'entry_date': entry_date,
                    'exit_date': results.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': current_position.name,
                    'pnl': pnl,
                    'return_pct': (pnl / position_size) * 100
                })
                
                # Reset position
                current_position = Position.FLAT
                trade_action = TradeAction.SELL
                entry_price = 0.0
            
            # Check for stop loss or take profit if long
            elif current_position == Position.LONG:
                position_size = entry_balance * position_size_pct
                unrealized_return_pct = (current_price / entry_price - 1) * 100
                
                # Check stop loss
                if stop_loss_pct is not None and unrealized_return_pct <= -stop_loss_pct:
                    # Calculate profit/loss
                    exit_value = position_size * (current_price / entry_price)
                    
                    # Apply commission
                    commission_cost = exit_value * commission
                    pnl = exit_value - position_size - commission_cost
                    
                    # Update balance
                    new_balance = results.iloc[i-1]['balance'] + pnl
                    results.iloc[i, results.columns.get_loc('balance')] = new_balance
                    
                    # Record trade
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': results.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position.name,
                        'pnl': pnl,
                        'return_pct': (pnl / position_size) * 100,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # Reset position
                    current_position = Position.FLAT
                    trade_action = TradeAction.SELL
                    entry_price = 0.0
                
                # Check take profit
                elif take_profit_pct is not None and unrealized_return_pct >= take_profit_pct:
                    # Calculate profit/loss
                    exit_value = position_size * (current_price / entry_price)
                    
                    # Apply commission
                    commission_cost = exit_value * commission
                    pnl = exit_value - position_size - commission_cost
                    
                    # Update balance
                    new_balance = results.iloc[i-1]['balance'] + pnl
                    results.iloc[i, results.columns.get_loc('balance')] = new_balance
                    
                    # Record trade
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': results.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': current_position.name,
                        'pnl': pnl,
                        'return_pct': (pnl / position_size) * 100,
                        'exit_reason': 'take_profit'
                    })
                    
                    # Reset position
                    current_position = Position.FLAT
                    trade_action = TradeAction.SELL
                    entry_price = 0.0
            
            # Update position and trade action
            results.iloc[i, results.columns.get_loc('position')] = current_position.value
            results.iloc[i, results.columns.get_loc('trade_action')] = trade_action.value
            
            # If balance was not updated (no trade), carry forward the previous balance
            if results.iloc[i]['balance'] == self.initial_balance and i > 0:
                results.iloc[i, results.columns.get_loc('balance')] = results.iloc[i-1]['balance']
            
            # Calculate equity (balance + unrealized P&L)
            if current_position == Position.LONG:
                position_size = entry_balance * position_size_pct
                unrealized_pnl = position_size * (current_price / entry_price - 1)
                results.iloc[i, results.columns.get_loc('equity')] = results.iloc[i]['balance'] + unrealized_pnl
            else:
                results.iloc[i, results.columns.get_loc('equity')] = results.iloc[i]['balance']
            
            # Calculate returns
            if i > 0:
                results.iloc[i, results.columns.get_loc('returns')] = (
                    results.iloc[i]['equity'] / results.iloc[i-1]['equity'] - 1
                )
        
        # Store equity curve
        self.equity_curve = results[['equity', 'returns']].copy()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results)
        
        logger.info(f"Backtest completed with final equity: ${metrics['final_equity']:.2f}")
        return metrics
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate trading performance metrics
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        # Convert trades list to DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Calculate basic metrics
        initial_equity = self.initial_balance
        final_equity = results['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        # Calculate drawdown
        results['drawdown'] = results['equity'] / results['equity'].cummax() - 1
        max_drawdown = results['drawdown'].min() * 100
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        daily_returns = results['returns'].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        
        # Trading metrics
        total_trades = len(self.trades)
        
        if total_trades > 0:
            winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
            win_rate = (winning_trades / total_trades) * 100
            
            # Average profit per winning trade
            avg_win = np.mean([t['return_pct'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
            
            # Average loss per losing trade
            losing_trades = total_trades - winning_trades
            avg_loss = np.mean([t['return_pct'] for t in self.trades if t['pnl'] <= 0]) if losing_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Max consecutive wins and losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_consecutive_wins = 0
            current_consecutive_losses = 0
            
            for trade in self.trades:
                if trade['pnl'] > 0:
                    current_consecutive_wins += 1
                    current_consecutive_losses = 0
                    consecutive_wins = max(consecutive_wins, current_consecutive_wins)
                else:
                    current_consecutive_losses += 1
                    current_consecutive_wins = 0
                    consecutive_losses = max(consecutive_losses, current_consecutive_losses)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            consecutive_wins = 0
            consecutive_losses = 0
        
        return {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'annualized_return_pct': ((1 + total_return / 100) ** (252 / len(results)) - 1) * 100 if len(results) > 0 else 0,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }


class MLStrategy(TradingStrategy):
    """Trading strategy based on ML model predictions"""
    
    def __init__(self, model, threshold: float = 0.5, initial_balance: float = 10000.0):
        """
        Initialize ML-based trading strategy
        
        Args:
            model: Trained ML model with predict method
            threshold: Probability threshold for taking long positions
            initial_balance: Initial account balance
        """
        super().__init__(initial_balance)
        self.model = model
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ML predictions
        
        Args:
            data: DataFrame with price and feature data
            
        Returns:
            DataFrame with added signal column
        """
        # Make a copy of the data
        result = data.copy()
        
        # Get feature columns used by the model
        feature_cols = self.model.feature_names
        
        # Make predictions
        if hasattr(self.model, 'predict'):
            X = data[feature_cols].values
            probabilities = self.model.predict(X)
            
            # Generate signals based on prediction probability
            result['prediction_prob'] = probabilities
            result['signal'] = 0
            
            # Long signal when probability exceeds threshold
            result.loc[probabilities > self.threshold, 'signal'] = 1
            
            # Exit signal when probability falls below threshold
            result.loc[probabilities <= self.threshold, 'signal'] = -1
            
            logger.info(f"Generated signals using ML model with threshold {self.threshold}")
            return result
        else:
            raise ValueError("Model object does not have a predict method")


class TechnicalStrategy(TradingStrategy):
    """Trading strategy based on technical indicators"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with added signal column
        """
        # Make a copy of the data
        result = data.copy()
        
        # Initialize signal column
        result['signal'] = 0
        
        # Check if we have the necessary indicators
        required_indicators = ['sma_20', 'sma_50', 'rsi_14']
        if not all(indicator in result.columns for indicator in required_indicators):
            raise ValueError(f"Data missing required indicators: {required_indicators}")
        
        # Generate signals based on indicator crossovers
        for i in range(1, len(result)):
            # Moving Average Crossover: Buy when SMA20 crosses above SMA50
            if (result['sma_20'].iloc[i-1] <= result['sma_50'].iloc[i-1] and 
                result['sma_20'].iloc[i] > result['sma_50'].iloc[i]):
                result.iloc[i, result.columns.get_loc('signal')] = 1
            
            # Moving Average Crossover: Sell when SMA20 crosses below SMA50
            elif (result['sma_20'].iloc[i-1] >= result['sma_50'].iloc[i-1] and 
                  result['sma_20'].iloc[i] < result['sma_50'].iloc[i]):
                result.iloc[i, result.columns.get_loc('signal')] = -1
            
            # RSI Overbought: Sell when RSI goes above 70
            elif result['rsi_14'].iloc[i] > 70:
                result.iloc[i, result.columns.get_loc('signal')] = -1
            
            # RSI Oversold: Buy when RSI goes below 30
            elif result['rsi_14'].iloc[i] < 30:
                result.iloc[i, result.columns.get_loc('signal')] = 1
        
        logger.info("Generated signals using technical indicators")
        return result