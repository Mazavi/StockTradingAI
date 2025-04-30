import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """Class to visualize price data, technical indicators, and backtest results"""
    
    def __init__(self, figures_dir: Union[str, Path] = 'figures'):
        """
        Initialize visualizer
        
        Args:
            figures_dir: Directory to save figures
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
    
    def plot_price_chart(self, data: pd.DataFrame, title: str = "Price Chart", 
                       moving_averages: List[str] = None, save_path: Optional[str] = None):
        """
        Plot price chart with optional moving averages
        
        Args:
            data: DataFrame with price data (must have 'close' column)
            title: Chart title
            moving_averages: List of moving average column names to include
            save_path: Path to save the figure, if None, display the figure
        """
        plt.figure(figsize=(12, 6))
        
        # Plot close price
        plt.plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        
        # Plot moving averages if provided
        if moving_averages:
            for ma in moving_averages:
                if ma in data.columns:
                    plt.plot(data.index, data[ma], label=ma.upper(), linewidth=1)
        
        # Set title and labels
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            full_path = self.figures_dir / save_path
            plt.savefig(full_path, dpi=300)
            logger.info(f"Price chart saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_technical_indicators(self, data: pd.DataFrame, indicators: Dict[str, List[str]], 
                                title: str = "Technical Indicators", save_path: Optional[str] = None):
        """
        Plot price chart with multiple technical indicators
        
        Args:
            data: DataFrame with price and indicator data
            indicators: Dictionary mapping subplot names to indicator columns
            title: Chart title
            save_path: Path to save the figure, if None, display the figure
        """
        n_subplots = len(indicators) + 1  # +1 for price chart
        
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots), sharex=True)
        fig.suptitle(title, fontsize=16)
        
        # Plot price chart on the first subplot
        axes[0].plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        axes[0].set_ylabel('Price', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot indicators on subsequent subplots
        i = 1
        for subplot_title, indicator_list in indicators.items():
            for indicator in indicator_list:
                if indicator in data.columns:
                    axes[i].plot(data.index, data[indicator], label=indicator, linewidth=1)
            
            axes[i].set_ylabel(subplot_title, fontsize=12)
            axes[i].legend()
            axes[i].grid(alpha=0.3)
            i += 1
        
        # Format x-axis dates for the last subplot
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save or display figure
        if save_path:
            full_path = self.figures_dir / save_path
            plt.savefig(full_path, dpi=300)
            logger.info(f"Technical indicators chart saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_backtest_results(self, equity_curve: pd.DataFrame, trades: List[Dict] = None, 
                           title: str = "Backtest Results", save_path: Optional[str] = None):
        """
        Plot backtest results with equity curve and trades
        
        Args:
            equity_curve: DataFrame with equity values over time
            trades: List of trade dictionaries with entry/exit dates and prices
            title: Chart title
            save_path: Path to save the figure, if None, display the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(title, fontsize=16)
        
        # Plot equity curve
        axes[0].plot(equity_curve.index, equity_curve['equity'], label='Equity', linewidth=1.5)
        axes[0].set_ylabel('Equity ($)', fontsize=12)
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        # Calculate drawdown
        if 'equity' in equity_curve.columns:
            drawdown = equity_curve['equity'] / equity_curve['equity'].cummax() - 1
            axes[1].fill_between(drawdown.index, 0, drawdown * 100, color='red', alpha=0.3, label='Drawdown')
            axes[1].set_ylabel('Drawdown (%)', fontsize=12)
            axes[1].set_ylim(drawdown.min() * 100 * 1.5, 5)  # Add some padding to y-axis
            axes[1].grid(alpha=0.3)
            axes[1].legend()
        
        # Mark trades on the equity curve if provided
        if trades:
            for trade in trades:
                if trade['pnl'] > 0:
                    color = 'green'
                    marker = '^'  # Up triangle for winning trades
                else:
                    color = 'red'
                    marker = 'v'  # Down triangle for losing trades
                
                if 'entry_date' in trade and 'exit_date' in trade:
                    entry_date = pd.to_datetime(trade['entry_date'])
                    exit_date = pd.to_datetime(trade['exit_date'])
                    
                    # Find equity at entry and exit
                    if entry_date in equity_curve.index and exit_date in equity_curve.index:
                        entry_equity = equity_curve.loc[entry_date, 'equity']
                        exit_equity = equity_curve.loc[exit_date, 'equity']
                        
                        # Mark entry and exit points
                        axes[0].scatter(entry_date, entry_equity, color=color, marker=marker, s=50, alpha=0.7)
                        axes[0].scatter(exit_date, exit_equity, color=color, marker='o', s=50, alpha=0.7)
                        
                        # Connect entry and exit points with a line
                        axes[0].plot([entry_date, exit_date], [entry_equity, exit_equity], color=color, alpha=0.5)
        
        # Format x-axis dates
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save or display figure
        if save_path:
            full_path = self.figures_dir / save_path
            plt.savefig(full_path, dpi=300)
            logger.info(f"Backtest results saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, title: str = "Feature Importance", 
                              top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance from ML model
        
        Args:
            importance_df: DataFrame with feature names and importance values
            title: Chart title
            top_n: Number of top features to display
            save_path: Path to save the figure, if None, display the figure
        """
        # Sort by importance and get top N features
        top_features = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        # Create horizontal bar chart
        sns.barplot(x='importance', y='feature', data=top_features)
        
        # Set title and labels
        plt.title(title, fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            full_path = self.figures_dir / save_path
            plt.savefig(full_path, dpi=300)
            logger.info(f"Feature importance chart saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_returns_distribution(self, returns: pd.Series, title: str = "Returns Distribution", 
                                save_path: Optional[str] = None):
        """
        Plot distribution of returns
        
        Args:
            returns: Series of returns
            title: Chart title
            save_path: Path to save the figure, if None, display the figure
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram and kernel density estimate
        sns.histplot(returns, kde=True, stat="density", alpha=0.6)
        
        # Add a normal distribution for comparison
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        plt.plot(x, y, 'r--', linewidth=1.5, label='Normal Distribution')
        
        # Add vertical line at mean
        plt.axvline(mu, color='r', linestyle='-', alpha=0.3, label='Mean')
        
        # Set title and labels
        plt.title(title, fontsize=14)
        plt.xlabel('Returns', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Add legend and grid
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            full_path = self.figures_dir / save_path
            plt.savefig(full_path, dpi=300)
            logger.info(f"Returns distribution saved to {full_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Correlation Matrix", 
                              threshold: float = 0.0, save_path: Optional[str] = None):
        """
        Plot correlation matrix of features
        
        Args:
            data: DataFrame with features
            title: Chart title
            threshold: Threshold for coloring correlations
            save_path: Path to save the figure, if None, display the figure
        """
        # Calculate correlation matrix
        corr = data.corr()
        
        # Create mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        # Set title
        plt.title(title, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display figure
        if save_path:
            full_path = self.figures_dir / save_path
            plt.savefig(full_path, dpi=300)
            logger.info(f"Correlation matrix saved to {full_path}")
            plt.close()
        else:
            plt.show()