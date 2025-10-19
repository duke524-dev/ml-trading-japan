"""
Walk-Forward Optimization Module

Parameter optimization using time-series cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
import itertools
import json
from pathlib import Path

from .backtester import Backtester
from .ml_model import MLSignalPredictor


class WalkForwardOptimizer:
    """
    Walk-Forward optimization for trading strategies.
    
    Features:
    - Time-series cross-validation (no lookahead bias)
    - Grid search over parameter space
    - Multi-fold validation
    - Results aggregation
    
    Example:
        >>> optimizer = WalkForwardOptimizer(n_splits=5)
        >>> best_params = optimizer.optimize(df, param_grid)
        >>> print(f"Best ROI: {best_params['avg_roi']:.2f}%")
    """
    
    def __init__(self, n_splits: int = 5, min_trades: int = 10,
                 initial_capital: float = 5_000_000,
                 tick_size: float = 0.5, multiplier: int = 10000):
        """
        Initialize optimizer.
        
        Args:
            n_splits: Number of time-series splits
            min_trades: Minimum trades required for valid result
            initial_capital: Starting capital
            tick_size: Minimum price movement
            multiplier: Contract multiplier
        """
        self.n_splits = n_splits
        self.min_trades = min_trades
        self.initial_capital = initial_capital
        self.tick_size = tick_size
        self.multiplier = multiplier
        self.results: List[Dict] = []
        
    def optimize(self, df: pd.DataFrame, param_grid: Dict,
                 feature_cols: List[str] = None,
                 verbose: bool = True) -> Dict:
        """
        Run walk-forward optimization.
        
        Args:
            df: DataFrame with OHLCV and features
            param_grid: Parameter grid to search
                Example: {
                    'profit_threshold': [0.05, 0.08, 0.10],
                    'lookahead_bars': [15, 20, 30],
                    'ml_confidence': [0.15, 0.20, 0.25],
                    'tp_multiplier': [2.0, 3.0, 4.0],
                    'sl_multiplier': [0.8, 1.0, 1.2],
                    'trail_multiplier': [1.0, 1.2, 1.5]
                }
            feature_cols: Feature columns (auto-detect if None)
            verbose: Print progress
            
        Returns:
            Best parameters with average performance
        """
        # Auto-detect feature columns
        if feature_cols is None:
            exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('label')]
        
        # Generate parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Walk-Forward Optimization")
            print(f"{'='*70}")
            print(f"Parameter combinations: {len(param_combinations)}")
            print(f"Time-series splits: {self.n_splits}")
            print(f"Total experiments: {len(param_combinations) * self.n_splits}")
            print(f"{'='*70}\n")
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Store all results
        all_results = []
        
        # Grid search
        for param_idx, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            
            if verbose and param_idx % 10 == 0:
                print(f"Testing combination {param_idx+1}/{len(param_combinations)}...")
            
            fold_results = []
            
            # Walk-forward validation
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df)):
                train_df = df.iloc[train_idx].copy()
                test_df = df.iloc[test_idx].copy()
                
                # Generate labels
                train_df = self._generate_labels(
                    train_df,
                    params.get('profit_threshold', 0.08),
                    params.get('lookahead_bars', 20)
                )
                
                # Train ML model
                model = MLSignalPredictor()
                X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                y_train = train_df['label'].values
                
                try:
                    model.train(X_train, y_train)
                except:
                    continue
                
                # Generate signals on test set
                X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                signals = model.predict(X_test, confidence_threshold=params.get('ml_confidence', 0.20))
                
                # Backtest
                backtester = Backtester(
                    initial_capital=self.initial_capital,
                    tick_size=self.tick_size,
                    multiplier=self.multiplier
                )
                
                results = backtester.run(
                    test_df,
                    signals,
                    tp_multiplier=params.get('tp_multiplier', 3.0),
                    sl_multiplier=params.get('sl_multiplier', 1.0),
                    trail_multiplier=params.get('trail_multiplier', 1.2)
                )
                
                # Validate results
                if results['total_trades'] >= self.min_trades:
                    fold_results.append(results['roi'])
            
            # Aggregate fold results
            if len(fold_results) >= 3:  # At least 3 valid folds
                avg_roi = np.mean(fold_results)
                std_roi = np.std(fold_results)
                min_roi = np.min(fold_results)
                max_roi = np.max(fold_results)
                
                all_results.append({
                    'params': params,
                    'avg_roi': avg_roi,
                    'std_roi': std_roi,
                    'min_roi': min_roi,
                    'max_roi': max_roi,
                    'valid_folds': len(fold_results),
                    'fold_results': fold_results
                })
        
        if not all_results:
            raise ValueError("No valid parameter combinations found!")
        
        # Sort by average ROI
        all_results.sort(key=lambda x: x['avg_roi'], reverse=True)
        self.results = all_results
        
        best = all_results[0]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Optimization Complete!")
            print(f"{'='*70}")
            print(f"Best Parameters:")
            for key, value in best['params'].items():
                print(f"  {key}: {value}")
            print(f"\nPerformance:")
            print(f"  Average ROI: {best['avg_roi']:.2f}%")
            print(f"  Std Dev:     {best['std_roi']:.2f}%")
            print(f"  Min ROI:     {best['min_roi']:.2f}%")
            print(f"  Max ROI:     {best['max_roi']:.2f}%")
            print(f"  Valid Folds: {best['valid_folds']}/{self.n_splits}")
            print(f"{'='*70}\n")
        
        return best
    
    def _generate_labels(self, df: pd.DataFrame,
                        profit_threshold: float,
                        lookahead_bars: int) -> pd.DataFrame:
        """
        Generate trading labels based on future price movement.
        
        Args:
            df: DataFrame with OHLCV
            profit_threshold: Profit threshold in decimal (e.g., 0.08 = 8%)
            lookahead_bars: Bars to look ahead
            
        Returns:
            DataFrame with 'label' column (1=Long, 0=Neutral, -1=Short)
        """
        df = df.copy()
        
        # Calculate future returns
        future_high = df['high'].rolling(window=lookahead_bars, min_periods=1).max().shift(-lookahead_bars)
        future_low = df['low'].rolling(window=lookahead_bars, min_periods=1).min().shift(-lookahead_bars)
        
        upside = (future_high - df['close']) / df['close']
        downside = (df['close'] - future_low) / df['close']
        
        # Label logic
        labels = np.zeros(len(df), dtype=int)
        labels[upside >= profit_threshold] = 1  # Long
        labels[downside >= profit_threshold] = -1  # Short
        
        df['label'] = labels
        
        return df
    
    def get_top_n(self, n: int = 10) -> List[Dict]:
        """
        Get top N parameter sets.
        
        Args:
            n: Number of top results
            
        Returns:
            List of top parameter dictionaries
        """
        return self.results[:n]
    
    def save_results(self, filepath: str):
        """
        Save optimization results to JSON.
        
        Args:
            filepath: Output JSON file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {filepath}")
    
    def load_results(self, filepath: str) -> List[Dict]:
        """
        Load optimization results from JSON.
        
        Args:
            filepath: Input JSON file path
            
        Returns:
            List of result dictionaries
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        return self.results
    
    def plot_results(self, top_n: int = 20, save_path: str = None):
        """
        Plot optimization results.
        
        Args:
            top_n: Number of top results to plot
            save_path: Save plot to file (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                print("⚠️  No results to plot!")
                return
            
            top_results = self.results[:top_n]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Walk-Forward Optimization Results', fontsize=16, fontweight='bold')
            
            # 1. Top N ROI comparison
            ax = axes[0, 0]
            rois = [r['avg_roi'] for r in top_results]
            x = range(1, len(rois) + 1)
            ax.bar(x, rois, color='steelblue', alpha=0.7)
            ax.set_xlabel('Parameter Set Rank')
            ax.set_ylabel('Average ROI (%)')
            ax.set_title(f'Top {top_n} Parameter Sets')
            ax.grid(True, alpha=0.3)
            
            # 2. ROI distribution
            ax = axes[0, 1]
            all_rois = [r['avg_roi'] for r in self.results]
            ax.hist(all_rois, bins=30, color='coral', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_rois), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_rois):.2f}%')
            ax.set_xlabel('Average ROI (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('ROI Distribution Across All Combinations')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Stability (Avg vs Std)
            ax = axes[1, 0]
            avg_rois = [r['avg_roi'] for r in top_results]
            std_rois = [r['std_roi'] for r in top_results]
            scatter = ax.scatter(std_rois, avg_rois, c=avg_rois, cmap='RdYlGn',
                               s=100, alpha=0.6, edgecolors='black')
            ax.set_xlabel('ROI Standard Deviation (%)')
            ax.set_ylabel('Average ROI (%)')
            ax.set_title('Stability Analysis (Lower Std = More Stable)')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Avg ROI (%)')
            
            # 4. Fold consistency
            ax = axes[1, 1]
            best = self.results[0]
            fold_rois = best['fold_results']
            ax.plot(range(1, len(fold_rois) + 1), fold_rois, 
                   marker='o', linewidth=2, markersize=8, color='green')
            ax.axhline(best['avg_roi'], color='blue', linestyle='--', 
                      label=f'Average: {best["avg_roi"]:.2f}%')
            ax.set_xlabel('Fold Number')
            ax.set_ylabel('ROI (%)')
            ax.set_title('Best Parameters - Fold-by-Fold Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved to: {save_path}")
            else:
                plt.show()
        
        except ImportError:
            print("⚠️  matplotlib/seaborn not installed. Cannot plot results.")


if __name__ == '__main__':
    print("Testing WalkForwardOptimizer")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01 09:00', periods=5000, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': 38000 + np.cumsum(np.random.randn(5000) * 10),
        'high': 38000 + np.cumsum(np.random.randn(5000) * 10) + 50,
        'low': 38000 + np.cumsum(np.random.randn(5000) * 10) - 50,
        'close': 38000 + np.cumsum(np.random.randn(5000) * 10),
        'volume': np.random.randint(1000, 10000, 5000),
        'atr_14': np.random.uniform(30, 60, 5000),
        'rsi_14': np.random.uniform(30, 70, 5000),
        'ema_fast': 38000 + np.random.randn(5000) * 100,
        'ema_slow': 38000 + np.random.randn(5000) * 100
    })
    
    # Small parameter grid for testing
    param_grid = {
        'profit_threshold': [0.05, 0.08],
        'lookahead_bars': [15, 20],
        'ml_confidence': [0.15, 0.20],
        'tp_multiplier': [2.0, 3.0],
        'sl_multiplier': [1.0],
        'trail_multiplier': [1.2]
    }
    
    # Run optimization
    optimizer = WalkForwardOptimizer(n_splits=3, min_trades=5)
    best = optimizer.optimize(df, param_grid, verbose=True)
    
    print(f"\n✅ Optimizer test completed!")
    print(f"   Best avg ROI: {best['avg_roi']:.2f}%")
