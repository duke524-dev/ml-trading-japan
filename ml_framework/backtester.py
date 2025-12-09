"""
Backtesting Engine Module

Vectorized backtesting engine with ATR-based stops.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Trade:
    """Trade record data class."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position: int  # 1=Long, -1=Short
    pnl: float
    pnl_pct: float
    exit_reason: str
    bars_held: int


class Backtester:
    """
    Vectorized backtesting engine for trading strategies.
    
    Features:
    - ATR-based dynamic stops (TP, SL, Trailing)
    - Position sizing
    - Trade logging
    - Performance metrics
    
    Example:
        >>> backtester = Backtester(initial_capital=5_000_000)
        >>> results = backtester.run(df, signals, config)
        >>> print(f"ROI: {results['roi']:.2f}%")
    """
    
    def __init__(self, initial_capital: float = 5_000_000,
                 tick_size: float = 0.5, multiplier: int = 10000):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            tick_size: Minimum price movement
            multiplier: Contract multiplier for P&L
        """
        self.initial_capital = initial_capital
        self.tick_size = tick_size
        self.multiplier = multiplier
        self.trades: List[Trade] = []
        
    def run(self, df: pd.DataFrame, signals: np.ndarray,
            tp_multiplier: float = 3.0, sl_multiplier: float = 1.0,
            trail_multiplier: float = 1.2, atr_column: str = 'atr_14',
            position_size: float = 1.0) -> Dict:
        """
        Run backtest on data with signals.
        
        Args:
            df: DataFrame with OHLCV and ATR columns
            signals: Array of signals (1=Long, 0=Neutral, -1=Short)
            tp_multiplier: Take-profit in ATR units
            sl_multiplier: Stop-loss in ATR units
            trail_multiplier: Trailing stop in ATR units
            atr_column: ATR column name in df
            position_size: Position size multiplier (1.0 = full size)
            
        Returns:
            Dictionary with performance metrics
        """
        self.trades = []
        capital = self.initial_capital
        position = None
        
        # Validate signals length
        if len(signals) != len(df):
            raise ValueError(
                f"Signals length ({len(signals)}) must match DataFrame length ({len(df)})"
            )
        
        for i in range(len(df)):
            current_bar = df.iloc[i]
            current_signal = int(signals[i])  # Ensure integer type
            
            # Entry logic
            if position is None and current_signal != 0:
                # Get ATR value with proper error handling
                try:
                    atr = current_bar[atr_column]
                    if pd.isna(atr) or atr <= 0:
                        atr = 50.0
                except (KeyError, IndexError):
                    atr = 50.0
                
                position = {
                    'entry_idx': i,
                    'entry_time': current_bar['datetime'],
                    'entry_price': current_bar['close'],
                    'direction': int(current_signal),  # 1 or -1
                    'atr': atr,
                    'position_size': position_size,
                    'tp': current_bar['close'] + (current_signal * tp_multiplier * atr),
                    'sl': current_bar['close'] - (current_signal * sl_multiplier * atr),
                    'trail': current_bar['close'] + (current_signal * trail_multiplier * atr),
                    'highest_price': current_bar['close'] if current_signal == 1 else None,
                    'lowest_price': current_bar['close'] if current_signal == -1 else None
                }
                continue
            
            # Exit logic
            if position is not None:
                current_price = current_bar['close']
                exit_reason = None
                
                # Update trailing stop
                if position['direction'] == 1:  # Long
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                        position['trail'] = current_price - trail_multiplier * position['atr']
                    
                    # Check exit conditions
                    if current_price >= position['tp']:
                        exit_reason = 'TP'
                    elif current_price <= position['sl']:
                        exit_reason = 'SL'
                    elif current_price <= position['trail']:
                        exit_reason = 'Trail'
                    elif current_signal == -1:
                        exit_reason = 'Signal'
                    elif i == len(df) - 1:
                        exit_reason = 'EOD'
                
                else:  # Short
                    if current_price < position['lowest_price']:
                        position['lowest_price'] = current_price
                        position['trail'] = current_price + trail_multiplier * position['atr']
                    
                    # Check exit conditions
                    if current_price <= position['tp']:
                        exit_reason = 'TP'
                    elif current_price >= position['sl']:
                        exit_reason = 'SL'
                    elif current_price >= position['trail']:
                        exit_reason = 'Trail'
                    elif current_signal == 1:
                        exit_reason = 'Signal'
                    elif i == len(df) - 1:
                        exit_reason = 'EOD'
                
                if exit_reason:
                    # Calculate P&L
                    price_diff = (current_price - position['entry_price']) * position['direction']
                    pnl = price_diff * self.multiplier * position['position_size']
                    pnl_pct = (price_diff / position['entry_price']) * 100
                    
                    # Record trade
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=current_bar['datetime'],
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        position=position['direction'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        bars_held=i - position['entry_idx']
                    )
                    self.trades.append(trade)
                    
                    # Update capital
                    capital += pnl
                    
                    # Close position
                    position = None
        
        # Calculate metrics
        return self.calculate_metrics(capital)
    
    def calculate_metrics(self, final_capital: float) -> Dict:
        """
        Calculate performance metrics from trades.
        
        Args:
            final_capital: Final capital after all trades
            
        Returns:
            Dictionary with performance metrics
        """
        if len(self.trades) == 0:
            return {
                'roi': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'final_capital': self.initial_capital
            }
        
        # Basic metrics
        total_pnl = final_capital - self.initial_capital
        roi = (total_pnl / self.initial_capital) * 100
        
        # Win/Loss analysis
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        # Profit factor
        total_win = sum(t.pnl for t in wins) if wins else 0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = total_win / total_loss if total_loss > 0 else 0
        
        # Average win/loss
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Max drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        
        return {
            'roi': roi,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'final_capital': final_capital,
            'sharpe_ratio': self._calculate_sharpe(),
            'avg_bars_held': np.mean([t.bars_held for t in self.trades])
        }
    
    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.pnl_pct for t in self.trades]
        excess_returns = np.mean(returns) - (risk_free_rate / 252)  # Daily risk-free rate
        std_returns = np.std(returns)
        
        return (excess_returns / std_returns) * np.sqrt(252) if std_returns > 0 else 0.0
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.
        
        Returns:
            DataFrame with all trade records
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position': 'Long' if t.position == 1 else 'Short',
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'bars_held': t.bars_held
            }
            for t in self.trades
        ])
    
    def print_summary(self, results: Dict):
        """Print backtest summary."""
        print("\n" + "="*70)
        print("Backtest Results Summary")
        print("="*70)
        print(f"Initial Capital:     ¥{self.initial_capital:,.0f}")
        print(f"Final Capital:       ¥{results['final_capital']:,.0f}")
        print(f"Total P&L:           ¥{results['total_pnl']:,.0f}")
        print(f"ROI:                 {results['roi']:.2f}%")
        print(f"\nTotal Trades:        {results['total_trades']}")
        print(f"Winning Trades:      {results['winning_trades']}")
        print(f"Losing Trades:       {results['losing_trades']}")
        print(f"Win Rate:            {results['win_rate']:.2f}%")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"\nAvg Win:             ¥{results['avg_win']:,.0f}")
        print(f"Avg Loss:            ¥{results['avg_loss']:,.0f}")
        print(f"Max Drawdown:        ¥{results['max_drawdown']:,.0f} ({results['max_drawdown_pct']:.2f}%)")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Avg Bars Held:       {results['avg_bars_held']:.1f}")
        print("="*70)


if __name__ == '__main__':
    # Test backtester
    print("Testing Backtester")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01 09:00', periods=1000, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': 38000 + np.cumsum(np.random.randn(1000) * 10),
        'high': 38000 + np.cumsum(np.random.randn(1000) * 10) + 50,
        'low': 38000 + np.cumsum(np.random.randn(1000) * 10) - 50,
        'close': 38000 + np.cumsum(np.random.randn(1000) * 10),
        'volume': np.random.randint(1000, 10000, 1000),
        'atr_14': np.random.uniform(30, 60, 1000)
    })
    
    # Generate random signals
    signals = np.random.choice([-1, 0, 1], 1000, p=[0.1, 0.8, 0.1])
    
    # Run backtest
    backtester = Backtester(initial_capital=5_000_000, tick_size=0.5, multiplier=10000)
    results = backtester.run(df, signals, tp_multiplier=3.0, sl_multiplier=1.0)
    
    # Print results
    backtester.print_summary(results)
    
    print(f"\n✅ Backtester test completed!")
    print(f"   Trades executed: {results['total_trades']}")
