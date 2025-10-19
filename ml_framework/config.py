"""
Configuration Management for ML Trading Framework

Provides market-specific configurations for different Japanese stock indices.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MarketConfig:
    """
    Market-specific configuration for trading instruments.
    
    Attributes:
        name: Market name (e.g., 'nikkei225', 'topix')
        tick_size: Minimum price movement
        multiplier: Contract multiplier for P&L calculation
        initial_capital: Starting capital for backtesting
        trading_hours_start: Market opening hour (24-hour format)
        trading_hours_end: Market closing hour (24-hour format)
    """
    name: str
    tick_size: float
    multiplier: int
    initial_capital: int = 5_000_000
    trading_hours_start: int = 9
    trading_hours_end: int = 15
    
    def calculate_pnl(self, entry_price: float, exit_price: float, 
                     position: int = 1) -> float:
        """
        Calculate P&L for a trade.
        
        Args:
            entry_price: Entry price level
            exit_price: Exit price level
            position: 1 for long, -1 for short
            
        Returns:
            Profit/loss in currency units
        """
        price_diff = (exit_price - entry_price) * position
        return price_diff * self.multiplier


@dataclass
class TradingConfig:
    """
    Trading strategy configuration parameters.
    
    Attributes:
        profit_threshold: Minimum predicted profit to trigger trade (e.g., 0.08 = 8%)
        lookahead_bars: Number of bars to look ahead for target calculation
        ml_confidence: Minimum ML model confidence threshold (0.0-1.0)
        tp_multiplier: Take-profit distance in ATR units
        sl_multiplier: Stop-loss distance in ATR units
        trail_multiplier: Trailing stop distance in ATR units
        max_holding_bars: Maximum bars to hold a position (0 = unlimited)
        risk_per_trade: Maximum % of capital to risk per trade
    """
    profit_threshold: float = 0.08
    lookahead_bars: int = 5
    ml_confidence: float = 0.20
    tp_multiplier: float = 3.0
    sl_multiplier: float = 1.0
    trail_multiplier: float = 1.2
    max_holding_bars: int = 0
    risk_per_trade: float = 0.02
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not 0.0 < self.profit_threshold < 1.0:
            raise ValueError(f"profit_threshold must be between 0 and 1, got {self.profit_threshold}")
        
        if self.lookahead_bars < 1:
            raise ValueError(f"lookahead_bars must be >= 1, got {self.lookahead_bars}")
        
        if not 0.0 <= self.ml_confidence <= 1.0:
            raise ValueError(f"ml_confidence must be between 0 and 1, got {self.ml_confidence}")
        
        if self.tp_multiplier <= 0:
            raise ValueError(f"tp_multiplier must be > 0, got {self.tp_multiplier}")
        
        if self.sl_multiplier <= 0:
            raise ValueError(f"sl_multiplier must be > 0, got {self.sl_multiplier}")
        
        if self.trail_multiplier < 0:
            raise ValueError(f"trail_multiplier must be >= 0, got {self.trail_multiplier}")
        
        return True


# Predefined market configurations
DEFAULT_CONFIGS: Dict[str, MarketConfig] = {
    'nikkei225': MarketConfig(
        name='Nikkei225 Mini',
        tick_size=10.0,
        multiplier=1000,
        initial_capital=5_000_000
    ),
    'topix': MarketConfig(
        name='TOPIX Future',
        tick_size=0.5,
        multiplier=10000,
        initial_capital=5_000_000
    ),
}


def get_market_config(market_name: str) -> MarketConfig:
    """
    Get predefined market configuration.
    
    Args:
        market_name: Market identifier ('nikkei225' or 'topix')
        
    Returns:
        MarketConfig object
        
    Raises:
        KeyError: If market_name not found
        
    Example:
        >>> config = get_market_config('topix')
        >>> print(config.tick_size)
        0.5
    """
    if market_name.lower() not in DEFAULT_CONFIGS:
        raise KeyError(
            f"Unknown market '{market_name}'. "
            f"Available markets: {list(DEFAULT_CONFIGS.keys())}"
        )
    
    return DEFAULT_CONFIGS[market_name.lower()]


def create_custom_config(name: str, tick_size: float, multiplier: int,
                        initial_capital: int = 5_000_000) -> MarketConfig:
    """
    Create a custom market configuration.
    
    Args:
        name: Market name
        tick_size: Minimum price movement
        multiplier: Contract multiplier
        initial_capital: Starting capital
        
    Returns:
        MarketConfig object
        
    Example:
        >>> config = create_custom_config('my_market', tick_size=1.0, multiplier=100)
        >>> print(config.name)
        'my_market'
    """
    return MarketConfig(
        name=name,
        tick_size=tick_size,
        multiplier=multiplier,
        initial_capital=initial_capital
    )


if __name__ == '__main__':
    # Test configurations
    print("Testing Market Configurations")
    print("=" * 50)
    
    # Test predefined configs
    for market_name in DEFAULT_CONFIGS:
        config = get_market_config(market_name)
        print(f"\n{config.name}:")
        print(f"  Tick Size: {config.tick_size}")
        print(f"  Multiplier: {config.multiplier}")
        print(f"  Initial Capital: ¥{config.initial_capital:,}")
        
        # Test P&L calculation
        pnl = config.calculate_pnl(entry_price=38000, exit_price=38100, position=1)
        print(f"  Example P&L (38000→38100): ¥{pnl:,}")
    
    # Test trading config validation
    print("\n\nTesting Trading Configuration")
    print("=" * 50)
    
    trading_config = TradingConfig(
        profit_threshold=0.08,
        ml_confidence=0.20,
        tp_multiplier=3.0
    )
    
    print(f"Trading Config:")
    print(f"  Profit Threshold: {trading_config.profit_threshold}")
    print(f"  ML Confidence: {trading_config.ml_confidence}")
    print(f"  TP Multiplier: {trading_config.tp_multiplier}")
    print(f"  Validation: {'✅ PASSED' if trading_config.validate() else '❌ FAILED'}")
