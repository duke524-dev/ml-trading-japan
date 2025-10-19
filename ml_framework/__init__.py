"""
ML Trading Japan - Core Framework Package

A machine learning trading framework achieving 72.56% ROI on Japanese stock indices.
"""

__version__ = "1.0.0"
__author__ = "ML Trading Japan Contributors"

from .feature_engineering import FeatureEngineer
from .config import MarketConfig, TradingConfig, get_market_config, create_custom_config
from .ml_model import MLSignalPredictor
from .backtester import Backtester
from .optimizer import WalkForwardOptimizer

__all__ = [
    "FeatureEngineer",
    "MarketConfig",
    "TradingConfig",
    "get_market_config",
    "create_custom_config",
    "MLSignalPredictor",
    "Backtester",
    "WalkForwardOptimizer",
]
