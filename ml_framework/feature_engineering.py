"""
Feature Engineering Module

Creates 63+ technical indicators from OHLCV data for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """
    Feature engineering class for creating technical indicators.
    
    Creates 63+ features including:
    - Price-based features (returns, ratios, shadows)
    - Trend indicators (EMA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, CCI, Williams %R, ROC)
    - Volatility indicators (ATR, Bollinger Bands, Historical Volatility)
    - Volume indicators (Volume ratios, OBV, VWAP)
    - Microstructure features (range, consecutive movements, pressure)
    - Time features (hour, minute, day of week, session)
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> df_features = engineer.create_all_features(df)
        >>> print(f"Created {len(engineer.get_feature_list())} features")
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names: List[str] = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicator features.
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
               and datetime column
            
        Returns:
            DataFrame with all original columns plus 63+ feature columns
            
        Example:
            >>> df = pd.read_csv('data.csv')
            >>> engineer = FeatureEngineer()
            >>> df_features = engineer.create_all_features(df)
        """
        df = df.copy()
        self.feature_names = []
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume', 'datetime']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create features
        df = self._add_price_features(df)
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_microstructure_features(df)
        df = self._add_time_features(df)
        
        # Replace infinite values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Created {len(self.feature_names)} features")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price features."""
        # Price changes
        df['returns'] = df['close'].pct_change()
        # Log returns with division by zero protection
        shifted_close = df['close'].shift(1)
        df['log_returns'] = np.where(
            (shifted_close > 0) & (df['close'] > 0),
            np.log(df['close'] / shifted_close),
            0
        )
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Candle shadow ratios
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        self.feature_names.extend([
            'returns', 'log_returns', 'high_low_ratio', 
            'close_open_ratio', 'upper_shadow', 'lower_shadow', 'body_ratio'
        ])
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators (EMA, MACD, ADX)."""
        # EMA (multiple periods)
        for period in [5, 10, 20, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_dist_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
            self.feature_names.extend([f'ema_{period}', f'ema_dist_{period}'])
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # ADX
        df = self._calculate_adx(df, period=14)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (RSI, Stochastic, CCI, Williams %R, ROC)."""
        # RSI (multiple periods)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
            self.feature_names.append(f'rsi_{period}')
        
        # Stochastic Oscillator
        for period in [14, 21]:
            df = self._calculate_stochastic(df, period)
        
        # CCI
        for period in [14, 20]:
            df[f'cci_{period}'] = self._calculate_cci(df, period)
            self.feature_names.append(f'cci_{period}')
        
        # Williams %R
        for period in [14, 21]:
            df[f'williams_r_{period}'] = self._calculate_williams_r(df, period)
            self.feature_names.append(f'williams_r_{period}')
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                   df['close'].shift(period) * 100)
            self.feature_names.append(f'roc_{period}')
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators (ATR, Bollinger Bands, Historical Vol)."""
        # ATR (multiple periods)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = self._calculate_atr(df, period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            self.feature_names.extend([f'atr_{period}', f'atr_ratio_{period}'])
        
        # Bollinger Bands
        for period in [20, 50]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = ma + 2 * std
            df[f'bb_lower_{period}'] = ma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            self.feature_names.extend([f'bb_width_{period}', f'bb_position_{period}'])
        
        # Historical Volatility
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252 * 390)
            self.feature_names.append(f'volatility_{period}')
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators (Volume ratios, OBV, VWAP)."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio_5'] = df['volume'] / df['volume'].rolling(window=5).mean()
        df['volume_ma_ratio_20'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        df['obv_divergence'] = df['obv'] - df['obv_ema']
        
        # VWAP deviation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
        
        self.feature_names.extend([
            'volume_change', 'volume_ma_ratio_5', 'volume_ma_ratio_20',
            'obv_divergence', 'vwap_dist'
        ])
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Price range
        df['range'] = df['high'] - df['low']
        df['range_ma_ratio'] = df['range'] / df['range'].rolling(window=20).mean()
        
        # Consecutive up/down movements
        df['up_count'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['down_count'] = (df['close'] < df['close'].shift(1)).astype(int)
        df['consecutive_ups'] = df['up_count'].groupby(
            (df['up_count'] != df['up_count'].shift()).cumsum()
        ).cumsum()
        df['consecutive_downs'] = df['down_count'].groupby(
            (df['down_count'] != df['down_count'].shift()).cumsum()
        ).cumsum()
        
        # Buy/sell pressure estimation
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        
        self.feature_names.extend([
            'range_ma_ratio', 'consecutive_ups', 'consecutive_downs',
            'buy_pressure', 'sell_pressure'
        ])
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Session features (open, mid, close)
        df['session_open'] = ((df['hour'] == 9) & (df['minute'] < 30)).astype(int)
        df['session_mid'] = ((df['hour'] >= 10) & (df['hour'] < 14)).astype(int)
        df['session_close'] = ((df['hour'] == 14) & (df['minute'] >= 30)).astype(int)
        
        self.feature_names.extend([
            'hour', 'minute', 'day_of_week', 
            'session_open', 'session_mid', 'session_close'
        ])
        return df
    
    # ============== Helper calculation methods ==============
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        self.feature_names.extend([f'stoch_k_{period}', f'stoch_d_{period}'])
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md + 1e-10)
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        # Calculate +DI and -DI
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self._calculate_atr(df, period)
        pos_di = 100 * pos_dm.rolling(window=period).mean() / (atr + 1e-10)
        neg_di = 100 * neg_dm.rolling(window=period).mean() / (atr + 1e-10)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        df['adx'] = dx.rolling(window=period).mean()
        df['di_diff'] = pos_di - neg_di
        
        self.feature_names.extend(['adx', 'di_diff'])
        return df
    
    def get_feature_list(self) -> List[str]:
        """
        Get list of all feature names.
        
        Returns:
            List of feature column names
        """
        return self.feature_names.copy()
    
    def prepare_ml_features(self, df: pd.DataFrame, 
                           drop_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare features for ML model (drop non-feature columns).
        
        Args:
            df: DataFrame with all features
            drop_columns: Additional columns to drop (default: datetime, OHLCV)
            
        Returns:
            DataFrame with only ML features
        """
        if drop_columns is None:
            drop_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        available_drops = [col for col in drop_columns if col in df.columns]
        return df.drop(columns=available_drops)


if __name__ == '__main__':
    # Test feature engineering
    print("Testing Feature Engineering")
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
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    print(f"Sample data: {len(df)} rows\n")
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"\nFeature names ({len(engineer.get_feature_list())} total):")
    for i, feat in enumerate(engineer.get_feature_list(), 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\n✅ Feature engineering test completed!")
