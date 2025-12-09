"""
ML Model Module - XGBoost 3-Class Classifier

Predicts Long (1), Neutral (0), or Short (-1) trading signals.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MLSignalPredictor:
    """
    XGBoost-based 3-class trading signal predictor.
    
    Predicts Long (1), Neutral (0), or Short (-1) signals based on
    technical indicators. Uses probability thresholds for confidence filtering.
    
    Example:
        >>> predictor = MLSignalPredictor(n_estimators=200)
        >>> predictor.train(X_train, y_train)
        >>> signals = predictor.predict(X_test, confidence_threshold=0.25)
    """
    
    def __init__(self, lookahead_bars: int = 5, n_estimators: int = 200,
                 max_depth: int = 6, learning_rate: float = 0.05):
        """
        Initialize ML predictor.
        
        Args:
            lookahead_bars: Number of bars to look ahead for labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        self.lookahead_bars = lookahead_bars
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_cols = None
        self.feature_importance_df = None
        
    def prepare_labels(self, df: pd.DataFrame, profit_threshold: float = 0.08) -> pd.DataFrame:
        """
        Prepare training labels from price data.
        
        Creates 3-class labels:
        - 1 (Long): Future price expected to rise significantly
        - 0 (Neutral): Future price expected to stay flat
        - -1 (Short): Future price expected to fall significantly
        
        Args:
            df: DataFrame with OHLC data
            profit_threshold: Minimum profit percentage to trigger signal (e.g., 0.08 = 8%)
            
        Returns:
            DataFrame with 'signal' column added
        """
        df = df.copy()
        
        # Calculate future returns
        df['future_return'] = df['close'].shift(-self.lookahead_bars) / df['close'] - 1
        df['future_return_pct'] = df['future_return'] * 100
        
        # Calculate future high/low within lookahead window
        df['future_high'] = df['high'].rolling(window=self.lookahead_bars).max().shift(-self.lookahead_bars)
        df['future_low'] = df['low'].rolling(window=self.lookahead_bars).min().shift(-self.lookahead_bars)
        
        # Calculate potential upside/downside
        df['potential_upside'] = (df['future_high'] / df['close'] - 1) * 100
        df['potential_downside'] = (df['future_low'] / df['close'] - 1) * 100
        
        # Generate 3-class labels
        conditions = [
            (df['potential_upside'] >= profit_threshold) & (df['potential_downside'] > -profit_threshold/2),
            (df['potential_downside'] <= -profit_threshold) & (df['potential_upside'] < profit_threshold/2),
        ]
        choices = [1, -1]  # Long, Short
        df['signal'] = np.select(conditions, choices, default=0)  # Neutral
        
        # Remove rows with future NaN
        df = df.dropna(subset=['future_return'])
        
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> dict:
        """
        Train XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (signal: 1=Long, 0=Neutral, -1=Short)
            verbose: Print training progress
            
        Returns:
            Training results dictionary
        """
        # Convert signals to 0, 1, 2 for XGBoost
        # Map: -1 (Short) -> 0, 0 (Neutral) -> 1, 1 (Long) -> 2
        if isinstance(y, pd.Series):
            y_train = y.map({-1: 0, 0: 1, 1: 2}).values
        else:
            y_train = np.where(y == -1, 0, np.where(y == 0, 1, 2))
        
        # Store feature columns
        if isinstance(X, pd.DataFrame):
            self.feature_cols = X.columns.tolist()
            X_train = X.values
        else:
            X_train = X
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        if self.feature_cols:
            self.feature_importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate training accuracy
        train_score = self.model.score(X_train, y_train)
        
        if verbose:
            print(f"✅ Model trained successfully")
            print(f"   Training accuracy: {train_score:.4f}")
            print(f"   Features: {len(self.feature_cols) if self.feature_cols else X.shape[1]}")
        
        return {
            'train_score': train_score,
            'n_features': len(self.feature_cols) if self.feature_cols else X.shape[1]
        }
    
    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.20) -> np.ndarray:
        """
        Predict trading signals with confidence threshold.
        
        Args:
            X: Feature DataFrame
            confidence_threshold: Minimum probability to trigger signal (0-1)
            
        Returns:
            Array of signals (1=Long, 0=Neutral, -1=Short)
        """
        if isinstance(X, pd.DataFrame):
            X_pred = X[self.feature_cols].values if self.feature_cols else X.values
        else:
            X_pred = X
        
        # Get probabilities
        probas = self.model.predict_proba(X_pred)
        
        # Get predicted classes and their probabilities
        pred_classes = self.model.predict(X_pred)
        max_probas = probas.max(axis=1)
        
        # Convert to -1, 0, 1
        signals = pred_classes.copy()
        signals[pred_classes == 0] = -1  # Short
        signals[pred_classes == 1] = 0   # Neutral
        signals[pred_classes == 2] = 1   # Long
        
        # Apply confidence threshold
        signals[max_probas < confidence_threshold] = 0  # Set to Neutral if low confidence
        
        return signals
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities shape (n_samples, 3)
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_cols:
                missing_cols = set(self.feature_cols) - set(X.columns)
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")
                X_pred = X[self.feature_cols].values
            else:
                X_pred = X.values
        else:
            X_pred = X
        
        return self.model.predict_proba(X_pred)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and importance scores
        """
        if self.feature_importance_df is None:
            raise ValueError("Model not trained yet")
        
        return self.feature_importance_df.head(top_n)


if __name__ == '__main__':
    # Test ML model
    print("Testing ML Signal Predictor")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 63
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.choice([-1, 0, 1], n_samples))
    
    # Train model
    predictor = MLSignalPredictor(lookahead_bars=5)
    results = predictor.train(X_train, y_train)
    
    # Predict
    X_test = X_train.iloc[:10]
    signals = predictor.predict(X_test, confidence_threshold=0.25)
    
    print(f"\nTest predictions:")
    print(f"Signals: {signals}")
    print(f"\n✅ ML model test completed!")
