"""
test_ml_edge_engine.py - Tests for ML edge detection
"""

import pytest
from ai.ml_edge_engine import (
    MLEdgeEngine,
    MLModelType,
    MLSignal,
    initialize_ml_engine,
    get_ml_engine,
)


def test_ml_engine_disabled_by_default():
    """Test that ML engine is disabled by default"""
    engine = MLEdgeEngine(model_type=MLModelType.DISABLED)
    assert engine.model_type == MLModelType.DISABLED
    
    # Should not generate signals when disabled
    signal = engine.predict(None)
    assert signal is None


def test_ml_engine_requires_approval():
    """Test that ML predictions require human approval"""
    engine = MLEdgeEngine(
        model_type=MLModelType.LSTM_RNN,
        require_human_approval=True
    )
    
    # Without approval, should not generate signals
    assert not engine.is_approved
    signal = engine.predict(None)
    assert signal is None


def test_ml_training_result_validation():
    """Test that training requires minimum Sharpe ratio"""
    import pandas as pd
    
    engine = MLEdgeEngine(
        model_type=MLModelType.GRADIENT_BOOSTING,
        min_sharpe_for_deployment=1.5
    )
    
    # Create dummy features
    X_train = pd.DataFrame({'feature': [1, 2, 3]})
    y_train = pd.Series([0, 1, 0])
    
    # Train model
    result = engine.train_model(X_train, y_train)
    
    # Should indicate model is not ready
    assert result.out_of_sample_sharpe < engine.min_sharpe
    assert not result.is_profitable


def test_ml_approval_requires_sharpe():
    """Test that approval is blocked if Sharpe is too low"""
    engine = MLEdgeEngine(
        model_type=MLModelType.ENSEMBLE,
        min_sharpe_for_deployment=1.5
    )
    
    # Try to request approval without training
    approved = engine.request_approval_for_deployment("Model looks good")
    assert not approved  # No training results yet
    
    # Add a training result with low Sharpe
    import pandas as pd
    X = pd.DataFrame({'f': [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    engine.train_model(X, y)
    
    # Try to approve again - should still fail (low Sharpe)
    approved = engine.request_approval_for_deployment("Model looks good")
    assert not approved  # Sharpe too low


def test_ml_signal_structure():
    """Test MLSignal dataclass"""
    signal = MLSignal(
        direction='buy',
        confidence=0.75,
        next_move_pct=1.5,
        probability=0.72,
        metadata={'model': 'lstm', 'lookback': 50}
    )
    
    assert signal.direction == 'buy'
    assert signal.confidence == 0.75
    assert signal.next_move_pct == 1.5
    assert signal.probability == 0.72
    assert signal.metadata['model'] == 'lstm'
    assert signal.timestamp is not None


def test_ml_engine_status():
    """Test status reporting"""
    engine = MLEdgeEngine(
        model_type=MLModelType.RANDOM_FOREST,
        min_sharpe_for_deployment=1.5
    )
    
    status = engine.get_status()
    assert status['model_type'] == 'random_forest'
    assert status['is_enabled']
    assert not status['is_trained']
    assert not status['is_approved']
    assert status['min_sharpe_required'] == 1.5


def test_ml_engine_singleton():
    """Test ML engine can be initialized as singleton"""
    import ai.ml_edge_engine as ml_module
    
    # Reset the global instance
    ml_module._ml_engine = None
    
    # Initialize
    engine1 = initialize_ml_engine(MLModelType.GRADIENT_BOOSTING)
    assert engine1 is not None
    
    # Get should return same instance
    engine2 = get_ml_engine()
    assert engine1 is engine2


def test_ml_live_performance_tracking():
    """Test tracking of live vs backtest performance"""
    engine = MLEdgeEngine(model_type=MLModelType.LSTM_RNN)
    
    # Train model (backtest)
    import pandas as pd
    X = pd.DataFrame({'f': [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    result = engine.train_model(X, y)
    
    # Record live performance
    engine.record_live_performance(
        sharpe_ratio=0.4,  # 50% of backtest (realistic)
        win_rate=0.48,
        notes="Live validation period"
    )
    
    assert engine.live_performance_sharpe == 0.4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
