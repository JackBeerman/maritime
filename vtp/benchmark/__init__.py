"""
A portable, model-agnostic benchmark for vessel trajectory prediction.

    from vtp.benchmark import BenchmarkSet, Predictor, default_baselines, compare_predictors

    bench = BenchmarkSet.load('benchmarks/danish_60min.npz')
    df, results, per_window = compare_predictors(default_baselines(), bench)

To enter a model, implement one method:

    class MyModel(Predictor):
        name = "my-model"
        probabilistic = True
        def predict(self, window, n_samples=32):
            return ...   # (n_samples, len(window.target_times), 2) lon/lat
"""
from .protocol import Window, BenchmarkSet, haversine_km
from .predictors import (
    Predictor, PersistencePredictor, ConstantVelocityPredictor,
    CTRVPredictor, GaussianNoiseCVPredictor, default_baselines, BASELINES,
)
from .metrics import (
    window_metrics, energy_score, evaluate_predictor, compare_predictors,
)

__all__ = [
    'Window', 'BenchmarkSet', 'haversine_km',
    'Predictor', 'PersistencePredictor', 'ConstantVelocityPredictor',
    'CTRVPredictor', 'GaussianNoiseCVPredictor', 'default_baselines', 'BASELINES',
    'window_metrics', 'energy_score', 'evaluate_predictor', 'compare_predictors',
]
