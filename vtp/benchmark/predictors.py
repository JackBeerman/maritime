"""
The predictor interface, and the reference baselines every result should
be reported against.

Implementing `Predictor` is the only thing an external model needs to do
to appear in the comparison table:

    class MyModel(Predictor):
        name = "my-model"
        def predict(self, window, n_samples=32):
            # -> (n_samples, len(window.target_times), 2) absolute lon/lat
            ...

Deterministic models return `n_samples` identical trajectories; the
metrics handle that correctly (their sample spread is zero, so they
score identically on mean-based metrics and are simply uninformative on
calibration ones).

On the baselines: **persistence is not a serious bar.** Its error is
exactly how far the vessel moved, so beating it says only that the model
noticed the ship is moving. Constant velocity is the honest reference,
and CTRV is the honest reference for manoeuvring vessels -- which
matters here because turning is the measured dominant failure mode.
Report against all three.
"""
from abc import ABC, abstractmethod

import numpy as np

from .protocol import Window


DEG_PER_KM_LAT = 1.0 / 110.574


def _deg_per_km_lon(lat_deg):
    return 1.0 / (111.320 * max(np.cos(np.radians(lat_deg)), 1e-6))


class Predictor(ABC):
    """A model that can be scored on the benchmark."""

    name: str = "unnamed"
    #: whether the predictor produces genuinely different samples.
    #: Deterministic predictors are excluded from calibration metrics,
    #: which would otherwise report a meaningless zero spread.
    probabilistic: bool = False

    @abstractmethod
    def predict(self, window: Window, n_samples: int = 32) -> np.ndarray:
        """Return (n_samples, F, 2) absolute lon/lat at window.target_times."""

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r}>"


class PersistencePredictor(Predictor):
    """
    "The vessel stays where it is."

    Included because it is the historical baseline in this project, and
    because seeing it next to constant velocity makes clear how weak it
    is. Its error equals the vessel's actual displacement by definition.
    """
    name = "persistence"

    def predict(self, window, n_samples=32):
        F = len(window.target_times)
        path = np.repeat(window.anchor[None, :], F, axis=0)
        return np.repeat(path[None, ...], n_samples, axis=0)


class ConstantVelocityPredictor(Predictor):
    """
    Dead reckoning: last observed velocity, extrapolated linearly.

    Velocity is taken from the last two reports using their REAL elapsed
    time, so this is well defined for irregular sampling. Ships mostly
    travel straight, which makes this a genuinely strong baseline and the
    number a learned model has to beat to be worth anything.
    """
    name = "constant-velocity"

    def predict(self, window, n_samples=32):
        p, t = window.context_positions, window.context_times
        if len(p) < 2 or abs(t[-1] - t[-2]) < 1e-6:
            return PersistencePredictor().predict(window, n_samples)
        v = (p[-1] - p[-2]) / (t[-1] - t[-2])          # degrees per second
        path = np.stack([p[-1] + v * dt for dt in window.target_times])
        return np.repeat(path[None, ...], n_samples, axis=0)


class CTRVPredictor(Predictor):
    """
    Constant turn rate and velocity -- the standard manoeuvring model in
    vehicle and vessel tracking.

    Speed and heading come from the last two reports; turn rate from the
    last three. Where constant velocity assumes a straight line, this
    continues an arc, so it is the fair reference for exactly the cases
    where a straight-line model is known to fail. Given that turning is
    this project's dominant error driver, a learned model that beats
    constant velocity but not CTRV has mostly rediscovered circular
    motion.
    """
    name = "ctrv"

    def predict(self, window, n_samples=32):
        p, t = window.context_positions, window.context_times
        if len(p) < 3:
            return ConstantVelocityPredictor().predict(window, n_samples)

        lat0 = p[-1, 1]
        kx, ky = 1.0 / _deg_per_km_lon(lat0), 1.0 / DEG_PER_KM_LAT   # deg -> km
        xy = np.stack([(p[:, 0] - p[-1, 0]) * kx, (p[:, 1] - p[-1, 1]) * ky], axis=1)

        dt1, dt2 = t[-1] - t[-2], t[-2] - t[-3]
        if abs(dt1) < 1e-6 or abs(dt2) < 1e-6:
            return ConstantVelocityPredictor().predict(window, n_samples)

        v1 = (xy[-1] - xy[-2]) / dt1
        v0 = (xy[-2] - xy[-3]) / dt2
        speed = float(np.linalg.norm(v1))
        if speed < 1e-9:
            return PersistencePredictor().predict(window, n_samples)

        h1, h0 = np.arctan2(v1[1], v1[0]), np.arctan2(v0[1], v0[0])
        dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi)
        omega = dh / dt1                                       # rad/sec

        pts = []
        for dt in window.target_times:
            if abs(omega) < 1e-9:
                dx, dy = speed * dt * np.cos(h1), speed * dt * np.sin(h1)
            else:
                h2 = h1 + omega * dt
                dx = (speed / omega) * (np.sin(h2) - np.sin(h1))
                dy = (speed / omega) * (-np.cos(h2) + np.cos(h1))
            pts.append([p[-1, 0] + dx / kx, p[-1, 1] + dy / ky])
        path = np.asarray(pts)
        return np.repeat(path[None, ...], n_samples, axis=0)


class GaussianNoiseCVPredictor(Predictor):
    """
    Constant velocity with Gaussian spread that grows with horizon.

    A deliberately simple *probabilistic* reference. Any learned model
    claiming useful uncertainty should beat this on calibration, not just
    on point accuracy -- otherwise the sampling machinery is decoration.
    """
    name = "cv+noise"
    probabilistic = True

    def __init__(self, sigma_km_per_hour=1.5, seed=0):
        self.sigma = sigma_km_per_hour
        self.rng = np.random.default_rng(seed)

    def predict(self, window, n_samples=32):
        base = ConstantVelocityPredictor().predict(window, 1)[0]     # (F, 2)
        lat0 = window.anchor[1]
        out = np.empty((n_samples, base.shape[0], 2))
        for f, dt in enumerate(window.target_times):
            s_km = self.sigma * (dt / 3600.0)
            dlon = self.rng.normal(0, s_km * _deg_per_km_lon(lat0), n_samples)
            dlat = self.rng.normal(0, s_km * DEG_PER_KM_LAT, n_samples)
            out[:, f, 0] = base[f, 0] + dlon
            out[:, f, 1] = base[f, 1] + dlat
        return out


BASELINES = [
    PersistencePredictor,
    ConstantVelocityPredictor,
    CTRVPredictor,
    GaussianNoiseCVPredictor,
]


def default_baselines():
    return [cls() for cls in BASELINES]
