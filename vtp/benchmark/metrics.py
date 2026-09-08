"""
Metrics, using the names the trajectory-prediction literature uses so
results drop straight into a comparison table.

  ADE   average displacement error -- mean error over all predicted steps
  FDE   final displacement error   -- error at the last predicted step
  minADE_k / minFDE_k  best of k samples; the standard way to score a
        multi-modal predictor without punishing it for offering
        alternatives that did not happen

Point metrics (ADE/FDE) are computed from the sample MEAN, which is what
a deterministic consumer of the model would use. minADE/minFDE are
computed over samples. Reporting both matters: a model can look strong
on minADE simply by having wide spread, so a wide gap between ADE and
minADE is a signal to check calibration rather than a win.

Also reported:

  * **energy score** -- a proper scoring rule for the whole predicted
    distribution, and the objective this project trains against. Lower
    is better, and unlike ADE it cannot be gamed by widening spread.
  * **spread-error correlation** -- does the model's own uncertainty
    track how wrong it turns out to be? This is the question the
    sampling head exists to answer.
  * **skill scores** against each baseline: 1 - error/baseline_error.
    Positive means better than the baseline, 0 means parity, negative
    means worse. Reporting a skill score rather than a raw win-rate
    avoids the trap where "beats baseline 89% of the time" hides that
    the wins are tiny and the losses large.
"""
import numpy as np

from .protocol import haversine_km


def _pairwise_km(a, b):
    return haversine_km(a[..., 0], a[..., 1], b[..., 0], b[..., 1])


def window_metrics(pred, window, n_samples_for_min=None):
    """
    pred: (S, F, 2) absolute lon/lat samples for one window.
    Returns per-window metrics in km.
    """
    truth = window.target_positions                 # (F, 2)
    mean_path = pred.mean(axis=0)                   # (F, 2)

    err_per_step = _pairwise_km(mean_path, truth)   # (F,)
    ade = float(err_per_step.mean())
    fde = float(err_per_step[-1])

    # per-sample errors, for the min-over-k metrics
    S = pred.shape[0]
    k = S if n_samples_for_min is None else min(n_samples_for_min, S)
    sample_err = np.stack([_pairwise_km(pred[s], truth) for s in range(k)])  # (k, F)
    min_ade = float(sample_err.mean(axis=1).min())
    min_fde = float(sample_err[:, -1].min())

    # spread: mean distance of samples from their own mean at the final step
    centre = mean_path[-1]
    spread = float(np.mean(_pairwise_km(pred[:, -1, :], centre[None, :])))

    displacement = float(haversine_km(window.anchor[0], window.anchor[1],
                                       truth[-1, 0], truth[-1, 1]))

    return {'ade_km': ade, 'fde_km': fde,
            'min_ade_km': min_ade, 'min_fde_km': min_fde,
            'spread_km': spread, 'displacement_km': displacement,
            'horizon_sec': window.horizon_sec,
            'err_per_step_km': err_per_step}


def energy_score(pred, window, beta=1.0):
    """
    Energy score for one window: E||X - y|| - 0.5 E||X - X'||, in km.

    A proper scoring rule, so it cannot be improved by inflating spread
    -- which is exactly why it is the training objective here and worth
    reporting alongside ADE.
    """
    truth = window.target_positions
    S = pred.shape[0]
    term1 = np.mean([_pairwise_km(pred[s], truth).sum() ** beta for s in range(S)])
    if S < 2:
        return float(term1)
    idx = np.random.default_rng(0).permutation(S)
    term2 = np.mean([_pairwise_km(pred[i], pred[j]).sum() ** beta
                     for i, j in zip(idx[:-1], idx[1:])])
    return float(term1 - 0.5 * term2)


def evaluate_predictor(predictor, benchmark, n_samples=32,
                        min_move_km=1.0, horizon_buckets_min=(0, 15, 30, 60, 120, 1e9),
                        verbose=False):
    """
    Score one predictor over a whole benchmark set.

    `min_move_km` splits out genuinely moving windows. Aggregate metrics
    over all windows are dominated by stationary vessels (where every
    method scores near zero), so the moving subset is where methods
    actually separate -- both are reported.
    """
    rows, energies = [], []
    for w in benchmark.windows:
        pred = predictor.predict(w, n_samples=n_samples)
        m = window_metrics(pred, w)
        m['energy_score'] = energy_score(pred, w)
        m['regime'] = w.regime
        m['mmsi'] = w.mmsi
        rows.append(m)

    ade = np.array([r['ade_km'] for r in rows])
    fde = np.array([r['fde_km'] for r in rows])
    min_ade = np.array([r['min_ade_km'] for r in rows])
    min_fde = np.array([r['min_fde_km'] for r in rows])
    spread = np.array([r['spread_km'] for r in rows])
    disp = np.array([r['displacement_km'] for r in rows])
    es = np.array([r['energy_score'] for r in rows])
    hor = np.array([r['horizon_sec'] for r in rows]) / 60.0

    moving = disp > min_move_km
    out = {
        'predictor': predictor.name,
        'n_windows': len(rows),
        'n_moving': int(moving.sum()),
        'ade_km': float(ade.mean()),
        'fde_km': float(fde.mean()),
        'min_ade_km': float(min_ade.mean()),
        'min_fde_km': float(min_fde.mean()),
        'energy_score': float(es.mean()),
        'ade_km_moving': float(ade[moving].mean()) if moving.any() else float('nan'),
        'fde_km_moving': float(fde[moving].mean()) if moving.any() else float('nan'),
        'median_horizon_min': float(np.median(hor)),
    }
    # calibration is meaningless for a deterministic predictor (spread is
    # identically zero), so report it only where it means something
    if getattr(predictor, 'probabilistic', False) and spread.std() > 1e-12:
        out['spread_error_corr'] = float(np.corrcoef(spread, fde)[0, 1])
        out['spread_displacement_corr'] = float(np.corrcoef(spread, disp)[0, 1])
    else:
        out['spread_error_corr'] = float('nan')
        out['spread_displacement_corr'] = float('nan')

    # error by horizon bucket -- essential when windows have different
    # horizons, since a single mean silently mixes 5-min and 90-min tasks
    buckets = {}
    edges = list(horizon_buckets_min)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (hor >= lo) & (hor < hi) & moving
        if sel.sum() >= 5:
            label = f"{int(lo)}-{int(hi)}min" if hi < 1e8 else f">{int(lo)}min"
            buckets[label] = {'n': int(sel.sum()),
                              'ade_km': float(ade[sel].mean()),
                              'fde_km': float(fde[sel].mean())}
    out['by_horizon'] = buckets

    if verbose:
        print(f"{predictor.name}: ADE {out['ade_km']:.3f} km | FDE {out['fde_km']:.3f} km "
              f"| moving FDE {out['fde_km_moving']:.3f} km")
    return out, rows


def compare_predictors(predictors, benchmark, n_samples=32, reference='constant-velocity',
                        min_move_km=1.0):
    """
    Evaluate several predictors on the same windows and return a table.

    Skill is reported against `reference` (constant velocity by default,
    since persistence is too weak to be informative): positive means the
    predictor beats it, negative means a straight line would have done
    better.
    """
    import pandas as pd

    results, per_window = {}, {}
    for p in predictors:
        res, rows = evaluate_predictor(p, benchmark, n_samples=n_samples,
                                        min_move_km=min_move_km)
        results[p.name] = res
        per_window[p.name] = rows

    if reference in results:
        ref_fde = results[reference]['fde_km_moving']
        ref_ade = results[reference]['ade_km_moving']
        for name, r in results.items():
            r['fde_skill_vs_' + reference] = float(1 - r['fde_km_moving'] / ref_fde) if ref_fde else float('nan')
            r['ade_skill_vs_' + reference] = float(1 - r['ade_km_moving'] / ref_ade) if ref_ade else float('nan')

    cols = ['n_windows', 'n_moving', 'median_horizon_min',
            'ade_km', 'fde_km', 'ade_km_moving', 'fde_km_moving',
            'min_ade_km', 'min_fde_km', 'energy_score',
            'spread_error_corr']
    cols += [c for c in next(iter(results.values())) if c.startswith(('fde_skill', 'ade_skill'))]
    df = pd.DataFrame(results).T[cols]
    return df, results, per_window
