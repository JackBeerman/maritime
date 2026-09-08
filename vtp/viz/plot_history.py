"""
Plot training loss against held-out validation from a run's history file.

The gap between the two curves is the overfitting signal: training loss
falling while validation error flattens or rises means the extra epochs
are memorising, and the `_best.pt` checkpoint (not the last one) is the
model to use.

    python3 plot_history.py checkpoints/irregular_history.json
    python3 plot_history.py checkpoints/irregular_history.json out.png
"""
import json
import sys

import matplotlib.pyplot as plt


def main(path, out=None):
    hist = json.load(open(path))
    if not hist:
        print("history is empty -- has a validation check run yet?")
        return

    ep = [h['epoch'] for h in hist]
    train = [h['train_loss'] for h in hist]
    val = [h.get('model_mean_error_km', h.get('fde_km_moving')) for h in hist]
    base = [h.get('trivial_baseline_mean_error_km') for h in hist]
    corr = [h.get('spread_correlation') for h in hist]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, train, 'o-', color='tab:blue')
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss (normalized units)')
    ax.set_title('Training loss'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep, val, 'o-', color='tab:red', label='validation error')
    if any(b is not None for b in base):
        ax.plot(ep, base, '--', color='0.6', label='persistence baseline')
    best_i = min(range(len(val)), key=lambda i: val[i])
    ax.axvline(ep[best_i], color='tab:green', ls=':',
               label=f'best: {val[best_i]:.2f} km @ ep {ep[best_i]}')
    ax.set_xlabel('epoch'); ax.set_ylabel('held-out error (km)')
    ax.set_title('Validation'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep, corr, 'o-', color='tab:green')
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_xlabel('epoch'); ax.set_ylabel('spread / displacement correlation')
    ax.set_title('Uncertainty calibration'); ax.grid(alpha=0.3)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
        print(f"saved {out}")
    else:
        plt.show()

    print(f"\nbest validation {val[best_i]:.3f} km at epoch {ep[best_i]} "
          f"(of {len(hist)} checks)")
    if best_i < len(val) - 1:
        worse = val[-1] > val[best_i]
        print(f"  latest is {val[-1]:.3f} km at epoch {ep[-1]} -- "
              f"{'worse' if worse else 'better'} than best"
              + ("; use the _best.pt checkpoint" if worse else ""))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)