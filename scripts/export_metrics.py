"""
Export held-out evaluation metrics to JSON for cross-branch comparison.

The two branches cannot be imported into one Python process -- both
define `model.py` and `graph_data.py` -- so `compare_approaches.ipynb`
compares them through files instead. Run this once in each worktree.

    # fixed-interval branch
    cd ~/maritime
    python3 export_metrics.py --checkpoint checkpoints/checkpoint_batched.pt \
        --out results/main_metrics.json --seq-len 12 --future-len 4

    # irregular branch
    cd ~/maritime-irregular
    python3 export_metrics.py --checkpoint checkpoints/irregular_large.pt \
        --out ~/maritime/results/irregular_metrics.json \
        --seq-len 24 --future-len 8 --min-ping-gap-sec 900

Pass the SAME vessel-selection arguments the training run used -- the
evaluation set is reconstructed from them, and a mismatch silently
evaluates on vessels the model trained on.
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ais-path', default='/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv')
    p.add_argument('--land-shp', default='data/ne_10m_land/ne_10m_land.shp')
    p.add_argument('--ports-csv', default='data/ports_denmark.csv')
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--future-len', type=int, default=4)
    p.add_argument('--n-underway', type=int, default=300)
    p.add_argument('--n-stationary', type=int, default=300)
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--val-seed', type=int, default=42)
    # irregular-branch only; ignored on main
    p.add_argument('--min-ping-gap-sec', type=float, default=900.0)
    p.add_argument('--staleness-cutoff-sec', type=float, default=1800.0)
    p.add_argument('--neighbor-radius-deg', type=float, default=0.5)
    p.add_argument('--max-neighbors', type=int, default=150)
    args = p.parse_args()

    # Detect which branch we're in by which viz module is present, rather
    # than by directory name -- the worktrees can be renamed.
    try:
        import vtp.viz.plots as V
        branch = 'irregular'
    except ImportError:
        import viz as V
        branch = 'main'
    print(f"branch: {branch}")

    if branch == 'irregular':
        ctx = V.setup(ais_path=args.ais_path, land_shp=args.land_shp,
                      ports_csv=args.ports_csv, seq_len=args.seq_len,
                      future_len=args.future_len,
                      min_ping_gap_sec=args.min_ping_gap_sec,
                      staleness_cutoff_sec=args.staleness_cutoff_sec,
                      neighbor_radius_deg=args.neighbor_radius_deg,
                      max_neighbors=args.max_neighbors,
                      n_underway=args.n_underway, n_stationary=args.n_stationary,
                      val_seed=args.val_seed, held_out_only=True)
    else:
        ctx = V.setup(ais_path=args.ais_path, land_shp=args.land_shp,
                      ports_csv=args.ports_csv, seq_len=args.seq_len,
                      future_len=args.future_len,
                      n_underway=args.n_underway, n_stationary=args.n_stationary,
                      val_seed=args.val_seed, held_out_only=True)

    model, scale, meta = V.load_checkpoint_model(args.checkpoint)
    print(f"checkpoint: {meta}")

    metrics = V.evaluate(model, ctx.combined_val, scale, n_samples=args.n_samples)
    metrics['branch'] = branch
    metrics['checkpoint'] = os.path.basename(args.checkpoint)
    metrics.update({f'ckpt_{k}': v for k, v in meta.items()})
    if branch == 'main':
        # main always predicts a fixed grid, so the horizon is implied
        # rather than measured -- record it so the comparison table can
        # check the two branches are actually at matched horizons
        metrics['median_horizon_min'] = args.future_len * 15.0
        metrics['per_step_error_km'] = V.per_step_errors(
            model, ctx.combined_val, scale, verbose=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {args.out}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")


if __name__ == '__main__':
    main()