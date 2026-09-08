#!/usr/bin/env bash
#
# Reorganize the irregular-sampling worktree into a proper package.
#
# Run from ~/maritime-irregular. Uses `git mv` so history follows the
# files. Nothing is deleted -- the fixed-interval branch is untouched and
# remains the comparison arm.
#
# Import style changes from flat (`from mesh import ...`) to package
# (`from vtp.data.mesh import ...`). The sed pass below rewrites the
# common cases; check anything it reports as unresolved by hand.
#
#   bash reorganize.sh          # dry run: prints what it would do
#   bash reorganize.sh --apply  # actually do it
#
set -euo pipefail
APPLY=${1:-}
run() { if [ "$APPLY" = "--apply" ]; then eval "$@"; else echo "  would: $*"; fi }

echo "=== creating package layout ==="
for d in vtp vtp/data vtp/models vtp/training vtp/benchmark vtp/viz \
         configs notebooks slurm scripts benchmarks results assets; do
  run "mkdir -p $d"
done
for d in vtp vtp/data vtp/models vtp/training vtp/benchmark vtp/viz; do
  run "touch $d/__init__.py"
done

echo "=== data layer ==="
run "git mv mesh.py            vtp/data/mesh.py"
run "git mv coastline.py       vtp/data/coastline.py"
run "git mv irregular_ingest.py vtp/data/ingest.py"
run "git mv ais_cache.py       vtp/data/cache.py"
run "git mv graph_data.py      vtp/data/graphs.py"

echo "=== model layer ==="
run "git mv cached_attention.py vtp/models/attention.py"
run "git mv model.py            vtp/models/irregular_vtp.py"
run "git mv losses.py           vtp/models/losses.py"

echo "=== training layer ==="
run "git mv batching.py        vtp/training/batching.py"
run "git mv distributed.py     vtp/training/distributed.py"
run "git mv train_irregular.py vtp/training/train.py"

echo "=== viz layer ==="
run "git mv viz_irregular.py vtp/viz/plots.py"

echo "=== scripts and notebooks ==="
run "git mv export_metrics.py scripts/export_metrics.py 2>/dev/null || true"
run "git mv sweep.py          scripts/sweep.py 2>/dev/null || true"
run "mv *.slurm slurm/ 2>/dev/null || true"
run "mv *.ipynb notebooks/ 2>/dev/null || true"

echo
echo "=== rewriting imports ==="
# Old flat name -> new dotted path. Ordering matters: longer names first
# so 'irregular_ingest' is not partially matched by 'ingest'.
MAP=(
  "irregular_ingest:vtp.data.ingest"
  "cached_attention:vtp.models.attention"
  "ais_cache:vtp.data.cache"
  "graph_data:vtp.data.graphs"
  "coastline:vtp.data.coastline"
  "distributed:vtp.training.distributed"
  "batching:vtp.training.batching"
  "viz_irregular:vtp.viz.plots"
  "losses:vtp.models.losses"
  "model:vtp.models.irregular_vtp"
  "mesh:vtp.data.mesh"
)
for pair in "${MAP[@]}"; do
  old="${pair%%:*}"; new="${pair##*:}"
  run "grep -rl --include='*.py' --include='*.ipynb' -E '(from|import) ${old}\\b' . | xargs -r sed -i -E 's/(from|import) ${old}\\b/\\1 ${new}/g'"
done

echo
echo "=== next steps (not automated) ==="
cat <<'NOTE'
  1. pip install -e .            (so `import vtp` works from anywhere)
  2. Update slurm scripts:       python3 -m vtp.training.train  ...
  3. Check for stragglers:       grep -rn "^from \(mesh\|model\|losses\)" --include=*.py .
  4. Run a smoke test before trusting anything.

  The `model` -> `vtp.models.irregular_vtp` rewrite is the riskiest, since
  "model" is a common word. Review that diff specifically.
NOTE
