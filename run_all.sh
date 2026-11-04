#!/usr/bin/env bash
# usage: bash run_all.sh | TINY=1 bash run_all.sh (CPU smoke)
# env: MODEL, SEEDS, OUT, EXTRA
set -euo pipefail
cd "$(dirname "$0")"
SEEDS="${SEEDS:-0 1 2}"
if [ "${TINY:-0}" = "1" ]; then
  OUT="${OUT:-results_smoke}"; ARGS="--tiny"
else
  OUT="${OUT:-results}"; ARGS="--model ${MODEL:-Qwen/Qwen2.5-7B}"
fi
mkdir -p "$OUT/logs"
for s in $SEEDS; do
  echo "=== seed $s ==="
  python sequential_books.py $ARGS --seed "$s" --out "$OUT" ${EXTRA:-} 2>&1 | tee "$OUT/logs/seed$s.log"
done
python make_report.py --dir "$OUT"
echo "Wrote $OUT/results.csv $OUT/forgetting.png $OUT/summary.md"
