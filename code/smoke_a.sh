#!/usr/bin/env bash
# Wave 1-A GPU gate + 1-epoch smoke for every model (full data, the real input sizes).
set -u
cd ~/MM_Wave1/code
mkdir -p ~/MM_Wave1/logs/A_smoke
python gate_a.py --device cuda 2>&1 | tee ~/MM_Wave1/logs/A_smoke/gate.log
grep -q "GATE A: PASS" ~/MM_Wave1/logs/A_smoke/gate.log || { echo "GATE FAILED, smoke not started"; exit 1; }
for m in img aud late late_lin tfn lmf gblend ogmge; do
  python mm_a.py --model $m --epochs 1 --seed 777 --out ~/MM_Wave1/runs/A_smoke --force \
    > ~/MM_Wave1/logs/A_smoke/$m.log 2>&1
  tail -1 ~/MM_Wave1/logs/A_smoke/$m.log
done
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
echo "SMOKE A FINISHED"
