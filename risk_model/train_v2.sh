#!/bin/bash
# 2차 학습. mori 를 끈 채 4코어로 돌리고, 끝나면(실패해도) mori 를 다시 켠다.
#   setsid nohup risk_model/train_v2.sh > data/risk_train/train_v2.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
trap 'systemctl --user start mori; echo "mori 다시 켬"' EXIT
systemctl --user stop mori
PY=.venv/bin/python
$PY -u risk_model/train.py --epochs 8 --lr 1.5e-4 --threads 4 --out models/risk_ko_v2.onnx 2>&1 \
  | grep --line-buffered -v "Warn\|MISSING\|UNEXPECTED\|Notes\|ignored\|newly\|onnx.export\|padding_length\|is_causal\|opset9\|Loading weights\|LOAD REPORT\|^Key\|^---\|^$"
$PY -c "
from pathlib import Path
from risk_model.runtime import quantize
quantize(Path('models/risk_ko_v2.onnx'), Path('models/risk_ko_v2.int8.onnx'))
print('int8 저장:', round(Path('models/risk_ko_v2.int8.onnx').stat().st_size/1e6,1), 'MB')" 2>&1 | grep -v onnxruntime
echo "끝"
