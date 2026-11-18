#!/bin/bash
# 보강 배치가 끝나길 기다렸다가 받아서 학습하고 int8 로 줄인다. SSH 가 끊겨도 돌게
# setsid nohup 으로 띄운다:  setsid nohup risk_model/run_all.sh > data/risk_train/run_all.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until $PY build_risk_augment.py collect | tee /dev/stderr | grep -q "^저장:"; do sleep 60; done
nice -n 10 $PY risk_model/train.py --epochs 5 --threads 3 2>&1 \
  | grep -v "Warn\|MISSING\|UNEXPECTED\|Notes\|ignored\|newly\|onnx.export\|padding_length\|is_causal\|opset9\|Loading weights\|LOAD REPORT\|^Key\|^---"
$PY -c "
from pathlib import Path
from risk_model.runtime import quantize
quantize(Path('models/risk_ko.onnx'), Path('models/risk_ko.int8.onnx'))
print('int8 저장:', round(Path('models/risk_ko.int8.onnx').stat().st_size/1e6,1), 'MB')"
echo "끝"
