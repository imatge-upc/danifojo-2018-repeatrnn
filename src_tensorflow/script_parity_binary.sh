#!/bin/bash

python parity.py --tau 1. --dont-print-results  --use-binary --lr 0.001 --vram-fraction 0.3 &
python parity.py --tau 0.5 --dont-print-results  --use-binary --lr 0.001 --vram-fraction 0.3 &
python parity.py --tau 0.1 --dont-print-results  --use-binary --lr 0.001 --vram-fraction 0.3
