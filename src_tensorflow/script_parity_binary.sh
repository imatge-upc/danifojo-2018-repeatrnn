#!/bin/bash

python parity.py --tau 0.001 --dont-print-results  --use-binary --lr 0.001 &
python parity.py --tau 0.0001 --dont-print-results  --use-binary --lr 0.001 &
python parity.py --tau 0.0 --dont-print-results  --use-binary --lr 0.001
