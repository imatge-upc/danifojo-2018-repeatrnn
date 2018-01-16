#!/bin/bash

python parity.py --tau 0.001 --dont-print-results --lr 0.001 --vram-fraction 0.3 &
python parity.py --tau 0.0001 --dont-print-results --lr 0.001 --vram-fraction 0.3 &
python parity.py --tau 0.0 --dont-print-results --lr 0.001 --vram-fraction 0.3
