#!/bin/bash

python addition.py --tau 0.01 --dont-print-results --lr 0.001 --vram-fraction 0.3 &
python addition.py --tau 0.001 --dont-print-results --lr 0.001 --vram-fraction 0.3 &
python addition.py --tau 0.0 --dont-print-results --lr 0.001 --vram-fraction 0.3
