#!/bin/bash

python addition.py --tau 0.1 --dont-print-results --use-binary --lr 0.001 --vram-fraction 0.3 &
python addition.py --tau 0.5 --dont-print-results --use-binary --lr 0.001 --vram-fraction 0.3 &
python addition.py --tau 1. --dont-print-results --use-binary --lr 0.001 --vram-fraction 0.3
