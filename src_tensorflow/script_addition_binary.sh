#!/bin/bash

python addition.py --tau 0.001 --dont-print-results --use-binary --lr 0.001 &
python addition.py --tau 0.0001 --dont-print-results --use-binary --lr 0.001 &
python addition.py --tau 0.0 --dont-print-results --use-binary --lr 0.001
