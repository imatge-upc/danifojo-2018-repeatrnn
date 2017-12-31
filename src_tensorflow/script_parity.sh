#!/bin/bash

python parity.py --tau 0.001 --dont-print-results &
python parity.py --tau 0.0001 --dont-print-results &
python parity.py --tau 0.0 --dont-print-results
