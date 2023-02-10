#!/bin/bash

export PYTHONPATH=$(pwd)/../..:$PYTHONPATH

python3 kde.py
python3 statistics_plot.py
