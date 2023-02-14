#!/bin/bash

export PYTHONPATH=$(pwd)/../..:$PYTHONPATH

rm -rf run_kde *.png
python3 kde.py
python3 statistics_plot.py
