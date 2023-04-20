#!/bin/bash

export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH

rm -rf run
python3 run.py
