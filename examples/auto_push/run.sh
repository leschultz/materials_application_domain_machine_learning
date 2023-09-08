#!/bin/bash

export PYTHONPATH=$(pwd)/../../src:$PYTHONPATH

rm -rf runs
python3 fit.py
