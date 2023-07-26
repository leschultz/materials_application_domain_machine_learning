#!/bin/bash

export PYTHONPATH=$(pwd)/../../../../../src:$PYTHONPATH

rm -rf run
python3 fit.py
