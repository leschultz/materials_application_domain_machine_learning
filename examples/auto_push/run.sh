#!/bin/bash

export PYTHONPATH=$(pwd)/../..:$PYTHONPATH

rm -rf run
python3 fit.py
