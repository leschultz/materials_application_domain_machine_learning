#!/bin/bash

export PYTHONPATH=$(pwd)/../../../../../../src:$PYTHONPATH

rm -rf run
time python3 fit.py
