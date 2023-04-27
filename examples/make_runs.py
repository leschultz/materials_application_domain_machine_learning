#!/bin/bash

sets=(
      "friedman1"
      "fluence"
      "diffusion"
      "steel_yield_strength"
      )

mkdir -p runs
for i in "${sets[@]}"
do

    echo ${i}
    cp -r template "runs/${i}"
    cd "runs/${i}"
    sed -i "s/replace_data/${i}/g" run.py
    cd - > /dev/null
done
