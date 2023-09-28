#!/bin/bash

sets=(
      "make_regression"
      "friedman1"
      "fluence"
      "diffusion"
      "diffusion_all_features"
      "steel_yield_strength"
      "steel_yield_strength_all_features"
      "super_cond"
      )

mkdir -p runs
for i in "${sets[@]}"
do

    echo ${i}
    cp -r template "runs/${i}"
    cd "runs/${i}"

    sed -i "s/replace_data/'${i}'/g" fit.py

    cd - > /dev/null
done
