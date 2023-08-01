#!/bin/bash

sets=(
      "diffusion"
      "friedman1"
      "fluence"
      "diffusion"
      "steel_yield_strength"
      "super_cond"
      )

grid=("gaussian" "tophat" "epanechnikov" "exponential" "linear" "cosine")

for i in "${sets[@]}"
do

for j in "${grid[@]}"
do

    echo "runs/${i}/${j}"

    mkdir -p "runs/${i}"
    cp -r template "runs/${i}/${j}"

    cd "runs/${i}/${j}"
    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/replace_kernel/'${j}'/g" fit.py

    cd - > /dev/null

done
done
