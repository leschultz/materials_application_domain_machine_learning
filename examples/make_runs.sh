#!/bin/bash

sets=(
      "friedman1"
      "fluence"
      "diffusion"
      "steel_yield_strength"
      "super_cond"
      )

mkdir -p runs
for i in "${sets[@]}"
do

    echo ${i}
    cp -r template "runs/${i}"
    cd "runs/${i}"

    sed -i "s/replace_data/'${i}'/g" fit.py

    if [ ${i} = "friedman1" ]
    then
        sed -i "s/replace_args/, n_samples=1000, n_features=5/g" fit.py
    else
        sed -i "s/replace_args//g" fit.py
    fi

    cd - > /dev/null
done
