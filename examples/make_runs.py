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

    if [ ${i} = "friedman1" ]
    then
        sed -i "s/replace_data/${i}(n_samples=1000, n_features=5)/g" run.py
    else
        sed -i "s/replace_data/${i}()/g" run.py
    fi

    cd - > /dev/null
done
