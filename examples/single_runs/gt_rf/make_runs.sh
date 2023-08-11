#!/bin/bash

sets=(
      "friedman1"
      "fluence"
      "diffusion"
      "steel_yield_strength"
      "super_cond"
      )

gtgrid=(0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0)

for i in "${sets[@]}"
do

for j in "${gtgrid[@]}"
do

    run="runs/${i}/gt_${j}"
    echo ${run}

    mkdir -p ${run}
    cp -r template/* ${run}

    cd ${run}
    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/gts/gts=${j}/g" fit.py
    sed -i "s/gtb/gtb=${j}/g" fit.py

    cd - > /dev/null

done
done
