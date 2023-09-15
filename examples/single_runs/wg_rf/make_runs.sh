#!/bin/bash

sets=(
      "make_regression"
      "friedman1"
      "fluence"
      "diffusion"
      "steel_yield_strength"
      "super_cond"
      )

wtgrid=(
	features
	bandwidths
	scores
	none
        )

for i in "${sets[@]}"
do

for j in "${wtgrid[@]}"
do

    run="runs/${i}/wt_${j}"
    echo ${run}

    mkdir -p ${run}
    cp -r template/* ${run}

    cd ${run}
    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/weigh_replace/weigh='${j}'/g" fit.py

    cd - > /dev/null

done
done
