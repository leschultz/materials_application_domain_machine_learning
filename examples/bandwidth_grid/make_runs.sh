#!/bin/bash

bws=(0.1 1.0 10.0 100.0 None)

data=(
      "diffusion"
      )

models=(
	"rf"
	)

for i in "${data[@]}"
do

for j in "${models[@]}"
do

for b in "${bws[@]}"
do

    job_dir="runs/data_${i}/model_${j}/bandwidth_${b}"

    mkdir -p ${job_dir}
    cp -r template/* ${job_dir}
    cd ${job_dir}

    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/replace_model/'${j}'/g" fit.py
    sed -i "s/replace_bandwidth/${b}/g" fit.py

    cd - > /dev/null

done
done
done
