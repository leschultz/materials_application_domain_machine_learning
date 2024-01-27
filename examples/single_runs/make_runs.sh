#!/bin/bash

data=(
      "friedman1"
      "fluence"
      "diffusion"
      "strength"
      "supercond"
      )

models=(
	"rf"
	"bols"
	"bsvr"
	"bnn"
	)

for i in "${data[@]}"
do

for j in "${models[@]}"
do

    echo "Making (data, model)=(${i}, ${j})"
    job_dir="runs/data_${i}/model_${j}"

    mkdir -p ${job_dir}
    cp -r template/* ${job_dir}
    cd ${job_dir}

    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/replace_model/'${j}'/g" fit.py

    cd - > /dev/null

done
done
