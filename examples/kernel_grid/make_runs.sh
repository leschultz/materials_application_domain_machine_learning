#!/bin/bash

kernels=("gaussian" "tophat" "epanechnikov" "exponential" "linear" "cosine")

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

for k in "${kernels[@]}"
do

    job_dir="runs/data_${i}/model_${j}/kernel_${k}"

    mkdir -p ${job_dir}
    cp -r template/* ${job_dir}
    cd ${job_dir}

    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/replace_model/'${j}'/g" fit.py
    sed -i "s/replace_kernel/'${k}'/g" fit.py

    cd - > /dev/null

done
done
done
