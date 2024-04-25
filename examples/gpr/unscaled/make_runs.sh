#!/bin/bash

data=(
      "friedman1"
      "fluence"
      "diffusion"
      "strength"
      )

models=(
	"rf"
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

    # Define the repeats
    if [ "${i}" == "fluence" ] && [ "${j}" == "bnn" ]; then
        r=3
    elif [ "${i}" == "friedman1" ] && [ "${j}" == "bnn" ]; then
        r=3
    elif [ "${i}" == "supercond" ] && [ "${j}" == "bnn" ]; then
        r=2
    else
        r=5
    fi

    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/replace_model/'${j}'/g" fit.py
    sed -i "s/replace_repeats/${r}/g" fit.py

    cd - > /dev/null

done
done
