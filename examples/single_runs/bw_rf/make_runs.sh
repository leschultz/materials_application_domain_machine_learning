#!/bin/bash

sets=(
      "diffusion"
      )

grid=(0.001 0.01 0.1 1.0 10.0 100.0 1000.0 False)

for i in "${sets[@]}"
do

for j in "${grid[@]}"
do

    echo "runs/${i}/${j}"

    mkdir -p "runs/${i}"
    cp -r template "runs/${i}/${j}"

    cd "runs/${i}/${j}"
    sed -i "s/replace_data/'${i}'/g" fit.py
    sed -i "s/replace_bw/bandwidth=${j}/g" fit.py

    cd - > /dev/null

done
done
