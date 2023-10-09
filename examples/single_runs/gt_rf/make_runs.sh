#!/bin/bash

sets=(
      "diffusion"
      )

gtgrid=(
	0.001
       	0.002
       	0.003
       	0.004
       	0.005
       	0.006
       	0.007
       	0.008
       	0.009
       	0.01
       	0.02
       	0.03
       	0.04
       	0.05
       	0.06
       	0.07
       	0.08
       	0.09
       	0.1
       	0.2
       	0.3
       	0.4
       	0.5
       	0.6
       	0.7
       	0.8
       	0.9
       	1.0
       	2.0
       	3.0
       	4.0
       	5.0
       	6.0
       	7.0
       	8.0
       	9.0
       	10.0
        )

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
