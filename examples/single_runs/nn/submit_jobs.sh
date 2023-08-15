#!/bin/bash

submit=submit.sh
for i in $(find ${1} -type f -name ${submit})
do
	cd $(dirname ${i})
	sbatch ${submit}
	cd - > /dev/null
done
