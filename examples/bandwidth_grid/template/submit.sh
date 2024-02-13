#!/bin/sh
#SBATCH --partition=morgan
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=4000
#SBATCH --error=job.e.%J
#SBATCH --output=job.o.%J

./run.sh
