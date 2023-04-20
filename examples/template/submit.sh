#PBS -S /bin/bash
#PBS -m be
#PBS -q bardeen
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=72:00:00
#PBS -N job

cd $PBS_O_WORKDIR 

./run.sh
