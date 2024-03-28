#PBS -S /bin/bash
#PBS -q izabela
#PBS -l select=1:ncpus=12:mpiprocs=12
#PBS -l walltime=72:00:00
#PBS -N job

cd $PBS_O_WORKDIR

./run.sh
