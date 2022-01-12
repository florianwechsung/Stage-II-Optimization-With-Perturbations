#!/bin/bash
#
#SBATCH --job-name=stochastic
#SBATCH --output=logs/slurm_%A.out
#SBATCH --error=logs/slurm_%A.err
#SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=20GB

#SBATCH --nodes=8
#SBATCH --tasks=128
#SBATCH --tasks-per-node=16


#SBATCH --cpus-per-task=1


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=1

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate

mkdir -p logs
mkdir -p output

echo "About to execute python3 driver.py ${1}"
mpiexec python3 driver.py ${1}
