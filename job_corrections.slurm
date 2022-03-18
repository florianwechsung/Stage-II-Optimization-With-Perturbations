#!/bin/bash
#
#SBATCH --job-name=corrections
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=22GB

#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=4

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate

mkdir -p logs
mkdir -p output

WELL=""
#WELL="--well"
SIDX=`expr $SLURM_ARRAY_TASK_ID - 1`

for OUTDIRIDX in 0 1 2 3; do
    for SIGMA in 1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4 8e-4 9e-4 1e-3; do
        for CL in 1 2 3; do
            python3 error_correction.py  $WELL  --outdiridx $OUTDIRIDX --sampleidx $SIDX --sigma $SIGMA --correctionlevel $CL
        done
    done
done
