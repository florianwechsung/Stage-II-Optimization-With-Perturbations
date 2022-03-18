#!/bin/bash
#
#SBATCH --job-name=corrections
#SBATCH --output=logs/slurm_correction_%A_%a.out
#SBATCH --error=logs/slurm_correction_%A_%a.err
#SBATCH -p cs
#SBATCH --time=2:00:00
#SBATCH --mem=15GB
##SBATCH --mem=60GB

#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=2

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate

mkdir -p logs
mkdir -p output
mkdir -p corrections

SAMPLEIDX=$SLURM_ARRAY_TASK_ID
#for SIGMA in 0.00010 0.00025 0.00050 0.00075 0.00100 0.00125 0.00150 0.00175 0.00200; do
for SIGMA in 0.00175 0.00200; do
	mpiexec python3 error_correction.py  --outdiridx 0 --sigma ${SIGMA} --sampleidx $SAMPLEIDX
	mpiexec python3 error_correction.py  --outdiridx 1 --sigma ${SIGMA} --sampleidx $SAMPLEIDX
	mpiexec python3 error_correction.py  --outdiridx 2 --sigma ${SIGMA} --sampleidx $SAMPLEIDX
	mpiexec python3 error_correction.py  --outdiridx 3 --sigma ${SIGMA} --sampleidx $SAMPLEIDX
	mpiexec python3 error_correction.py  --outdiridx 0 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --well
	mpiexec python3 error_correction.py  --outdiridx 1 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --well
	mpiexec python3 error_correction.py  --outdiridx 2 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --well
	mpiexec python3 error_correction.py  --outdiridx 3 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --well
	mpiexec python3 error_correction.py  --outdiridx 0 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr 
	mpiexec python3 error_correction.py  --outdiridx 1 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr 
	mpiexec python3 error_correction.py  --outdiridx 2 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr 
	mpiexec python3 error_correction.py  --outdiridx 3 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr 
	mpiexec python3 error_correction.py  --outdiridx 0 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr  --well
	mpiexec python3 error_correction.py  --outdiridx 1 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr  --well
	mpiexec python3 error_correction.py  --outdiridx 2 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr  --well
	mpiexec python3 error_correction.py  --outdiridx 3 --sigma ${SIGMA} --sampleidx $SAMPLEIDX --adjcurr  --well
done
