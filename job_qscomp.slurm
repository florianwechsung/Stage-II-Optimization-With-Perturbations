#!/bin/bash
#
#SBATCH --job-name=deterministic
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH -p cs
#SBATCH --time=12:00:00
#SBATCH --mem=10GB

#SBATCH --nodes=1
#SBATCH --tasks=4

#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=4

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate



# run with sbatch --array=0-63 job_qscomp.slurm

mkdir -p logs
mkdir -p output

SIDX=$SLURM_ARRAY_TASK_ID
for OUTDIRIDX in 2 3; do
	for SIGMA in 1e-4 2.5e-4 5e-4 7.5e-4 1e-3; do
		for CL in 0 1 2 3; do
			python3 error_correction.py --well --outdiridx $OUTDIRIDX --sampleidx $SIDX --sigma $SIGMA --correctionlevel $CL
			python3 eval_compute_bmn.py --well --outdiridx $OUTDIRIDX --sampleidx $SIDX --sigma $SIGMA --correctionlevel $CL
		done
	done
done

