#!/bin/bash
#
#SBATCH --job-name=qfm
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH -p cs
#SBATCH --time=2:00:00
#SBATCH --mem=16GB

#SBATCH --nodes=1
#SBATCH --tasks=8

#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=8

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate



# run with sbatch --array=0-63 job_qfm.slurm

sampleidx=$SLURM_ARRAY_TASK_ID

mkdir -p qfmsurfaces
#python3 eval_compute_qfm.py --well --outdiridx 3 --sigma 0.001  --sampleidx $sampleidx --flux 0.3
#python3 eval_compute_qfm.py --well --outdiridx 3 --sigma 0.001  --sampleidx $sampleidx --flux 1.0
#python3 eval_compute_qfm.py --well --outdiridx 3 --sigma 0.001  --sampleidx $sampleidx --flux 0.3 --correction
#python3 eval_compute_qfm.py --well --outdiridx 3 --sigma 0.001  --sampleidx $sampleidx --flux 1.0 --correction
#python3 eval_compute_qfm.py --well --outdiridx 3 --sigma 0.001  --sampleidx $sampleidx --flux 0.3 --correction --adjcurr
#python3 eval_compute_qfm.py --well --outdiridx 3 --sigma 0.001  --sampleidx $sampleidx --flux 1.0 --correction --adjcurr

#python3 eval_compute_qfm.py --well --outdiridx 2 --sigma 0.001  --sampleidx $sampleidx --flux 0.3 --zeromean
#python3 eval_compute_qfm.py --well --outdiridx 2 --sigma 0.001  --sampleidx $sampleidx --flux 1.0 --zeromean
#python3 eval_compute_qfm.py --well --outdiridx 5 --sigma 0.001  --sampleidx $sampleidx --flux 0.3 --zeromean
#python3 eval_compute_qfm.py --well --outdiridx 5 --sigma 0.001  --sampleidx $sampleidx --flux 1.0 --zeromean
python3 eval_compute_qfm.py --well --outdiridx 7 --sigma 0.001  --sampleidx $sampleidx --flux 0.3
python3 eval_compute_qfm.py --well --outdiridx 7 --sigma 0.001  --sampleidx $sampleidx --flux 1.0
