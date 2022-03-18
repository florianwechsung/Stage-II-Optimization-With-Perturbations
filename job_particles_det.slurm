#!/bin/bash
#
#SBATCH --job-name=particle
#SBATCH --output=logs/slurm_particle_%A_%a.out
#SBATCH --error=logs/slurm_particle_%A_%a.err
#SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=90GB

#SBATCH --nodes=20
#SBATCH --tasks=200
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1



module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=1

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate

mkdir -p logs
mkdir -p output

# run with sbatch --array=0-7 job_particles_det.slurm

OUTDIRIDX=$SLURM_ARRAY_TASK_ID


#mpiexec python3 eval_particles.py --outdiridx $OUTDIRIDX --sigma 0  --sampleidx -1 --spawnidx 25 --resolution 60 --nparticles 20000
mpiexec python3 eval_particles.py --outdiridx $OUTDIRIDX --sigma 0  --sampleidx -1 --spawnidx 25 --resolution 60 --well --nparticles 20000

