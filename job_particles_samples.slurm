#!/bin/bash
#

#SBATCH --job-name=particles
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH -p cs
#SBATCH --time=24:00:00
#SBATCH --mem=80GB

#SBATCH --nodes=4
#SBATCH --tasks=12
#SBATCH --tasks-per-node=3
#SBATCH --cpus-per-task=2


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=2

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate

mkdir -p logs
mkdir -p output


#WELL=""
WELL="--well"
SIDX=$SLURM_ARRAY_TASK_ID

# run with sbatch --array=0-63 job_particles_samples.slurm

#for OUTDIRIDX in 3; do
#    for SIGMA in 1e-3; do
#        for CL in 1 2 3; do
#            mpiexec python3 eval_particles.py    $WELL  --outdiridx $OUTDIRIDX --sampleidx $SIDX --sigma $SIGMA --correctionlevel $CL --resolution 60 --spawnidx 25 --seed $SIDX --nparticles 1200
#        done
#    done
#done
 
#for OUTDIRIDX in 2 3 6 7; do


#for OUTDIRIDX in 0 1 2 3; do
for OUTDIRIDX in 4 5 6 7; do
    for SIGMA in 1e-3; do
        for CL in 0; do
            mpiexec python3 eval_particles.py    $WELL  --outdiridx $OUTDIRIDX --sampleidx $SIDX --sigma $SIGMA --correctionlevel $CL --resolution 60 --spawnidx 25 --seed $SIDX --nparticles 1200
        done
    done
done

