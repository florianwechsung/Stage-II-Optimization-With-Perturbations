#!/bin/bash
#
#SBATCH --job-name=stochastic
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
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



# run with sbatch --array=0-7 job.slurm

IG=$SLURM_ARRAY_TASK_ID
#MAXL=17
#MAXL=18
#MAXL=19
#MAXL=20
#MAXL=21
#MAXL=22
#MAXL=23
MAXL=24

ORDER=16

SIGMA=3e-3
#SIGMA=2e-3
#SIGMA=1e-3
#SIGMA=5e-4

#mpiexec python3 driver.py        --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5
#mpiexec python3 driver.py --well --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5

mpiexec python3 driver.py        --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --zeromean
mpiexec python3 driver.py --well --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --zeromean

#mpiexec python3 driver.py        --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --zeromean --usedetig
#mpiexec python3 driver.py --well --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --zeromean --usedetig

#mpiexec python3 driver.py        --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --zeromean --noalen
#mpiexec python3 driver.py --well --order ${ORDER} --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --zeromean --noalen
