#!/bin/bash
#
#SBATCH --job-name=deterministic
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH -p cs
#SBATCH --time=6:00:00
#SBATCH --mem=10GB

#SBATCH --nodes=1
#SBATCH --tasks=8

#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1


module load intel/19.1.2
module load python/intel/3.8.6
module load openmpi/intel/4.0.5
export OMP_NUM_THREADS=4

cd /scratch/fw18/stochastic-stage-two/src/Stochastic-Stage-II-Optimization-2/
source /scratch/fw18/stochastic-stage-two/bin/activate



# run with sbatch --array=0-7 job_det.slurm

IG=$SLURM_ARRAY_TASK_ID

ORDER=16


#MAXL=18
#MAXL=20
#MAXL=22
MAXL=24

mkdir -p logs
mkdir -p output

python3 driver.py --well --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 0 --mindist 0.10 --maxkappa 5 --maxmsc 5 --order ${ORDER} --expquad --noalen
python3 driver.py        --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 0 --mindist 0.10 --maxkappa 5 --maxmsc 5 --order ${ORDER} --expquad --noalen
python3 driver.py --well --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 0 --mindist 0.10 --maxkappa 5 --maxmsc 5 --order ${ORDER} --expquad --alstart 0
python3 driver.py        --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 0 --mindist 0.10 --maxkappa 5 --maxmsc 5 --order ${ORDER} --expquad --alstart 0
#python3 driver.py --well --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 0 --mindist 0.10 --maxkappa 5 --maxmsc 5 --order ${ORDER} --expquad --alstart 1
#python3 driver.py        --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 0 --mindist 0.10 --maxkappa 5 --maxmsc 5 --order ${ORDER} --expquad --alstart 1
