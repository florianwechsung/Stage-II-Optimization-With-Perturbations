#for WELL in {" ","--well"}; do
for WELL in "--well"; do
	#for ZM in "--zeromean" " "; do
	for ZM in " "; do
		for IG in 0 1 2 3 4 5 6 7; do
			for SIGMA in 5e-4 1e-3; do
				for MOD in " " "--noalen" "--usedetig"; do
					#for MAXL in 20 21 22 23 24; do
					for MAXL in 22; do
						ARGS="${WELL} --order 16 --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 1024 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --maxmsc 5 ${ZM} ${MOD}"
						echo $ARGS
						sbatch job_arg.slurm "${ARGS}"
						sleep 1
					done
				done
			done
		done
	done
done
