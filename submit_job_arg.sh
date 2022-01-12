#for WELL in {" ","--well"}; do
for WELL in "--well"; do
	#for ZM in "--zeromean"; do
	for ZM in "--zeromean" " "; do
		for IG in 0 1 2 3 4 5 6 7; do
			for SIGMA in 1e-3; do
				#for MOD in " " "--noalen" "--usedetig"; do
				#for MOD in " " "--usedetig"; do
				for MOD in "--usedetig"; do
					#for MAXL in 20 21 22 23 24; do
					for MAXL in 22; do
						ARGS="${WELL} --order 16 --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 4096 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --maxmsc 5 ${ZM} ${MOD} --expquad --glob"
						echo $ARGS
						sbatch job_arg.slurm "${ARGS}"
						sleep 1
					done
				done
			done
		done
	done
done
