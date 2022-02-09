#for WELL in {" ","--well"}; do
for WELL in " " "--well"; do
	#for ZM in "--zeromean"; do
	for ZM in " "; do
		for IG in 0 1 2 3 4 5 6 7; do
			for SIGMA in 1e-3; do
				for USEIG in " ""--usedetig"; do
					for ALEN in " " "--noalen"; do
						for MAXL in 22 24; do
							ARGS="${WELL} --order 16 --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 4096 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --maxmsc 5 ${ZM} ${ALEN} ${USEIG} --expquad"
							echo $ARGS
							sbatch job_arg.slurm "${ARGS}"
							sleep 1
						done
					done
				done
			done
		done
	done
done
