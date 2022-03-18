for MAXL in 18 20 22 24; do
    for WELL in "--well" " "; do
        for IG in 0 1 2 3 4 5 6 7; do
            for SIGMA in 1e-3; do
                for USEIG in " " "--usedetig"; do
                    for ALEN in " " "--noalen"; do
                        ARGS="${WELL} --order 16 --lengthbound ${MAXL} --fil 0 --ig ${IG} --nsamples 4096 --sigma ${SIGMA} --mindist 0.10 --maxkappa 5 --maxmsc 5 ${ALEN} ${USEIG} --expquad"
                        echo $ARGS
                        sbatch job_arg.slurm "${ARGS}"
                        sleep 1
                    done
                done
            done
        done
    done
done
