findbestdet:
	for len in {18,20,22,24}; do \
		python3 eval_find_best.py         --nsamples 0 --sigma 1e-3 --lengthbound $${len} --expquad --maxmsc 5 --order 16; \
		done

findbestwelldet:
	for len in {18,20,22,24}; do \
		python3 eval_find_best.py --well  --nsamples 0 --sigma 1e-3 --lengthbound $${len} --expquad --maxmsc 5 --order 16; \
		done

findbeststoch:
	for len in {18,20,22,24}; do python3 eval_find_best.py         --nsamples 4096 --sigma 1e-3 --lengthbound $${len} --expquad --maxmsc 5 --order 16; done
	for len in {18,20,22,24}; do python3 eval_find_best.py         --nsamples 4096 --sigma 1e-3 --lengthbound $${len} --expquad --maxmsc 5 --order 16 --usedetig; done

findbestwellstoch:
	for len in {18,20,22,24}; do python3 eval_find_best.py --well  --nsamples 4096 --sigma 1e-3 --lengthbound $${len} --expquad --maxmsc 5 --order 16; done
	for len in {18,20,22,24}; do python3 eval_find_best.py --well  --nsamples 4096 --sigma 1e-3 --lengthbound $${len} --expquad --maxmsc 5 --order 16 --usedetig; done

eval_geo: # evaluate curvature, distance, and write the coil shapes and currents to txt files
	python3 eval_geo.py
	python3 eval_geo.py --well

eval_vis: # create paraview files of coils and surfaces
	python3 eval_vis.py
	python3 eval_vis.py --well

eval_flux:
	python3 eval_flux_error_impact.py --correctionlevel 0 --stoch
	python3 eval_flux_error_impact.py --correctionlevel 1
	python3 eval_flux_error_impact.py --correctionlevel 2
	python3 eval_flux_error_impact.py --correctionlevel 3
	python3 eval_flux_error_impact.py --well --correctionlevel 0 --stoch
	python3 eval_flux_error_impact.py --well --correctionlevel 1
	python3 eval_flux_error_impact.py --well --correctionlevel 2
	python3 eval_flux_error_impact.py --well --correctionlevel 3

