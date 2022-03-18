from objective import get_outdir
import matplotlib.pyplot as plt
import numpy as np

well = False
sigma = 1e-3

datsave = []
labels = []
for i in range(0, 4):
    outdir = get_outdir(well, i)
    outname = "qsmeasures/" + outdir.replace("/", "_") + f"qsmeasures_sigma_{sigma}"
    m = np.loadtxt(outname + ".txt", skiprows=1, delimiter=',')
    plt.semilogy(m[:, 0], m[:, 1], "--", label=f"{i}")
    datsave.append(m[:, 0])
    datsave.append(m[:, 1])
    labels.append(f"s_det{18 + 2*i}")
    labels.append(f"bmn_det{18 + 2*i}")
np.savetxt(f"analysisresults/bmns_det_well_{well}.txt", np.asarray(datsave).T, delimiter=',', newline='\n', header=",".join(labels), comments='')
plt.legend()
plt.show()

datsave = []
labels = []
for i in range(0, 8):
    correctionlevel = 0
    outdir = get_outdir(well, i)
    dats = []
    for sampleidx in range(128):
        outname = "qsmeasures/" + outdir.replace("/", "_") + f"qsmeasures_sigma_{sigma}"
        if sampleidx is not None:
            outname += f"_sampleidx_{sampleidx}_correctionlevel_{correctionlevel}_ls_10_32"
        try:
            dat = np.loadtxt(outname + ".txt", skiprows=1, delimiter=',')
            dats.append(dat)
        except:
            print(f"Failed to read {outname}")
    m = np.mean(dats, axis=0)
    lt = "--" if i < 4 else "."
    plt.semilogy(m[:, 0], m[:, 1], lt, label=f"{i}")
    datsave.append(m[:, 0])
    datsave.append(m[:, 1])
    if i < 4:
        labels.append(f"s_det{18 + 2*i}")
        labels.append(f"bmn_det{18 + 2*i}")
    else:
        labels.append(f"s_stoch{18 + 2*(i-4)}")
        labels.append(f"bmn_stoch{18 + 2*(i-4)}")
np.savetxt(f"analysisresults/bmns_det_v_stoch_well_{well}.txt", np.asarray(datsave).T, delimiter=',', newline='\n', header=",".join(labels), comments='')

plt.legend()
plt.show()

datsave = []
labels = []
for i in [3]:
    outdir = get_outdir(well, i)
    for correctionlevel in range(4):
        dats = []
        for sampleidx in range(128):
            outname = "qsmeasures/" + outdir.replace("/", "_") + f"qsmeasures_sigma_{sigma}"
            if sampleidx is not None:
                outname += f"_sampleidx_{sampleidx}_correctionlevel_{correctionlevel}_ls_10_32"
            try:
                dat = np.loadtxt(outname + ".txt", skiprows=1, delimiter=',')
                dats.append(dat)
            except:
                print(f"Failed to read {outname}")
        m = np.mean(dats, axis=0)
        plt.semilogy(m[:, 0], m[:, 1], label=f"{i}_{correctionlevel}")
        datsave.append(m[:, 0])
        datsave.append(m[:, 1])
        labels.append(f"s_corrlev_{correctionlevel}")
        labels.append(f"bmn_corrlev_{correctionlevel}")
np.savetxt(f"analysisresults/bmns_correction_well_{well}_24.txt", np.asarray(datsave).T, delimiter=',', newline='\n', header=",".join(labels), comments='')
plt.legend()
plt.show()
