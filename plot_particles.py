import numpy as np
import matplotlib.pyplot as plt

datafiles = [
    "losses/output_well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy",
    "losses/output_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_alstart_0_expquad_losses_sigma_0.0_sampleidx_None_correctionlevel_0_spawnidx_25_n_60_seed_1.txt.npy"
]
legends = [
    "QA_18_Det_Exact",
    "QA_20_Det_Exact",
    "QA_22_Det_Exact",
    "QA_24_Det_Exact",
    "QAWell_18_Det_Exact",
    "QAWell_20_Det_Exact",
    "QAWell_22_Det_Exact",
    "QAWell_24_Det_Exact",
]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ts = np.logspace(-5, np.log10(0.1999), num=200)

all_data = [ts]
all_labels = ["ts"]

for i, (fil, lab) in enumerate(zip(datafiles, legends)):
    try:
        res_tys = np.load(fil)
    except:
        continue
    finalts = res_tys[:, 1, 0]
    # ts, fractions = zip(*[(t, i / len(finalts)) for i, t in enumerate(sorted(finalts)) if t < 0.1999999])
    fractions = [np.sum(finalts < t)/len(finalts) for t in ts]
    if "Pert" in lab:
        if "Stoch" in lab:
            plt.semilogx(ts, fractions, ":", label=lab, color=colors[i])
        else:
            plt.semilogx(ts, fractions, "--", label=lab, color=colors[i])
    else:
        plt.semilogx(ts, fractions, label=lab, color=colors[i])
    print(lab, f"{100*fractions[-1]:.2f}, len(finalts) =", len(finalts))
    all_data.append(100*np.asarray(fractions))
    all_labels.append(lab)

################################################################################
################################################################################
################################################################################


def plot(datafun, label, style, coloridx):
    datafiles = [datafun(i) for i in range(128)]

    fractions_all = []
    for fil in datafiles:
        try:
            res_tys = np.load(fil)
        except Exception as ex:
            # print(ex)
            print("skip", fil)
            continue
        finalts = res_tys[:, 1, 0]
        fractions = [np.sum(finalts < t)/len(finalts) for t in ts]
        fractions_all.append(fractions)
        # plt.semilogx(ts, fractions, ":", linewidth=0.5, alpha=0.5)

    fractions_mean = np.mean(fractions_all, axis=0)
    fractions_std = np.std(fractions_all, axis=0)

    plt.semilogx(ts, fractions_mean, style, label=label, linewidth=2, color=colors[coloridx])
    plt.fill_between(ts, fractions_mean-fractions_std/np.sqrt(len(fractions_all)), fractions_mean+fractions_std/np.sqrt(len(fractions_all)), alpha=0.2, color=colors[coloridx])
    # plt.fill_between(ts, fractions_mean-fractions_std, fractions_mean+fractions_std, alpha=0.1, color=colors[coloridx])
    print(label, f"{100*fractions_mean[-1]:.2f}")
    all_data.append(100*fractions_mean)
    all_labels.append(label)



# # import sys; sys.exit()
def det(i):
    return f"losses/output_well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det, "QADet24", "-", 3)

def det1(i):
    return f"losses/output_well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_1_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det1, "QADet24_1", "-.", 4)

def det2(i):
    return f"losses/output_well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_2_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det2, "QADet24_2", "-.", 5)

def det3(i):
    return f"losses/output_well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_3_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det3, "QADet24_3", "-.", 6)

def stoch(i):
    return f"losses/output_well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch, "QAStoch24", ":", 3)

def det22(i):
    return f"losses/output_well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det22, "QADet22", "-", 4)

def stoch22(i):
    return f"losses/output_well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_0_order_16_alstart_0_expquad_samples_4096_sigma_0.001_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch22, "QAStoch22", ":", 4)

def det20(i):
    return f"losses/output_well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det20, "QADet20", "-", 5)

def stoch20(i):
    return f"losses/output_well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_samples_4096_sigma_0.001_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch20, "QAStoch20", ":", 5)

def det18(i):
    return f"losses/output_well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det18, "QADet18", "-", 6)

def stoch18(i):
    return f"losses/output_well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch18, "QAStoch18", ":", 6)




def det(i):
    return f"losses/output_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det, "QAwDet24", "-", 3)

def det1(i):
    return f"losses/output_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_1_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det1, "QAwDet24_1", "-", 4)

def det2(i):
    return f"losses/output_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_2_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det2, "QAwDet24_2", "-", 5)

def det3(i):
    return f"losses/output_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_3_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det3, "QAwDet24_3", "-", 6)

def stoch(i):
    return f"losses/output_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch, "QAwStoch24", ":", 3)

def det22(i):
    return f"losses/output_well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det22, "QAwDet22", "-", 4)

def stoch22(i):
    return f"losses/output_well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch22, "QAwStoch22", ":", 4)

def det20(i):
    return f"losses/output_well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_3_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det20, "QAwDet20", "-", 5)

def stoch20(i):
    return f"losses/output_well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_alstart_0_expquad_samples_4096_sigma_0.001_usedetig_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch20, "QAwStoch20", ":", 5)

def det18(i):
    return f"losses/output_well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_1_order_16_alstart_0_expquad_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(det18, "QAwDet18", "-", 6)

def stoch18(i):
    return f"losses/output_well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_alstart_0_expquad_samples_4096_sigma_0.001_losses_sigma_0.001_sampleidx_{i}_correctionlevel_0_spawnidx_25_n_60_seed_{i}.txt.npy"
plot(stoch18, "QAwStoch18", ":", 6)

np.savetxt("analysisresults/alpha.txt", np.asarray(all_data).T, delimiter=",", header=",".join(all_labels), comments="")

# import sys; sys.exit()

# import sys; sys.exit()

# import pandas as pd
# M = pd.read_csv("~/Documents/Uni/PostDoc/papers/CoilsForPreciseQS/plots/alpha_det.txt")
# plt.plot(M["QA18_x"], M["QA18_y"], ":", label="Simple 18")
# plt.plot(M["QA20_x"], M["QA20_y"], ":", label="Simple 20")
# plt.plot(M["QA22_x"], M["QA22_y"], ":", label="Simple 22")
# plt.plot(M["QA24_x"], M["QA24_y"], ":", label="Simple 24")
# print(f"{np.max(M['QA18_y'])*100:.2f}")
# print(f"{np.max(M['QA20_y'])*100:.2f}")
# print(f"{np.max(M['QA22_y'])*100:.2f}")
# print(f"{np.max(M['QA24_y'])*100:.2f}")

# plt.plot(M["QAw18_x"], M["QAw18_y"], ":", label="Simple 18")
# plt.plot(M["QAw20_x"], M["QAw20_y"], ":", label="Simple 20")
# plt.plot(M["QAw22_x"], M["QAw22_y"], ":", label="Simple 22")
# plt.plot(M["QAw24_x"], M["QAw24_y"], ":", label="Simple 24")
# print(f"{np.max(M['QAw18_y'])*100:.2f}")
# print(f"{np.max(M['QAw20_y'])*100:.2f}")
# print(f"{np.max(M['QAw22_y'])*100:.2f}")
# print(f"{np.max(M['QAw24_y'])*100:.2f}")
# import IPython; IPython.embed()
# import sys; sys.exit()


################################################################################
################################################################################
################################################################################

plt.gca().set_prop_cycle(None)

datafiles = [
    '20211030-01-010-for_Florian_smallStuff/20211030-01-003_simple_s0.3/well1_length18.0_ig6_qfm_None_nfp2_stellsym_aScaling/confined_fraction.dat',
    '20211030-01-010-for_Florian_smallStuff/20211030-01-003_simple_s0.3/well1_length20.0_ig4_qfm_None_nfp2_stellsym_aScaling/confined_fraction.dat',
    '20211030-01-010-for_Florian_smallStuff/20211030-01-003_simple_s0.3/well1_length22.0_ig7_qfm_None_nfp2_stellsym_aScaling/confined_fraction.dat',
    '20211030-01-010-for_Florian_smallStuff/20211030-01-003_simple_s0.3/well1_length24.0_ig5_qfm_None_nfp2_stellsym_aScaling/confined_fraction.dat',
]

legends = [
    "SIMPLE QA+Well[18]",
    "SIMPLE QA+Well[20]",
    "SIMPLE QA+Well[22]",
    "SIMPLE QA+Well[24]",
]
for fil, lab in zip(datafiles, legends):
    continue
    data = np.loadtxt(fil)
    t = data[1:, 0]
    confined_frac_passing = data[1:, 1]
    confined_frac_trapped = data[1:, 2]
    confined_frac = confined_frac_passing + confined_frac_trapped
    lost_frac = 1 - confined_frac
    plt.semilogx(t, lost_frac, ":", label=lab)

plt.title("Sigma=0.001")
plt.legend(loc="upper left")
plt.grid()
plt.xlim((1e-3, 0.2))
# plt.savefig("alpha_0p001.png", bbox_inches='tight', dpi=400)
# plt.xlim((1e-2, 0.2))
# plt.ylim((0, 0.12))
# plt.savefig("alpha_0p001_zoom.png", bbox_inches='tight', dpi=400)
plt.show()
