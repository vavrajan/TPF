## Import global packages
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

data_name = 'hein-daily'

### Setting up directories
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, 'data', data_name)
fit_dir = os.path.join(source_dir, 'pf-fits')
data_dir = os.path.join(source_dir, 'clean')
save_dir = os.path.join(source_dir, 'fits')
fig_dir = os.path.join(source_dir, 'figs')
tab_dir = os.path.join(source_dir, 'tabs')

K = 25
name = "TPF_voc_combined"
time_periods = "sessions"
ending = "_K" + str(K) + "_" + time_periods
varfams = ["MVnormal", "normal"]
ars = ["AR", "RW"]
ars_nice = {"AR": r'$\delta_{kv} \in \mathbb{R}$', "RW": r'$\delta_{kv} = 1$'}
varfams_nice = {"MVnormal": r'$\phi_{hkv}^{\mathsf{var}}$ general', "normal": r'$\phi_{hkv}^{\mathsf{var}}$ diagonal'}

### Table
col_names = ["$\\delta_{kv}$", "$\\bm \\phi_{hkv}^\\covm$", "$\\mathsf{ELBO}", "Reconstr.",
             "\\mathsf{VAIC}", "\\mathsf{VBIC}", "sec/epoch", "sec/\\mathsf{ELBO}"]

with open(os.path.join(tab_dir, 'models_VIC.tex'), 'w') as file:
    file.write('\\begin{tabular}{ll|' + 'r' * 6 + '}\n')
    file.write('\\toprule\n')
    file.write('\\multicolumn{1}{c}{')
    file.write('} & \\multicolumn{1}{c}{'.join(col_names))
    file.write('}\\\\\n')
    file.write('\\midrule\n')
    for ar in ars:
        for varfam in varfams:
            epoch_data = pd.read_csv(os.path.join(save_dir, name + "_" + ar + "_" + varfam + ending, "epoch_data.csv"),
                                     index_col=False, usecols=["ELBO", "reconstruction", "VAIC", "VBIC"])
            time_data = pd.read_csv(os.path.join(save_dir, name + "_" + ar + "_" + varfam + ending, "epoch_data.csv"),
                                    index_col=False, usecols=["sec/epoch", "sec/ELBO"])
            last_epoch = epoch_data.iloc[-1].to_list()
            time_average = time_data.mean()
            if ar == "AR":
                file.write("$\\in \\R$")
            else:
                file.write("$= 1$")
            file.write(" & ")
            if varfam == "MVnormal":
                file.write("general")
            else:
                file.write("diagonal")
            file.write(" & ")
            file.write(" & ".join("{:.0f}".format(x) for x in last_epoch))
            file.write(" & ")
            file.write(" & ".join("{:.2f}".format(x) for x in time_average))
            file.write('\\\\\n')
    file.write('\\bottomrule\n')
    file.write('\\end{tabular}\n')

### Plot
epoch_data = {}
start_epoch = 10
end_epoch = 100
legend = []
for ar in ars:
    for varfam in varfams:
        ar_varfam = ar + "_" + varfam
        ar_varfam_nice = ars_nice[ar] + ", " + varfams_nice[varfam]
        legend.append(ar_varfam_nice)
        epoch_data[ar_varfam] = pd.read_csv(os.path.join(save_dir, name + "_" + ar_varfam + ending, "epoch_data.csv"),
                                            index_col=False)
        epoch_data[ar_varfam]["log_prior_plus_entropy"] = epoch_data[ar_varfam]["log_prior"] + epoch_data[ar_varfam]["entropy"]
        epochs = (epoch_data[ar_varfam]['epoch'] >= start_epoch) & (epoch_data[ar_varfam]['epoch'] <= end_epoch)
        epoch_data[ar_varfam] = epoch_data[ar_varfam][epochs]

legend_loc = {"reconstruction": "lower right", "ELBO": "lower right", "VAIC": "upper right", "VBIC": "upper right",
              "sec/epoch": "upper left", "sec/ELBO": "center", "log_prior_plus_entropy": "lower right",
              "effective_number_of_parameters": "upper right"}
ylabel = {"reconstruction": "Reconstruction", "ELBO": "ELBO", "VAIC": "VAIC", "VBIC": "VBIC",
          "sec/epoch": "Seconds / epoch", "sec/ELBO": "Seconds / ELBO evaluation",
          "log_prior_plus_entropy": "Log_prior + entropy",
          "effective_number_of_parameters": "Effective number of parameters"}

for y in ["reconstruction", "log_prior_plus_entropy", "effective_number_of_parameters",
          "ELBO", "VAIC", "VBIC", "sec/epoch", "sec/ELBO"]:
    for ar in ars:
        for varfam in varfams:
            ar_varfam = ar + "_" + varfam
            plt.plot(epoch_data[ar_varfam]['epoch'], epoch_data[ar_varfam][y])
    #if y in ["VAIC", "VBIC"]:
    #    plt.yscale("log")
    plt.ylabel(ylabel[y])
    plt.xlabel('Epoch')
    plt.legend(legend, loc=legend_loc[y], ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'models_comparison_' + y.replace("/", "_") + '.png'))
    plt.close()

