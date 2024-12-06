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
col_names = ["Model", "Reconstruction", "Log-prior", "Entropy", "ELBO", "VAIC", "VBIC"]

with open(os.path.join(tab_dir, 'table_ELBO.tex'), 'w') as file:
    file.write('\\begin{tabular}{l|' + 'r' * 6 + '}\n')
    file.write('\\toprule\n')
    file.write('\\multicolumn{1}{c}{')
    file.write('} & \\multicolumn{1}{c}{'.join(col_names))
    file.write('}\\\\\n')
    file.write('\\midrule\n')
    for ar in ars:
        for varfam in varfams:
            epoch_data = pd.read_csv(os.path.join(save_dir, name + "_" + ar + "_" + varfam + ending, "epoch_data.csv"),
                                     index_col=False,
                                     usecols=["reconstruction", "log_prior", "entropy", "ELBO", "VAIC", "VBIC"])
            last_epoch = epoch_data.iloc[-1].to_list()
            file.write(ar + ", " + varfam + " & ")
            file.write(" & ".join(str(x) for x in last_epoch))
            file.write('\\\\\n')
    file.write('\\bottomrule\n')
    file.write('\\end{tabular}\n')
