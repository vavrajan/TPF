## Import global packages
import os
import time

import numpy as np
import pandas as pd

### Setting up directories
project_dir = os.getcwd()
data_dir = os.path.join(project_dir, 'data')
K = 6

data = pd.DataFrame()
models = []
params = []
for delta in ["zero", "half", "one"]:
    for T in [10, 20]:
        param = 'delta_' + delta + '_T_' + str(T)
        data_name = 'simulation_' + param
        params.append(param)
        source_dir = os.path.join(data_dir, data_name)
        clean_dir = os.path.join(source_dir, 'clean')
        for ar in ["AR", "RW", "ART"]:
            for varfam in ["MVnormal", "normal"]:
                model = ar + '_' + varfam
                for rep in range(1):
                    chname = "TPF_" + model + "_K" + str(K) + "_" + str(rep)
                    fit_dir = os.path.join(source_dir, 'fits', chname)
                    edata = pd.read_csv(os.path.join(fit_dir,  'epoch_data.csv'))
                    row = edata.iloc[-1]
                    row['delta'] = delta
                    row['T'] = T
                    row['model'] = model
                    row['param'] = param
                    row['rep'] = rep
                    data = data.append(row)

for ar in ["AR", "RW", "ART"]:
    for varfam in ["MVnormal", "normal"]:
        model = ar + '_' + varfam
        models.append(model)

model_names = {"AR_MVnormal": 'A',
               "AR_normal": 'B',
               "RW_MVnormal": 'C',
               "RW_normal": 'D',
               "ART_MVnormal": 'E',
               "ART_normal": 'F'}
delta_values = {"zero": 0.0,
                "half": 0.5,
                "one": 1.0}

path = os.path.join(data_dir, 'simulation', 'tabs', 'model_comparison.tex')
with open(path, 'w') as file:
    file.write('\\begin{tabular}{cc|' + 'r'*len(models) + '|' + 'r'*len(models) + '}\n')
    file.write('\\toprule\n')
    file.write('\\multirow{2}{*}{$T$} & \\multirow{2}{*}{$\\delta$} & ')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{VAIC / 1000} &')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{VBIC / 1000} \\\\\n')
    file.write(' & & \\multicolumn{1}{c}{')
    file.write("} & \\multicolumn{1}{c}{".join([model_names[model] for model in models]))
    file.write('} & \\multicolumn{1}{c}{')
    file.write("} & \\multicolumn{1}{c}{".join([model_names[model] for model in models]))
    file.write('}\\\\\n')
    file.write('\\midrule\n')
    for T in [10, 20]:
        for delta in ["zero", "half", "one"]:
            param = 'delta_' + delta + '_T_' + str(T)
            subdata = data[data['param'] == param]
            file.write(str(T) + ' & ' + str(delta_values[delta]) + ' & ')

            VAIC = subdata['VAIC'].groupby(subdata['model']).mean().to_numpy()
            VAIC_nice = ['{0:.1f}'.format(x / 1000) for x in VAIC]
            file.write(" & ".join(VAIC_nice))
            file.write(" & ")

            VBIC = subdata['VBIC'].groupby(subdata['model']).mean().to_numpy()
            VBIC_nice = ['{0:.1f}'.format(x / 1000) for x in VBIC]
            file.write(" & ".join(VBIC_nice))
            file.write('\\\\\n')
    file.write('\\toprule\n')
    file.write('\\multirow{2}{*}{$T$} & \\multirow{2}{*}{$\\delta$} & ')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{sec/epoch} &')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{sec/ELBO} \\\\\n')
    file.write(' & & \\multicolumn{1}{c}{')
    file.write("} & \\multicolumn{1}{c}{".join([model_names[model] for model in models]))
    file.write('} & \\multicolumn{1}{c}{')
    file.write("} & \\multicolumn{1}{c}{".join([model_names[model] for model in models]))
    file.write('}\\\\\n')
    file.write('\\midrule\n')
    for T in [10, 20]:
        for delta in ["zero", "half", "one"]:
            param = 'delta_' + delta + '_T_' + str(T)
            subdata = data[data['param'] == param]
            file.write(str(T) + ' & ' + str(delta_values[delta]) + ' & ')

            sec1 = subdata['sec/epoch'].groupby(subdata['model']).mean().to_numpy()
            sec1_nice = ['{0:.2f}'.format(x) for x in sec1]
            file.write(" & ".join(sec1_nice))
            file.write(" & ")

            sec2 = subdata['sec/ELBO'].groupby(subdata['model']).mean().to_numpy()
            sec2_nice = ['{0:.2f}'.format(x) for x in sec2]
            file.write(" & ".join(sec2_nice))
            file.write('\\\\\n')
    file.write('\\bottomrule\n')
    file.write('\\end{tabular}\n')





