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
                for rep in range(10):
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

for delta in ["zero", "half", "one"]:
    for T in [10, 20]:
        param = 'delta_' + delta + '_T_' + str(T)
        data_name = 'simulation_' + param
        params.append(param)
        source_dir = os.path.join(data_dir, data_name)
        clean_dir = os.path.join(source_dir, 'clean')
        model = "RW_normal"
        for rep in range(10):
            chname = "DPF_" + model + "_K" + str(K) + "_" + str(rep)
            fit_dir = os.path.join(source_dir, 'fits', chname)
            edata = pd.read_csv(os.path.join(fit_dir,  'epoch_data.csv'))
            row = edata.iloc[-1]
            row['delta'] = delta
            row['T'] = T
            row['model'] = "DPF_" + model
            row['param'] = param
            row['rep'] = rep
            data = data.append(row)
models.append("DPF_RW_normal")

model_names = {"AR_MVnormal": 'A',
               "AR_normal": 'B',
               "RW_MVnormal": 'C',
               "RW_normal": 'D',
               "ART_MVnormal": 'E',
               "ART_normal": 'F',
               "DPF_RW_normal": "G"}
delta_values = {"zero": 0.0,
                "half": 0.5,
                "one": 1.0}

path = os.path.join(data_dir, 'simulation', 'tabs', 'model_comparison.tex')
with open(path, 'w') as file:
    file.write('\\begin{tabular}{cc|' + 'r'*len(models) + '|' + 'r'*len(models) + '}\n')
    file.write('\\toprule\n')
    file.write('\\multirow{2}{*}{$T$} & \\multirow{2}{*}{$\\delta$} & ')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{$\\mathsf{VAIC} / 1000$} &')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{$\\mathsf{VBIC} / 1000$} \\\\\n')
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

            VAIC = subdata['VAIC'].groupby(subdata['model']).mean()
            VAIC = VAIC[models].to_numpy()
            VAIC_nice = ['${0:,.1f}$'.format(x / 1000).replace(",", "\\,") for x in VAIC]
            Vmin = np.argmin(VAIC)
            VAIC_nice[Vmin] = '\\contour{black}{' + VAIC_nice[Vmin] + '}'
            file.write(" & ".join(VAIC_nice))
            file.write(" & ")

            VBIC = subdata['VBIC'].groupby(subdata['model']).mean()
            VBIC = VBIC[models].to_numpy()
            VBIC_nice = ['${0:,.1f}$'.format(x / 1000).replace(",", "\\,") for x in VBIC]
            Vmin = np.argmin(VBIC)
            VBIC_nice[Vmin] = '\\contour{black}{' + VBIC_nice[Vmin] + '}'
            file.write(" & ".join(VBIC_nice))
            file.write('\\\\\n')
    file.write('\\toprule\n')
    file.write('\\multirow{2}{*}{$T$} & \\multirow{2}{*}{$\\delta$} & ')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{sec / epoch} &')
    file.write('\\multicolumn{' + str(len(models)) + '}{c}{sec / $\\mathsf{ELBO}$} \\\\\n')
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

            sec1 = subdata['sec/epoch'].groupby(subdata['model']).mean()
            sec1 = sec1[models].to_numpy()
            sec1_nice = ['${0:,.2f}$'.format(x).replace(",", "\\,") for x in sec1]
            smin = np.argmin(sec1)
            sec1_nice[smin] = '\\contour{black}{' + sec1_nice[smin] + '}'
            file.write(" & ".join(sec1_nice))
            file.write(" & ")

            sec2 = subdata['sec/ELBO'].groupby(subdata['model']).mean()
            sec2 = sec2[models].to_numpy()
            sec2_nice = ['${0:,.2f}$'.format(x).replace(",", "\\,") for x in sec2]
            smin = np.argmin(sec2)
            sec2_nice[smin] = '\\contour{black}{' + sec2_nice[smin] + '}'
            file.write(" & ".join(sec2_nice))
            file.write('\\\\\n')
    file.write('\\bottomrule\n')
    file.write('\\end{tabular}\n')





