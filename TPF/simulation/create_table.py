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



cols = {
    "VAIC": "$\\mathsf{VAIC}/1000$",
    "VBIC": "$\\mathsf{VBIC}/1000$",
    "sec/epoch": "sec/epoch",
    "sec/ELBO": "sec/$\\mathsf{ELBO}$",
}

cols_digits = {
    "VAIC": 1,
    "VBIC": 1,
    "sec/epoch": 2,
    "sec/ELBO": 2,
}

cols_min_or_max = {
    "VAIC": "min",
    "VBIC": "min",
    "sec/epoch": "min",
    "sec/ELBO": "min",
}

cols_constant = {
    "VAIC": 0.001,
    "VBIC": 0.001,
    "sec/epoch": 1.0,
    "sec/ELBO": 1.0,
}

model_names2 = {"AR_MVnormal": 'A = TPF, $\\delta \\in \\R$, $\\bm\\phi_{hkv}^\\covm$ general',
                "AR_normal": 'B = TPF, $\\delta \\in \\R$, $\\bm\\phi_{hkv}^\\covm$ diagonal',
                "RW_MVnormal": 'C = TPF, $\\delta = 1$, $\\bm\\phi_{hkv}^\\covm$ general',
                "RW_normal": 'D = TPF, $\\delta = 1$, $\\bm\\phi_{hkv}^\\covm$ diagonal',
                "ART_MVnormal": 'E = TPF, $\\delta \\in [-1, 1]$, $\\bm\\phi_{hkv}^\\covm$ general',
                "ART_normal": 'F = TPF, $\\delta \\in [-1, 1]$, $\\bm\\phi_{hkv}^\\covm$ diagonal',
                "DPF_RW_normal": 'G = DPF, $\\delta = 1$, $\\bm\\phi_{hkv}^\\covm$ diagonal'}


model_names3 = {"AR_MVnormal": 'A & TPF & $\\R$ & general',
                "AR_normal": 'B & TPF & $\\R$ & diagonal',
                "RW_MVnormal": 'C & TPF & $1$ & general',
                "RW_normal": 'D & TPF & $1$ & diagonal',
                "ART_MVnormal": 'E & TPF & $[-1, 1]$ & general',
                "ART_normal": 'F & TPF & $[-1, 1]$ & diagonal',
                "DPF_RW_normal": 'G & DPF & $1$ & diagonal'}

path = os.path.join(data_dir, 'simulation', 'tabs', 'model_comparison_tall.tex')
save_data = pd.DataFrame()
with open(path, 'w') as file:
    # file.write('\\begin{tabular}{ccl|' + 'r'*len(cols) + '}\n')
    file.write('\\begin{tabular}{cc|cccc|' + 'r' * len(cols) + '}\n')
    file.write('\\toprule\n')
    # file.write('$T$ & $\\delta$ & \\multicolumn{1}{c}{Model} &')
    file.write('\\multirow{2}{*}{$T$} & \\multirow{2}{*}{$\\delta$} & \\multicolumn{4}{c|}{Model} & \\multirow{2}{*}{')
    file.write("} & \\multirow{2}{*}{".join(cols.values()))
    file.write('} \\\\\n')
    file.write(' & & & & $\\delta$ & $\\bm\\phi_{hkv}^\\covm$' + ' & ' * len(cols) + '\\\\\n')
    for T in [10, 20]:
        for delta in ["zero", "half", "one"]:
            param = 'delta_' + delta + '_T_' + str(T)
            subdata = data[data['param'] == param].reset_index(drop=True)
            print_data = pd.DataFrame()
            agg_data = pd.DataFrame()
            print_data['model'] = models
            agg_data['model'] = models
            for col in cols:
                agg_data[col] = subdata[col].groupby(subdata['model']).mean()[models].to_numpy()
                if cols_min_or_max[col] == "min":
                    index = np.argmin(agg_data[col])
                elif cols_min_or_max[col] == "max":
                    index = np.argmax(agg_data[col])
                pattern = '${0:,.' + str(cols_digits[col]) + 'f}$'
                print_data[col] = [pattern.format(x * cols_constant[col]).replace(",", "\\,") for x in agg_data[col]]
                print_data[col].iloc[index] = '\\contour{black}{' + print_data[col].iloc[index] + '}'

            file.write('\\midrule\n')
            for model in model_names:
                print_row = print_data[print_data['model'] == model]
                agg_row = agg_data[agg_data['model'] == model]
                # multirow for only the first model
                if model_names[model] == 'A':
                    file.write('\\multirow{' + str(len(model_names)) + '}{*}{' + str(T) + '} & ')
                    file.write('\\multirow{' + str(len(model_names)) + '}{*}{' + str(delta_values[delta]) + '} & ')
                else:
                    file.write(' & & ')
                file.write(model_names3[model] + ' & ')
                file.write(' & '.join(print_row[cols.keys()].iloc[0]))
                file.write(' \\\\\n')
                save_data_row = {"T": T, "delta": delta, "model": model}
                for col in cols:
                    save_data_row[col] = agg_row[col].iloc[0]
                save_data = save_data.append(save_data_row, ignore_index=True)


    file.write('\\bottomrule\n')
    file.write('\\end{tabular}\n')

save_data.to_csv(os.path.join(data_dir, "simulation", "tableVIC.csv"))
data.to_csv(os.path.join(data_dir, "simulation", "data_tableVIC.csv"))
