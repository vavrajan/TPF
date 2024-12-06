import numpy as np
import os

# First set up directories on the cluster:
data_name = 'simulation'
project_dir = os.getcwd()
python_dir = os.path.join(project_dir, 'python', 'DPF', 'analysis')

# For now just use the environment for testing.
# partition = 'gpu-test'
partition = 'gpu'

### A dictionary of scenarios to be explored
# Default values correspond with the classical TBIP model (with topic-specific locations) with gamma and CAVI updates.
# List only the FLAGS that you want to be changed.

# First scenario
scenarios = {}
names_delta = ["zero", "half", "one"]
deltas = [0.0, 0.5, 1.0]
K = 6

with open(os.path.join(project_dir, 'slurm', data_name, 'run_all_DPF.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    all_file.write('#SBATCH --partition=' + partition + '\n')
    all_file.write('\n')
    for idelta in range(len(deltas)):
        for num_times in [10, 20]:
            name = data_name + '_delta_' + names_delta[idelta] + '_T_' + str(num_times)
            slurm_dir = os.path.join(project_dir, 'slurm', name)
            dpf_slurm_dir = os.path.join(slurm_dir, 'DPF')
            out_dir = os.path.join(project_dir, 'out', name)
            err_dir = os.path.join(project_dir, 'err', name)

            if not os.path.exists(slurm_dir):
                os.mkdir(slurm_dir)
            if not os.path.exists(dpf_slurm_dir):
                os.mkdir(dpf_slurm_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            if not os.path.exists(err_dir):
                os.mkdir(err_dir)

            source_dir = os.path.join(project_dir, 'data', name)
            pffit_dir = os.path.join(source_dir, 'pf-fits')
            fit_dir = os.path.join(source_dir, 'fits')
            fig_dir = os.path.join(source_dir, 'figs')
            tab_dir = os.path.join(source_dir, 'tabs')
            txt_dir = os.path.join(source_dir, 'txts')

            if not os.path.exists(pffit_dir):
                os.mkdir(pffit_dir)
            if not os.path.exists(fit_dir):
                os.mkdir(fit_dir)
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)
            if not os.path.exists(tab_dir):
                os.mkdir(tab_dir)
            if not os.path.exists(txt_dir):
                os.mkdir(txt_dir)

            scenarios = {}
            for delta_prior in ["RW"]: # in original DPF paper always RW
                for varfam in ["normal"]: # in original DPF paper always mean-field
                    for rep in range(10):
                        chname = "DPF_" + delta_prior + '_' + varfam + "_K" + str(K) + "_" + str(rep)
                        scenarios[chname] = {"checkpoint_name": chname,
                                             "addendum": "_" + str(rep),
                                             "num_topics": K,
                                             "num_epochs": 61,
                                             "time_periods": "",
                                             "pre_initialize_parameters": "sim_true",
                                             "save_every": 5,
                                             "computeIC_every": 10,
                                             "batch_size": 512,
                                             "exact_entropy": True,
                                             "exact_log_prior": True,
                                             "exact_reconstruction": True,
                                             "delta": delta_prior,
                                             "varfam_ar_ak": varfam,
                                             "varfam_ar_kv": varfam,
                                             "ar_ak_mean_scl": 100,
                                             "ar_kv_mean_scl": 100,
                                             "ar_ak_prec_shp": 0.3,
                                             "ar_ak_prec_rte": 0.3,
                                             "ar_kv_prec_shp": 0.3,
                                             "ar_kv_prec_rte": 0.3,
                                             "ar_ak_delta_loc": 0.5,
                                             "ar_ak_delta_scl": 1.0,
                                             "ar_kv_delta_loc": 0.5,
                                             "ar_kv_delta_scl": 1.0,
                                             }


            ### Creating slurm files and one file to trigger all the jobs.
            with open(os.path.join(slurm_dir, 'run_all_DPF.slurm'), 'w') as all_name_file:
                all_name_file.write('#! /bin/bash\n\n')
                all_name_file.write('#SBATCH --partition=' + partition + '\n')
                all_name_file.write('\n')
                for chname in scenarios:
                    flags = '  --data_name='+name
                    for key in scenarios[chname]:
                        flags = flags+'  --'+key+'='+str(scenarios[chname][key])
                    with open(os.path.join(dpf_slurm_dir, chname+'.slurm'), 'w') as file:
                        file.write('#!/bin/bash\n')
                        file.write('#SBATCH --job-name=DPF_'+data_name+' # short name for your job\n')
                        file.write('#SBATCH --partition='+partition+'\n')

                        # Other potential computational settings.
                        # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
                        # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
                        # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
                        # file.write('#SBATCH --mem=1G # total memory per node\n')
                        file.write('#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task\n')
                        # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
                        # file.write('#SBATCH --mail-type=ALL\n')

                        file.write('#SBATCH -o '+os.path.join(out_dir, '%x_'+chname+'_%j.out')+'      # save stdout to file. '
                                                         'The filename is defined through filename pattern\n')
                        file.write('#SBATCH -e '+os.path.join(err_dir, '%x_'+chname+'_%j.err')+'      # save stderr to file. '
                                                         'The filename is defined through filename pattern\n')
                        file.write('#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)\n')
                        file.write('\n')
                        file.write('. /opt/apps/2023-04-11_lmod.bash\n')
                        file.write('ml purge\n')
                        file.write('ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn\n')
                        file.write('\n')
                        file.write('cd /home/jvavra/TPF/\n')
                        # file.write('conda activate env_TBIP\n')
                        file.write('conda activate tf_TBIP\n')
                        file.write('\n')
                        file.write('python '+os.path.join(python_dir, 'dpf_cluster.py')+flags+'\n')
                    # Add a line for running the batch script to the overall slurm job.
                    all_name_file.write('sbatch --dependency=singleton '+os.path.join(dpf_slurm_dir, chname+'.slurm'))
                    all_name_file.write('\n')
                    # Add a line for running the batch script to the overall slurm job.
                    all_file.write('sbatch --dependency=singleton ' + os.path.join(dpf_slurm_dir, chname+'.slurm'))
                    all_file.write('\n')

