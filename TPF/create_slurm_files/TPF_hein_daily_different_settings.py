import numpy as np
import os

# Data name
data_name = 'hein-daily'
addendum = '_voc_combined'

### First set up directories on the cluster:
project_dir = os.getcwd()
slurm_dir = os.path.join(project_dir, 'slurm', data_name)
tpf_slurm_dir = os.path.join(slurm_dir, 'TPF')
out_dir = os.path.join(project_dir, 'out', data_name)
err_dir = os.path.join(project_dir, 'err', data_name)
python_dir = os.path.join(project_dir, 'TPF', 'analysis')

if not os.path.exists(tpf_slurm_dir):
    os.mkdir(tpf_slurm_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)

# For now just use the environment for testing.
# partition = 'gpu-test'
partition = 'gpu'

# Number of epochs to do:
num_epochs = 101

# Different number of topics
Ks = [25]

# Different choices of times
time_periods = ['sessions']

# Different choices for theta prior
thetas = ["Gdrte"]

# Different precision tau shape
tau_shapes = [0.3]

# Prior scale for the mean of the AR sequence
ar_kv_mean_scl = 100

# How often to compute ELBO from the whole dataset (not just a batch)
computeIC_every = 5

### A dictionary of scenarios to be explored
# Default values correspond with the classical TBIP model (with topic-specific locations) with gamma and CAVI updates.
# List only the FLAGS that you want to be changed.

# First scenario
scenarios = {}

K = Ks[0]
time_period = time_periods[0]
theta = thetas[0]
tau_shape = tau_shapes[0]

for delta in ["AR", "ART", "RW"]:
    for varfam in ["normal", "MVnormal"]:
        name = "TPF" + addendum + '_' + delta + '_' + varfam + "_K" + str(K) + '_' + time_period
        scenarios[name] = {"checkpoint_name": name,
                           "addendum": addendum,
                           "num_topics": K,
                           "time_periods": time_period,
                           "pre_initialize_parameters": "PF",
                           "save_every": 1, "computeIC_every": computeIC_every,
                           "batch_size": 512,
                           "exact_entropy": True, "exact_log_prior": True, "exact_reconstruction": True,
                           "theta": theta, "delta": delta,
                           "varfam_ar_kv": varfam,
                           "ar_kv_mean_scl": ar_kv_mean_scl,
                           "ar_kv_prec_shp": tau_shape, "ar_kv_prec_rte": tau_shape,
                           }


### Creating slurm files and one file to trigger all the jobs.
with open(os.path.join(slurm_dir, 'run_all_TPF_different_settings_jobs' + addendum + '.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    all_file.write('#SBATCH --partition=' + partition + '\n')
    all_file.write('\n')
    for name in scenarios:
        flags = '  --num_epochs='+str(num_epochs)
        flags = flags + '  --data_name='+data_name
        for key in scenarios[name]:
            flags = flags+'  --'+key+'='+str(scenarios[name][key])
        with open(os.path.join(tpf_slurm_dir, name+'.slurm'), 'w') as file:
            file.write('#!/bin/bash\n')
            file.write('#SBATCH --job-name=TPF_'+data_name+' # short name for your job\n')
            file.write('#SBATCH --partition='+partition+'\n')

            # Other potential computational settings.
            # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
            # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
            # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
            # file.write('#SBATCH --mem=1G # total memory per node\n')
            file.write('#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task\n')
            # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
            # file.write('#SBATCH --mail-type=ALL\n')

            file.write('#SBATCH -o '+os.path.join(out_dir, '%x_%j_%N.out')+'      # save stdout to file.\n')
            file.write('#SBATCH -e '+os.path.join(err_dir, '%x_%j_%N.err')+'      # save stderr to file.\n')
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
            file.write('python '+os.path.join(python_dir, 'tpf_cluster.py')+flags+'\n')
        # Add a line for running the batch script to the overall slurm job.
        all_file.write('sbatch --dependency=singleton '+os.path.join(tpf_slurm_dir, name+'.slurm'))
        all_file.write('\n')
