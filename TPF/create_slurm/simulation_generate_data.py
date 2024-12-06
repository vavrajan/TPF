import numpy as np
import os

# First set up directories on the cluster:
data_name = 'simulation'
project_dir = os.getcwd()
python_dir = os.path.join(project_dir, 'TPF', 'simulation')
slurm_dir = os.path.join(project_dir, 'slurm', data_name)
gen_slurm_dir = os.path.join(slurm_dir, 'generate')
out_dir = os.path.join(project_dir, 'out', data_name)
err_dir = os.path.join(project_dir, 'err', data_name)

if not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
if not os.path.exists(gen_slurm_dir):
    os.mkdir(gen_slurm_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)

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

for idelta in range(len(deltas)):
    for num_times in [10, 20]:
        name = data_name + '_delta_' + names_delta[idelta] + '_T_' + str(num_times)

        scenarios[name] = {"addendum": '',
                           "num_authors": 1000,
                           "num_topics": 6,
                           "num_words": 500,
                           "num_times": num_times,
                           "num_replications": 10,
                           "ar_kv_delta": deltas[idelta],
                           "ar_kv_prec": 10.0,
                           "ar_kv_mean_start": -3.0,
                           "vowel_word": 4.0,
                           "consonant_width": 0.15
                           }


### Creating slurm files and one file to trigger all the jobs.
with open(os.path.join(slurm_dir, 'generate_simulation_data.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    all_file.write('#SBATCH --partition=' + partition + '\n')
    all_file.write('\n')
    for name in scenarios:
        flags = '  --simulation_name='+name
        for key in scenarios[name]:
            flags = flags+'  --'+key+'='+str(scenarios[name][key])
        with open(os.path.join(gen_slurm_dir, name+'.slurm'), 'w') as file:
            file.write('#!/bin/bash\n')
            file.write('#SBATCH --job-name=gen_sim # short name for your job\n')
            file.write('#SBATCH --partition='+partition+'\n')

            # Other potential computational settings.
            # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
            # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
            # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
            # file.write('#SBATCH --mem=1G # total memory per node\n')
            file.write('#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task\n')
            # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
            # file.write('#SBATCH --mail-type=ALL\n')

            file.write('#SBATCH -o '+os.path.join(out_dir, '%x_'+name+'_%j_%N.out')+'      # save stdout to file.\n')
            file.write('#SBATCH -e '+os.path.join(err_dir, '%x_'+name+'_%j_%N.err')+'      # save stderr to file.\n')
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
            file.write('python '+os.path.join(python_dir, 'generate_counts.py')+flags+'\n')
        # Add a line for running the batch script to the overall slurm job.
        all_file.write('sbatch --dependency=singleton '+os.path.join(gen_slurm_dir, name+'.slurm'))
        all_file.write('\n')
