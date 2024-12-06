import os

data_name = 'hein-daily'
addendum = '_voc_combined'


### First set up directories on the cluster:
project_dir = os.getcwd()
slurm_dir = os.path.join(project_dir, 'slurm', data_name)
pf_dir = os.path.join(slurm_dir, 'PF')
out_dir = os.path.join(project_dir, 'out', data_name)
err_dir = os.path.join(project_dir, 'err', data_name)
python_dir = os.path.join(project_dir, 'TPF', 'code')

if not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
if not os.path.exists(pf_dir):
    os.mkdir(pf_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)

# For now just use the environment for testing.
partition = 'gpu'

# Different number of topics
Ks = [25, 50]

# Different choices of times
time_periods = ['sessions', 'years', 'test']

### Creating slurm files and one file to trigger all the jobs.
with open(os.path.join(slurm_dir, 'run_all_PoisFact' + addendum + '.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    all_file.write('#SBATCH --partition=' + partition + '\n')
    all_file.write('\n')
    for K in Ks:
        for time_period in time_periods:
            flags = '  --data='+data_name+'  --num_topics='+str(K)+'  --addendum=' + addendum + '_' + time_period
            with open(os.path.join(pf_dir, 'PoisFact' + addendum + '_'+str(K)+'_'+time_period+'.slurm'), 'w') as file:
                file.write('#!/bin/bash\n')
                file.write('#SBATCH --job-name=PF_'+data_name+' # short name for your job\n')
                file.write('#SBATCH --partition='+partition+'\n')

                # Other potential computational settings.
                # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
                # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
                # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
                # file.write('#SBATCH --mem=1G # total memory per node\n')
                file.write('#SBATCH --mem-per-cpu=512000 # in megabytes, default is 4GB per task\n')
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
                file.write('conda activate env_TBIP\n')
                file.write('\n')
                file.write('python '+os.path.join(python_dir, 'poisson_factorization.py')+flags+'\n')
            # Add a line for running the batch script to the overall slurm job.
            all_file.write('sbatch --dependency=singleton ' + os.path.join(
                pf_dir, 'PoisFact' + addendum + '_'+str(K)+'_'+time_period+'.slurm'))
            all_file.write('\n')