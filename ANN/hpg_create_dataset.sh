#!/bin/bash
#SBATCH --job-name=ANNRP_create_dataset         # Job name
#SBATCH --mail-type=FAIL,END                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jgolabek1@ufl.edu           # Where to send mail
#SBATCH --ntasks=1                              # Run on a single CPU
#SBATCH --mem-per-cpu=30000mb                    # Job memory request
#SBATCH --time=02:00:00                         # Time limit hrs:min:sec
#SBATCH --output=logs/create_dataset.%A_%a.log   # Standard output and error log
#SBATCH --array=1-1%10

pwd; hostname; date

module load python/3.8
module list
python --version

data_dir=$1
output_path=$2
array_id=$SLURM_ARRAY_TASK_ID

python create_dataset.py $data_dir $output_path $array_id
