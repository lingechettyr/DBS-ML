#!/bin/bash
#SBATCH --job-name=ANNRP_hparam_search        # Job name
#SBATCH --mail-type=FAIL,END             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jgolabek1@ufl.edu    # Where to send mail
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --mem-per-cpu=7000mb              # Job memory request
#SBATCH --time=20:00:00                  # Time limit hrs:min:sec
#SBATCH --output=logs/hparam_search.%A_%a.log   # Standard output and error log
#SBATCH --array=1-500%40
pwd; hostname; date

module load tensorflow/2.4.1
module list
python --version

param_csv=$1
array_id=$SLURM_ARRAY_TASK_ID
offset=$2
index=$((SLURM_ARRAY_TASK_ID + offset))

python hparam_search_multi.py $param_csv $index