#!/bin/bash
#SBATCH --job-name=ANNRP_dti_axon_sim        # Job name
#SBATCH --mail-type=FAIL,END             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jgolabek1@ufl.edu    # Where to send mail
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --mem-per-cpu=5000mb              # Job memory request
#SBATCH --time=20:00:00                  # Time limit hrs:min:sec
#SBATCH --output=logs/axon_dti_sim.%A_%a.log   # Standard output and error log
#SBATCH --array=1-38%40
pwd; hostname; date

module load neuron/7.7.2
module list
python --version

comsol_file=$1
tract_file=$2
output_dir=$3

param_csv=$4
array_id=$SLURM_ARRAY_TASK_ID
offset=$5
index=$((SLURM_ARRAY_TASK_ID + offset))

python dti_axon_multi.py $comsol_file $tract_file $output_dir $param_csv $index