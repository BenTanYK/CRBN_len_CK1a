#!/bin/bash
#SBATCH -p RTX4060,RTX3080
#SBATCH -n 1
#SBATCH --gres=gpu:1 		

# Email notifications
#SBATCH --mail-user=s1934251@ed.ac.uk  # Your email address
#SBATCH --mail-type=END           # Types of notifications

# Initialize variables to store the numbers
RNUMBER=0
JNUMBER=0

# Use flags r and j to store the specified values
while getopts "r:j:" opt; do
  case $opt in
    r)
      RNUMBER=$OPTARG
      ;;
    j)
      JNUMBER=$OPTARG
      ;;
    \?)
      echo "Usage: $0 -r <number> -j <number>"
      exit 1
      ;;
  esac
done

# Shift off the processed options
shift $((OPTIND -1))

module load cuda

source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate openmm7.7

srun python run_window.py $RNUMBER $JNUMBER
