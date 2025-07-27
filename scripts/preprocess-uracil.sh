#!/bin/bash
#SBATCH -J preprocess-uracil                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=128G                     # server memory requested (per node)
#SBATCH -t 24:00:00                  # Time limit (hh:mm:ss)
#SBATCH --account=aip-necludov               
#SBATCH --ntasks-per-node=4
#SBATCH -c 2
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@90

cd ~/self-refining-dft
source ~/envs/srt/bin/activate

MOLECULE="uracil"
BASIS_NAME="sto-3g"

python scripts/preprocess_coefs.py \
  experiment=preprocess \
  data.dataset.dataset_name=${MOLECULE} \
  basis_name=${BASIS_NAME} \