#!/bin/bash
#SBATCH -J train_eth_srt
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 120:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=0-3

cd $PROJECT_ROOT
eval "$(micromamba shell hook --shell bash)"
micromamba activate $HOME/micromamba/envs/srt

MOLECULE="uracil"

# Define your experiment variants
experiments=("${MOLECULE}-0.1" "${MOLECULE}-1" "${MOLECULE}-10" "${MOLECULE}-100")

# Select based on SLURM_ARRAY_TASK_ID
EXPERIMENT=${experiments[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="${MOLECULE}-srt-${EXPERIMENT##*-}"  # extracts 0.1, 1, 10, 100
RUN_DIR='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}
CKPT_DIR=${RUN_DIR}/checkpoints

echo "Project root: $PROJECT_ROOT"
echo "Running experiment: $EXPERIMENT"
echo "Run name: $RUN_NAME"
echo "Run dir: $RUN_DIR"
echo "Checkpoint dir: $CKPT_DIR"

# For some reason, need to explicity pass cpus-per-task into srun
srun --cpus-per-task=$SLURM_CPUS_PER_TASK  $HOME/micromamba/envs/srt/bin/python -u src/train.py \
  experiment=train/baseline/${EXPERIMENT} \
  hydra.run.dir=${RUN_DIR} \
  ckpt_dir=${CKPT_DIR} \
  logger.id=${RUN_NAME} \
  logger.name=${RUN_NAME}
