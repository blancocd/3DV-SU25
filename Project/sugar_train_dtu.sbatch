#!/bin/bash
#SBATCH --job-name=sugar-train-dtu
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=08:00:00
#SBATCH --error=$3DV_DIR/Project/slurm/%J.err
#SBATCH --output=$3DV_DIR/Project/slurm/%J.out

# Copy preprocessed DTU dataset which is at $3DV_DATADIR/DTU_2DGS
cp -R $3DV_DATADIR/DTU_2DGS/ /scratch/$SLURM_JOB_ID/DTU_2DGS
echo "Finished copying preprocessed DTU dataset"

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
conda activate sugar
echo "Finished activating environment"
which python

cd $3DV_DIR/Project/Methods/SuGaR/
python train_dtu_w_2dgs_struct.py   --dtu /scratch/$SLURM_JOB_ID/DTU_2DGS
