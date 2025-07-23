#!/bin/bash
#SBATCH --job-name=2dgs-dtu
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=01:00:00
#SBATCH --error=$3DV_DIR/Project/slurm/%J.err
#SBATCH --output=$3DV_DIR/Project/slurm/%J.out

# Copy preprocessed DTU dataset which is at $3DV_DATADIR/DTU_2DGS
cp -R $3DV_DATADIR/DTU_2DGS/ /scratch/$SLURM_JOB_ID/DTU_2DGS
echo "Finished copying preprocessed DTU dataset"

# Copy gt DTU dataset which is at $3DV_DATADIR/DTU
cp -R $3DV_DATADIR/DTU/ /scratch/$SLURM_JOB_ID/DTU
echo "Finished copying gt DTU dataset"

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
conda activate surfel_splatting
echo "Finished activating environment"
which python

cd $3DV_DIR/Project/Methods/2d-gaussian-splatting/
python scripts/dtu_eval.py  --dtu /scratch/$SLURM_JOB_ID/DTU_2DGS   \
                            --DTU_Official /scratch/$SLURM_JOB_ID/DTU \

# It's possible to avoid stages
# --skip_training --skip_rendering --skip_evaluation
