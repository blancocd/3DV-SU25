#!/bin/bash
#SBATCH --job-name=2dgs-tnt
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=05:00:00
#SBATCH --error=$3DV_DIR/Project/slurm/%J.err
#SBATCH --output=$3DV_DIR/Project/slurm/%J.out

# Copy preprocessed training TNT dataset which is at $3DV_DATADIR/TNT_GOF/TrainingSet
cp -R $3DV_DATADIR/TNT_GOF/TrainingSet/ /scratch/$SLURM_JOB_ID/TNT_GOF
echo "Finished copying preprocessed TNT dataset"

# Copy gt TNT dataset which is at $3DV_DATADIR/TNT
cp -R $3DV_DATADIR/TNT/ /scratch/$SLURM_JOB_ID/TNT
echo "Finished copying gt TNT dataset"

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
# conda activate surfel_splatting
conda activate tnt_eval
echo "Finished activating environment"
which python

cd $3DV_DIR/Project/Methods/2d-gaussian-splatting/
python scripts/tnt_eval.py  --TNT_data /scratch/$SLURM_JOB_ID/TNT_GOF \
                            --TNT_GT /scratch/$SLURM_JOB_ID/TNT \

# It's possible to avoid stages:                        
# --skip_training --skip_rendering --skip_evaluation
