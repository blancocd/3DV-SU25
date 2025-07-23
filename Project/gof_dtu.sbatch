#!/bin/bash
#SBATCH --job-name=gof-dtu
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=02:30:00
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
conda activate gof
echo "Finished activating environment"
which python

cd $3DV_DIR/Project/Methods/gaussian-opacity-fields/
python scripts/run_dtu_w_2dgs_struct.py --dtu /scratch/$SLURM_JOB_ID/DTU_2DGS   \
                                        --DTU_Official /scratch/$SLURM_JOB_ID/DTU \
                                        
# It's possible to avoid stages:
# --skip_training --skip_meshing --skip_tsdf_meshing --skip_evaluation --skip_evaluation_tsdf --skip_evaluation_gs_alignment --skip_evaluation_tsdf_gs_alignment
