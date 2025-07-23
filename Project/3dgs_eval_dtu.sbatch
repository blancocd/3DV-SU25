#!/bin/bash
#SBATCH --job-name=3dgs-eval-dtu
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=05:30:00
#SBATCH --error=$3DV_DIR/Project/slurm/%J.err
#SBATCH --output=$3DV_DIR/Project/slurm/%J.out

# Activate conda and the 2dgs environment which has libraries for evaluation code
source ~/miniconda3/bin/activate
conda activate surfel_splatting
echo "Finished activating environment"
which python

cd $3DV_DIR/Project/Methods/SuGaR/
python eval_3dgs_dtu_w_2dgs_struct.py   --dtu $3DV_DATADIR/DTU_2DGS \
                                        --DTU_Official $3DV_DATADIR/DTU
