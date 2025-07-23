#!/bin/bash
#SBATCH --job-name=sugar-eval-tnt
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=05:00:00
#SBATCH --error=$3DV_DIR/Project/slurm/%J.err
#SBATCH --output=$3DV_DIR/Project/slurm/%J.out

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
conda activate tnt_eval
echo "Finished activating environment"
which python

cd $3DV_DIR/Project/Methods/SuGaR/
python eval_tnt_w_2dgs_struct.py    --TNT_data $3DV_DATADIR/TNT_GOF/TrainingSet \
                                    --TNT_GT $3DV_DATADIR/TNT
