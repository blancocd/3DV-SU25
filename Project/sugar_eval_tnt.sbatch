#!/bin/bash
#SBATCH --job-name=sugar-eval-tnt
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=05:00:00
#SBATCH --error=/home/stud213/3DV-SU25/Project/slurm/%J.err
#SBATCH --output=/home/stud213/3DV-SU25/Project/slurm/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cesar.diaz-blanco@student.uni-tuebingen.de

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
conda activate tnt_eval
echo "Finished activating environment"
which python

cd /home/stud213/3DV-SU25/Project/Methods/SuGaR/
python eval_tnt_w_2dgs_struct.py    --TNT_data /home/stud213/Datasets/TNT_GOF/TrainingSet \
                                    --TNT_GT /home/stud213/Datasets/TNT
