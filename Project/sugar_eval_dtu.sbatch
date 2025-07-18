#!/bin/bash
#SBATCH --job-name=sugar-eval-dtu
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=05:30:00
#SBATCH --error=/home/stud213/3DV-SU25/Project/slurm/%J.err
#SBATCH --output=/home/stud213/3DV-SU25/Project/slurm/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cesar.diaz-blanco@student.uni-tuebingen.de

# Activate conda and the 2dgs environment which has libraries for evaluation code
source ~/miniconda3/bin/activate
conda activate surfel_splatting
echo "Finished activating environment"
which python

cd /home/stud213/3DV-SU25/Project/Methods/SuGaR/
python eval_dtu_w_2dgs_struct.py    --dtu /home/stud213/Datasets/DTU_2DGS \
                                    --DTU_Official /home/stud213/Datasets/DTU
