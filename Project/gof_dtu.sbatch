#!/bin/bash
#SBATCH --job-name=gof-dtu
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=01:00:00
#SBATCH --error=/home/stud213/3DV-SU25/Project/slurm/%J.err
#SBATCH --output=/home/stud213/3DV-SU25/Project/slurm/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cesar.diaz-blanco@student.uni-tuebingen.de

# Copy preprocessed DTU dataset which is at /home/stud213/Datasets/DTU_2DGS
cp -R /home/stud213/Datasets/DTU_2DGS/ /scratch/$SLURM_JOB_ID/DTU_2DGS
echo "Finished copying preprocessed DTU dataset"

# Copy gt DTU dataset which is at /home/stud213/Datasets/DTU
cp -R /home/stud213/Datasets/DTU/ /scratch/$SLURM_JOB_ID/DTU
echo "Finished copying gt DTU dataset"

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
conda activate gof
echo "Finished activating environment"
which python

cd /home/stud213/3DV-SU25/Project/Methods/gaussian-opacity-fields/
python scripts/run_dtu_w_2dgs_struct.py --dtu /scratch/$SLURM_JOB_ID/DTU_2DGS   \
                                        --DTU_Official /scratch/$SLURM_JOB_ID/DTU \
                                        # --skip_training --skip_tsdf_meshing --skip_evaluation
