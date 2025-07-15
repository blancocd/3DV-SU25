#!/bin/bash
#SBATCH --job-name=gof-tnt
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=18:00:00
#SBATCH --error=/home/stud213/3DV-SU25/Project/slurm/%J.err
#SBATCH --output=/home/stud213/3DV-SU25/Project/slurm/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cesar.diaz-blanco@student.uni-tuebingen.de

# Copy preprocessed training TNT dataset which is at /home/stud213/Datasets/TNT_GOF/TrainingSet
cp -R /home/stud213/Datasets/TNT_GOF/TrainingSet/ /scratch/$SLURM_JOB_ID/TNT_GOF
echo "Finished copying preprocessed TNT dataset"

# Copy gt TNT dataset which is at /home/stud213/Datasets/TNT
cp -R /home/stud213/Datasets/TNT/ /scratch/$SLURM_JOB_ID/TNT
echo "Finished copying gt TNT dataset"

# Activate conda and the respective environment
source ~/miniconda3/bin/activate
conda activate gof
echo "Finished activating environment"
which python

cd /home/stud213/3DV-SU25/Project/Methods/gaussian-opacity-fields/
python scripts/run_tnt_w_2dgs_struct.py --TNT_data /scratch/$SLURM_JOB_ID/TNT_GOF \
                                        --TNT_GT /scratch/$SLURM_JOB_ID/TNT \
                                        --skip_meshing --skip_tsdf_meshing --skip_evaluation --skip_evaluation_tsdf
                                        # --skip_training --skip_meshing --skip_tsdf_meshing --skip_evaluation --skip_evaluation_tsdf
