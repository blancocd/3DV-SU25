#!/bin/bash
#SBATCH --job-name=sugar-train-tnt
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=14:00:00
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

# Create conda env for A4000
cd $3DV_DIR/Project/Methods/SuGaR/
conda activate sugara4000
python -c "import torch; print(torch.cuda.get_device_capability(0))"
pip uninstall diff-gaussian-rasterization simple-knn -y
pip cache purge
export TORCH_CUDA_ARCH_LIST="8.6"
pip install -e gaussian_splatting/submodules/diff-gaussian-rasterization
pip install -e gaussian_splatting/submodules/simple-knn
unset TORCH_CUDA_ARCH_LIST 

echo "Finished modifying and activating the environment"
which python

python train_tnt_w_2dgs_struct.py   --TNT_data /scratch/$SLURM_JOB_ID/TNT_GOF \
                                    --TNT_GT /scratch/$SLURM_JOB_ID/TNT
