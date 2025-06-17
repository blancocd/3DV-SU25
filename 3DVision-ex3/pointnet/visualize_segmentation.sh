#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=64gb
#SBATCH --output=visualize_segmentation.txt
#SBATCH --error=visualize_segmentation.txt

eval "$(~/miniconda3/bin/conda shell.bash hook)"
if [ -z "$CONDA_INTERPRETER" ]; then
	export CONDA_INTERPRETER="3DVision-ex3" # Check if CONDA_INTERPRETER is not set, then set it to garage_2
fi
source activate "$CONDA_INTERPRETER"
which python3

python eval_seg.py --i 0 --load_checkpoint best_model --exp_name 0

python eval_seg.py --i 1 --load_checkpoint best_model --exp_name 1


python eval_seg.py --i 2 --load_checkpoint best_model --exp_name 2


python eval_seg.py --i 3 --load_checkpoint best_model --exp_name 3


python eval_seg.py --i 4 --load_checkpoint best_model --exp_name 4
