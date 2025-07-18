import os
import argparse
from sugar_utils.general_utils import str2bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train a vanilla 3DGS model and extract a mesh.')
    parser.add_argument('-s', '--scene_path', type=str, required=True, help='Path to the scene data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background.')
    args = parser.parse_args()

    gs_checkpoint_dir = os.path.join(args.output_dir, "vanilla_gs_checkpoint")
    if gs_checkpoint_dir[-1] != os.path.sep:
            gs_checkpoint_dir += os.path.sep
    os.makedirs(gs_checkpoint_dir, exist_ok=True)
    
    white_background_str = '-w ' if args.white_background else ''
    os.system(
        f"CUDA_VISIBLE_DEVICES={args.gpu} python ./gaussian_splatting/train.py \
            -s {args.scene_path} \
            -m {gs_checkpoint_dir} \
            -r 2 \
            {white_background_str}\
            --iterations 30_000"
    )
    
    mesh_output_dir = os.path.join(args.output_dir, "baseline_mesh")
    os.system(
        f"python extract_mesh_from_3dgs.py \
            -s {args.scene_path} \
            -c {gs_checkpoint_dir} \
            --output_dir {mesh_output_dir} \
            -i 30000 \
            --gpu {args.gpu}"
    )