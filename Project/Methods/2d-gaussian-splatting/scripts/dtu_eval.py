import os
import sys
import argparse

dtu_scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
parser = argparse.ArgumentParser(description="Sequential training, rendering, and evaluation.")
parser.add_argument("--dtu", type=str, required=True, help="Path to the preprocessed DTU dataset.")
parser.add_argument("--output_path", type=str, default="./eval/dtu", help="Path to save model outputs.")
parser.add_argument("--DTU_Official", type=str, help="Path to the DTU ground truth dataset for evaluation.")
parser.add_argument("--skip_training", action="store_true", help="Skip the training stage.")
parser.add_argument("--skip_rendering", action="store_true", help="Skip the rendering stage.")
parser.add_argument("--skip_evaluation", action="store_true", help="Skip the evaluation stage.")
args = parser.parse_args()



if not args.skip_evaluation and not args.DTU_Official:
    print("\nError: --DTU_Official is required for evaluation.")
    print("Please provide the path to the ground truth data or use --skip_evaluation.")
    sys.exit(1)

if not args.skip_training:
    for scene in dtu_scenes:
        source_path = os.path.join(args.dtu, f"scan{scene}")
        model_output_path = os.path.join(args.output_path, f"scan{scene}")
        
        os.makedirs(model_output_path, exist_ok=True)
        print(f"Training scene scan{scene}")
        cmd_train = (
            f"python train.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"--quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
        )
        print(cmd_train)
        os.system(cmd_train)

if not args.skip_rendering:
    for scene in dtu_scenes:
        source_path = os.path.join(args.dtu, f"scan{scene}")
        model_output_path = os.path.join(args.output_path, f"scan{scene}")

        print(f"Rendering scene scan{scene}")
        cmd_render = (
            f"python render.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"--iteration 30000 --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 "
            f"--voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
        )
        print(cmd_render)
        os.system(cmd_render)

if not args.skip_evaluation:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for scene in dtu_scenes:
        model_output_path = os.path.join(args.output_path, f"scan{scene}")
        ply_file = os.path.join(model_output_path, "train/ours_30000/fuse_post.ply")

        print(f"Evaluating mesh for scan{scene}")
        cmd_eval = (
            f"python {script_dir}/eval_dtu/evaluate_single_scene.py "
            f"--input_mesh {ply_file} "
            f"--scan_id {scene} "
            f"--output_dir {script_dir}/tmp/scan{scene} "
            f"--mask_dir {args.dtu} "
            f"--DTU {args.DTU_Official}"
        )
        print(cmd_eval)
        os.system(cmd_eval)
