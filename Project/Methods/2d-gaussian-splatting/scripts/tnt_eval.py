import os
import sys
import argparse
tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
tnt_large_scenes = ['Meetingroom', 'Courthouse']
parser = argparse.ArgumentParser(description="Sequential training, rendering, and evaluation for Tanks and Temples.")
parser.add_argument("--TNT_data", type=str, required=True, help="Path to the Tanks and Temples dataset for training.")
parser.add_argument("--output_path", type=str, default="./eval/tnt", help="Path to save model outputs.")
parser.add_argument("--TNT_GT", type=str, help="Path to the Tanks and Temples ground truth for evaluation.")
parser.add_argument("--skip_training", action="store_true", help="Skip the training stage.")
parser.add_argument("--skip_rendering", action="store_true", help="Skip the rendering stage.")
parser.add_argument("--skip_evaluation", action="store_true", help="Skip the evaluation stage.")
args = parser.parse_args()



if not args.skip_evaluation and not args.TNT_GT:
    print("\nError: --TNT_GT is required for evaluation.")
    print("Please provide the path to the ground truth data or use --skip_evaluation.")
    sys.exit(1)

if not args.skip_training:
    for scene in tnt_360_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)
        
        os.makedirs(model_output_path, exist_ok=True)
        print(f"Training scene {scene}")
        cmd_train = (
            f"python train.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"--quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 100"
        )
        print(cmd_train)
        os.system(cmd_train)

    for scene in tnt_large_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)

        os.makedirs(model_output_path, exist_ok=True)
        print(f"Training scene {scene}")
        cmd_train = (
            f"python train.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"--quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 10"
        )
        print(cmd_train)
        os.system(cmd_train)

if not args.skip_rendering:
    for scene in tnt_360_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)

        print(f"Rendering scene {scene}")
        cmd_render = (
            f"python render.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"--iteration 30000 --quiet --depth_ratio 1.0 --num_cluster 1 "
            f"--voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
        )
        print(cmd_render)
        os.system(cmd_render)

    for scene in tnt_large_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)
        
        print(f"Rendering scene {scene}")
        cmd_render = (
            f"python render.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"--iteration 30000 --quiet --depth_ratio 1.0 --num_cluster 1 "
            f"--voxel_size 0.006 --sdf_trunc 0.024 --depth_trunc 4.5"
        )
        print(cmd_render)
        os.system(cmd_render)

if not args.skip_evaluation:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_scenes = tnt_360_scenes + tnt_large_scenes
    for scene in all_scenes:
        dataset_dir = os.path.join(args.TNT_GT, scene)
        traj_path = os.path.join(args.TNT_data, scene, f"{scene}_COLMAP_SfM.log")
        ply_path = os.path.join(args.output_path, scene, "train/ours_30000/fuse_post.ply")
        print(f"Evaluating mesh for {scene}")
        cmd_eval = (
            f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py "
            f"--dataset-dir {dataset_dir} "
            f"--traj-path {traj_path} "
            f"--ply-path {ply_path}"
        )
        print(cmd_eval)
        os.system(cmd_eval)
