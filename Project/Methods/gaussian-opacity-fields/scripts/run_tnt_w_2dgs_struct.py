import os
import sys
import argparse

tnt_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
parser = argparse.ArgumentParser(description="Sequential training and evaluation for Gaussian Opacity Fields on Tanks and Temples.")
parser.add_argument("--TNT_data", type=str, required=True, help="Path to the preprocessed TNT dataset for training.")
parser.add_argument("--output_path", type=str, default="./eval/tnt", help="Path to save model outputs and meshes.")
parser.add_argument("--TNT_GT", type=str, help="Path to the TNT ground truth dataset for evaluation.")
parser.add_argument("--skip_training", action="store_true", help="Skip the training stage.")
parser.add_argument("--skip_meshing", action="store_true", help="Skip the mesh extraction with the binary search tetrahedra method.")
parser.add_argument("--skip_tsdf_meshing", action="store_true", help="Skip the mesh extraction with the TSDF fusion method.")
parser.add_argument("--skip_evaluation", action="store_true", help="Skip the evaluation stage.")
parser.add_argument("--skip_evaluation_tsdf", action="store_true", help="Skip the evaluation stage.")
args = parser.parse_args()

if (not args.skip_evaluation or not args.skip_evaluation_tsdf) and not args.TNT_GT:
    print("\nError: --TNT_GT is required for evaluation.")
    print("Please provide the path to the ground truth data or use --skip_evaluation.")
    sys.exit(1)

if not args.skip_training:
    for scene in tnt_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)
        
        os.makedirs(model_output_path, exist_ok=True)
        print(f"Training scene: {scene}")
        cmd_train = (
            f"python train.py "
            f"-s {source_path} "
            f"-m {model_output_path} "
            f"-r 2 --eval --use_decoupled_appearance"
        )
        print(cmd_train)
        os.system(cmd_train)

if not args.skip_meshing:
    for scene in tnt_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)
        
        print(f"Extracting mesh for scene: {scene}")
        cmd_extract_tetra = (
            f"python extract_mesh.py "
            f"-s {source_path} "
            f"-m {model_output_path} --iteration 30000"
        )
        print(cmd_extract_tetra)
        os.system(cmd_extract_tetra)

if not args.skip_tsdf_meshing:
    for scene in tnt_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)

        print(f"Extracting TSDF mesh for scene: {scene}")
        cmd_extract_tsdf = (
            f"python extract_mesh_tsdf.py "
            f"-s {source_path} "
            f"-m {model_output_path} --iteration 30000"
        )
        print(cmd_extract_tsdf)
        os.system(cmd_extract_tsdf)

if not args.skip_evaluation:
    for scene in tnt_scenes:
        gt_dir = os.path.join(args.TNT_GT, scene)
        traj_path = os.path.join(args.TNT_data, scene, f"{scene}_COLMAP_SfM.log")
        ply_path = os.path.join(args.output_path, scene, "test/ours_30000/fusion/mesh_binary_search_7.ply")
        print(f"Evaluating scene from their mesh: {scene}")
        cmd_eval = (
            f"OMP_NUM_THREADS=4 python eval_tnt/run.py "
            f"--dataset-dir {gt_dir} "
            f"--traj-path {traj_path} "
            f"--ply-path {ply_path}"
        )
        print(cmd_eval)
        os.system(cmd_eval)

if not args.skip_evaluation_tsdf:
    for scene in tnt_scenes:
        gt_dir = os.path.join(args.TNT_GT, scene)
        traj_path = os.path.join(args.TNT_data, scene, f"{scene}_COLMAP_SfM.log")
        ply_path = os.path.join(args.output_path, scene, "test/ours_30000/tsdf/tsdf.ply")
        print(f"Evaluating scene from TSDF mesh: {scene}")
        cmd_eval = (
            f"OMP_NUM_THREADS=4 python eval_tnt/run.py "
            f"--dataset-dir {gt_dir} "
            f"--traj-path {traj_path} "
            f"--ply-path {ply_path}"
        )
        print(cmd_eval)
        os.system(cmd_eval)
