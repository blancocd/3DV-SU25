import os
import sys
import argparse

dtu_scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
parser = argparse.ArgumentParser(description="Sequential training and evaluation for Gaussian Opacity Fields.")
parser.add_argument("--dtu", type=str, required=True, help="Path to the preprocessed DTU dataset for training.")
parser.add_argument("--output_path", type=str, default="./eval/dtu", help="Path to save model outputs and meshes.")
parser.add_argument("--DTU_Official", type=str, help="Path to the DTU ground truth dataset for evaluation.")
parser.add_argument("--skip_training", action="store_true", help="Skip the training stage.")
parser.add_argument("--skip_meshing", action="store_true", help="Skip the mesh extraction stage with tetra struct.")
parser.add_argument("--skip_tsdf_meshing", action="store_true", help="Skip the mesh extraction stage with TSDF algo.")
parser.add_argument("--skip_evaluation", action="store_true", help="Skip the evaluation stage.")
parser.add_argument("--skip_evaluation_tsdf", action="store_true", help="Skip the evaluation stage.")
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
            f"-r 2 --use_decoupled_appearance --lambda_distortion 1000"
        )
        print(cmd_train)
        os.system(cmd_train)

if not args.skip_meshing:
    for scene in dtu_scenes:
        source_path = os.path.join(args.dtu, f"scan{scene}")
        model_output_path = os.path.join(args.output_path, f"scan{scene}")

        print(f"Extracting mesh for scan{scene}")
        cmd_extract_tetra = (
            f"python extract_mesh.py "
            f"-s {source_path} "
            f"-m {model_output_path} --iteration 30000"
        )
        print(cmd_extract_tetra)
        os.system(cmd_extract_tetra)

if not args.skip_tsdf_meshing:
    for scene in dtu_scenes:
        source_path = os.path.join(args.dtu, f"scan{scene}")
        model_output_path = os.path.join(args.output_path, f"scan{scene}")
        print(f"Extracting meshes for scan{scene} with TSDF")
        cmd_extract_tsdf = (
            f"python extract_mesh_tsdf.py "
            f"-s {source_path} "
            f"-m {model_output_path} --iteration 30000"
        )
        print(cmd_extract_tsdf)
        os.system(cmd_extract_tsdf)

if not args.skip_evaluation:
    script_dir_2dgs = '../2d-gaussian-splatting/scripts/'
    script_dir_gof = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        model_output_path = os.path.join(args.output_path, f"scan{scene}")
        ply_file = os.path.join(model_output_path, "test/ours_30000/fusion/mesh_binary_search_7.ply")

        print(f"Evaluating mesh for scan{scene}")
        cmd_eval = (
            f"python {script_dir_2dgs}/eval_dtu/evaluate_single_scene.py "
            f"--input_mesh {ply_file} "
            f"--scan_id {scene} "
            f"--output_dir {script_dir_gof}/tmp/scan{scene} "
            f"--mask_dir {args.dtu} "
            f"--DTU {args.DTU_Official}"
        )
        print(cmd_eval)
        os.system(cmd_eval)

if not args.skip_evaluation_tsdf:
    script_dir_2dgs = '../2d-gaussian-splatting/scripts/'
    script_dir_gof = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        model_output_path = os.path.join(args.output_path, f"scan{scene}")
        ply_file = os.path.join(model_output_path, "test/ours_30000/tsdf/tsdf.ply")
        print(f"Evaluating mesh for scan{scene}")
        cmd_eval = (
            f"python {script_dir_2dgs}/eval_dtu/evaluate_single_scene.py "
            f"--input_mesh {ply_file} "
            f"--scan_id {scene} "
            f"--output_dir {script_dir_gof}/tmp/scan{scene} "
            f"--mask_dir {args.dtu} "
            f"--DTU {args.DTU_Official}"
        )
        print(cmd_eval)
        os.system(cmd_eval)
