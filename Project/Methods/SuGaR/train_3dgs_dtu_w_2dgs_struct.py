import os
import argparse

dtu_scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

parser = argparse.ArgumentParser(description="Automated script to run the 3DGS baseline.")
parser.add_argument("--dtu_path", type=str, required=True, 
                    help="Path to the preprocessed DTU dataset directory.")
parser.add_argument("--baseline_output_path", type=str, default="./eval/dtu_3dgs_baseline", 
                    help="Path to save the baseline model outputs, to keep them separate from full SuGaR runs.")
args = parser.parse_args()

for scene in dtu_scenes:
    source_path = os.path.join(args.dtu_path, f"scan{scene}")
    scene_output_path = os.path.join(args.baseline_output_path, f"scan{scene}")

    cmd_train_baseline = (
        f"python run_3dgs_baseline.py "
        f"-s {source_path} "
        f"--output_dir {scene_output_path} "
        f"--white_background True "
    )
    print(cmd_train_baseline)
    os.system(cmd_train_baseline)
