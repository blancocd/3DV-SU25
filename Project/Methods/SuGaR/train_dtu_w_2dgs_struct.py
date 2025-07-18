import os
import argparse

dtu_scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
dtu_scenes = [110, 114, 118, 122]
parser = argparse.ArgumentParser(description="Automated SuGaR training script for the DTU dataset.")
parser.add_argument("--dtu", type=str, required=True, help="Path to the preprocessed DTU dataset directory.")
parser.add_argument("--DTU_Official", type=str, help="Path to the DTU ground truth dataset for evaluation.")
args = parser.parse_args()

for scene in dtu_scenes:
    source_path = os.path.join(args.dtu, f"scan{scene}")
    
    print(f"Training scene scan{scene}")
    cmd_train = (
        f"python train_full_pipeline.py "
        f"-s {source_path} "
        f"--regularization_type dn_consistency "
        f"--high_poly True "
        f"--low_poly False "
        f"--refinement_time long "
        f"--white_background True "
        f"--square_size 8 "
        f"--eval False "
        f"--port 6010 "
    )
    print(cmd_train)
    os.system(cmd_train)
