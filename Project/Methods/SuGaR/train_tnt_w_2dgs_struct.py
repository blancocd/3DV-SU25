import os
import argparse

tnt_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
parser = argparse.ArgumentParser(description="Sequential training and evaluation for Gaussian Opacity Fields on Tanks and Temples.")
parser.add_argument("--TNT_data", type=str, required=True, help="Path to the preprocessed TNT dataset for training.")
parser.add_argument("--TNT_GT", type=str, help="Path to the TNT ground truth dataset for evaluation.")
args = parser.parse_args()

for scene in tnt_scenes:
    source_path = os.path.join(args.TNT_data, scene)
    
    print(f"Training scene: {scene}")
    cmd_train = (
        f"python train_full_pipeline.py "
        f"-s {source_path} "
        f"--regularization_type dn_consistency "
        f"--high_poly True "
        f"--low_poly False "
        f"--refinement_time long "
        f"--white_background True "
        f"--square_size 8 "
        f"--eval False"
    )
    print(cmd_train)
    os.system(cmd_train)
