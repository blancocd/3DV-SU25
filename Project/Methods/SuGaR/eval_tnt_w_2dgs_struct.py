import os
import argparse

tnt_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
parser = argparse.ArgumentParser(description="Evaluation for SuGaR on Tanks and Temples.")
parser.add_argument("--TNT_data", type=str, required=True, help="Path to the preprocessed TNT dataset for training.")
parser.add_argument("--TNT_GT", type=str, help="Path to the TNT ground truth dataset for evaluation.")
args = parser.parse_args()

script_dir_2dgs = '../2d-gaussian-splatting/scripts/'
for scene in tnt_scenes:
    dataset_dir = os.path.join(args.TNT_GT, scene)
    traj_path = os.path.join(args.TNT_data, scene, f"{scene}_COLMAP_SfM.log")

    print(f"Evaluating coarse mesh for scan{scene}")
    ply_path = f'/home/stud213/3DV-SU25/Project/Methods/SuGaR/output/coarse_mesh/{scene}/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply'
    cmd_eval = (
        f"OMP_NUM_THREADS=4 python {script_dir_2dgs}/eval_tnt/run.py "
        f"--dataset-dir {dataset_dir} "
        f"--traj-path {traj_path} "
        f"--ply-path {ply_path}"
    )
    print(cmd_eval)
    os.system(cmd_eval)


    ply_path = f'/home/stud213/3DV-SU25/Project/Methods/SuGaR/output/refined_mesh/{scene}/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj'
    print(f"Evaluating refined mesh for scan{scene}")
    cmd_eval = (
        f"OMP_NUM_THREADS=4 python {script_dir_2dgs}/eval_tnt/run.py "
        f"--dataset-dir {dataset_dir} "
        f"--traj-path {traj_path} "
        f"--ply-path {ply_path}"
    )
    print(cmd_eval)
    os.system(cmd_eval)
