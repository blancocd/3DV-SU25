import os
import argparse

dtu_scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
parser = argparse.ArgumentParser(description="Automated SuGaR evaluation script for the DTU dataset.")
parser.add_argument("--dtu", type=str, required=True, help="Path to the preprocessed DTU dataset directory.")
parser.add_argument("--DTU_Official", type=str, help="Path to the DTU ground truth dataset for evaluation.")
args = parser.parse_args()

script_dir_2dgs = '../2d-gaussian-splatting/scripts/'
script_dir_sugar = os.path.dirname(os.path.abspath(__file__))
for scene in dtu_scenes:
    ply_file = f'/home/stud213/3DV-SU25/Project/Methods/SuGaR/output/coarse_mesh/scan{scene}/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply'
    print(f"Evaluating coarse mesh for scan{scene}")
    cmd_eval = (
        f"python {script_dir_2dgs}/eval_dtu/evaluate_single_scene.py "
        f"--input_mesh {ply_file} "
        f"--scan_id {scene} "
        f"--output_dir {script_dir_sugar}/tmp_coarse/scan{scene} "
        f"--mask_dir {args.dtu} "
        f"--DTU {args.DTU_Official}"
    )
    print(cmd_eval)
    os.system(cmd_eval)

    ply_file = f'/home/stud213/3DV-SU25/Project/Methods/SuGaR/output/refined_mesh/scan{scene}/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj'
    print(f"Evaluating refined mesh for scan{scene}")
    cmd_eval = (
        f"python {script_dir_2dgs}/eval_dtu/evaluate_single_scene.py "
        f"--input_mesh {ply_file} "
        f"--scan_id {scene} "
        f"--output_dir {script_dir_sugar}/tmp_fine/scan{scene} "
        f"--mask_dir {args.dtu} "
        f"--DTU {args.DTU_Official}"
    )
    print(cmd_eval)
    os.system(cmd_eval)
