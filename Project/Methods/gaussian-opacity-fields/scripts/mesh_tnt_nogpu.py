import os
import argparse

tnt_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
tnt_scenes = ['Meetingroom']
parser = argparse.ArgumentParser(description="Sequential training and evaluation for Gaussian Opacity Fields on Tanks and Temples.")
parser.add_argument("--TNT_data", type=str, required=True, help="Path to the preprocessed TNT dataset for training.")
parser.add_argument("--output_path", type=str, default="./eval/tnt", help="Path to save model outputs and meshes.")
parser.add_argument("--skip_meshing", action="store_true", help="Skip the mesh extraction with the binary search tetrahedra method.")
parser.add_argument("--skip_tsdf_meshing", action="store_true", help="Skip the mesh extraction with the TSDF fusion method.")
args = parser.parse_args()

if not args.skip_meshing:
    for scene in tnt_scenes:
        source_path = os.path.join(args.TNT_data, scene)
        model_output_path = os.path.join(args.output_path, scene)
        
        print(f"Extracting mesh for scene: {scene}")
        cmd_extract_tetra = (
            f"python extract_mesh.py "
            f"-s {source_path} "
            f"-m {model_output_path} --iteration 30000 --no_gpu"
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
            f"-m {model_output_path} --iteration 30000 --no_gpu"
        )
        print(cmd_extract_tsdf)
        os.system(cmd_extract_tsdf)
