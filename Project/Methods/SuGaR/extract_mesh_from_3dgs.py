import argparse
from sugar_utils.general_utils import str2bool
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to extract a mesh directly from a vanilla 3DGS checkpoint.')
    
    parser.add_argument('-s', '--scene_path', type=str, required=True)
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True,
                        help='Path to the vanilla 3DGS Checkpoint directory.')
    parser.add_argument('-i', '--iteration_to_load', type=int, default=30000)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('-l', '--surface_level', type=float, default=0.3)
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval', type=str2bool, default=True)

    args = parser.parse_args()

    mesh_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'iteration_to_load': args.iteration_to_load,
        'coarse_model_path': None,
        'surface_level': args.surface_level,
        'decimation_target': args.n_vertices_in_mesh,
        'project_mesh_on_surface_points': False,
        'mesh_output_dir': args.output_dir,
        'bboxmin': None,
        'bboxmax': None,
        'center_bbox': True,
        'gpu': args.gpu,
        'eval': args.eval,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': True,  # baseline, don't use their meshing method
        'use_vanilla_3dgs': True,
    })
    
    print("\n--- Extracting mesh using only vanilla 3DGS representation ---")
    extract_mesh_from_coarse_sugar(mesh_args)
    print(f"\n--- Mesh extraction complete. Output saved in {args.output_dir} ---")