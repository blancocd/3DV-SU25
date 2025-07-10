import numpy as np
import argparse

import torch
from train_cor import cor_model
from utils import create_dir, viz_cor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_points", type=int, default=8192, help="The number of points per object to be included in the input data")

    # Directories and checkpoint/sample iterations
    parser.add_argument("--load_checkpoint", type=str, default="best_model")
    parser.add_argument("--i", type=int, default=0, help="index of the first object to visualize")
    parser.add_argument("--j", type=int, default=1, help="index of the second object to visualize")

    parser.add_argument("--test_data", type=str, default="/common/share/3DVision-ex3-data/data/cor/data_test.npy")
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--exp_name", type=str, default="cor", help="The name of the experiment")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    create_dir(args.output_dir)

    model = cor_model(num_points=args.num_points).to(args.device)
    
    # Load Model Checkpoint
    model_path = f"./checkpoints/cor/{args.load_checkpoint}.pt"
    with open(model_path, "rb") as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    test_data = np.load(args.test_data)
    max_points = test_data.shape[1]
    ind = np.random.choice(max_points,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])[[args.i,args.j]]

    # TODO: Predict point cloud
    with torch.no_grad():
        pred_pc, _, _ = model(test_data.to(args.device).float())

    
    rendering_pc = pred_pc.detach().cpu()

    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_cor(rendering_pc[0], rendering_pc[1], f"{args.output_dir}/cor.gif", args.device)
