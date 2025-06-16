import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_seg_class", type=int, default=6, help="The number of segmentation classes")
    parser.add_argument("--num_points", type=int, default=8192, help="The number of points per object to be included in the input data")

    # Directories and checkpoint/sample iterations
    parser.add_argument("--load_checkpoint", type=str, default="best_model")
    parser.add_argument("--i", type=int, default=0, help="index of the object to visualize")

    parser.add_argument("--test_data", type=str, default="/common/share/3DVision-ex3-data/data/seg/data_test.npy")
    parser.add_argument("--test_label", type=str, default="/common/share/3DVision-ex3-data/data/seg/label_test.npy")
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--exp_name", type=str, default="exp", help="The name of the experiment")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    create_dir(args.output_dir)

    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = f"./checkpoints/seg/{args.load_checkpoint}.pt"
    with open(model_path, "rb") as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    test_data = np.load(args.test_data)
    N = test_data.shape[1]
    ind = np.random.choice(N,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[None, args.i,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[None, args.i,ind])
    test_label = test_label.to(args.device)
    
    # TODO: Make prediction
    with torch.no_grad():
        logits, _, _ = model(test_data.to(args.device).float())  # (1, N, num_seg_class)
        pred_label = logits.argmax(dim=2)  # (1, N)


    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[0], test_label[0], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[0], pred_label[0], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
