import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import cor_model
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir
from pytorch3d.loss import chamfer_distance



def train(train_dataloader, model, opt, epoch, args, writer):
    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        point_clouds = batch
        point_clouds = point_clouds[:, :args.num_points]  # Sample points per object
        point_clouds = point_clouds.to(args.device)
        
        # TODO: Forward Pass
        prediction, r1, r2 = model(point_clouds)

            
        # TODO: Compute Loss
        loss = chamfer_distance(prediction, point_clouds)
        
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("train_loss", loss.item(), step+i)

    return epoch_loss


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir("./logs")

    # Tensorboard Logger
    writer = SummaryWriter("./logs/{0}".format(args.task+"_"+args.exp_name))

    model = cor_model(num_points=args.num_points).to(args.device)
    
    # Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
        with open(model_path, "rb") as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print ("successfully loaded checkpoint from {}".format(model_path))

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # Dataloader for Training & Testing
    train_dataloader = get_data_loader(args=args, train=True)

    print ("successfully loaded data")

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{})".format(args.task+"_"+args.exp_name))
    
    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        print ("epoch: {}   train loss: {:.4f}".format(epoch, train_epoch_loss))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)


    print ("======== training complete ========")


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument("--num_points", type=int, default=8192, help="The number of points per object (default 2048)")
    
    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images in a batch.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of threads to use for the DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (default 0.001)")
    parser.add_argument("--task", type=str, default="cor", help="The task: cor (correspondence)")
    
    parser.add_argument("--exp_name", type=str, default="exp", help="The name of the experiment")

    # Directories and checkpoint/sample iterations
    parser.add_argument("--main_dir", type=str, default="/common/share/3DVision-ex3-data/data/")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int , default=10)

    parser.add_argument("--load_checkpoint", type=str, default="")
    

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific

    main(args)