import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import cls_model, seg_model
from data_loader import get_data_loader
from utils import randomize, save_checkpoint, create_dir
from tqdm import tqdm



def train(train_dataloader, model, opt, epoch, args, writer):
    
    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in tqdm(enumerate(train_dataloader)):
        point_clouds, labels = batch
        point_clouds = point_clouds[:, :args.num_points]  # Sample points per object
        point_clouds = point_clouds.to(args.device)
        if args.randomization:
            point_clouds = randomize(point_clouds) # Randomize the point clouds
        labels = labels.to(args.device).to(torch.long)

        # TODO: Forward Pass
        logits, rot1, rot2 = model(point_clouds)
        predictions = logits

        if (args.task == "seg"):
            labels = labels[:, :args.num_points].reshape([-1])
            predictions = predictions.reshape([-1, args.num_seg_class])
            
        # TODO: Compute Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)

        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("train_loss", loss.item(), step+i)

    return epoch_loss

def test(test_dataloader, model, epoch, args, writer=None):
    
    model.eval()

    # Evaluation in Classification Task
    if (args.task == "cls"):
        correct_obj = 0
        num_obj = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds[:, :args.num_points]
            if args.randomization:
                point_clouds = randomize(point_clouds)
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # TODO: Forward pass and compute predicted labels
            with torch.no_grad():
                logits, _, _ = model(point_clouds)
                pred_labels = logits.max(1)[1]
                
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_obj / num_obj
                
    # Evaluation in Segmentation Task
    else:
        correct_point = 0
        num_point = 0
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds[:, :args.num_points]
            if args.randomization:
                point_clouds = randomize(point_clouds)
            point_clouds = point_clouds.to(args.device)
            labels = labels[:, :args.num_points].to(args.device).to(torch.long)

            # TODO: Forward pass and compute predicted labels
            with torch.no_grad():
                logits, _, _ = model(point_clouds)
                pred_labels = logits.max(2)[1]  # shape (B, N)
                pred_labels = pred_labels.reshape(-1)

            #correct_point += pred_labels.eq(labels.data).cpu().sum().item()
            correct_point += pred_labels.eq(labels.reshape(-1)).cpu().sum().item()
            num_point += labels.reshape([-1,1]).size()[0]

        # Compute Accuracy of Test Dataset
        accuracy = correct_point / num_point
    if not args.eval:
        writer.add_scalar("test_acc", accuracy, epoch)
    return accuracy


def main(args):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir("./logs")

    # Tensorboard Logger
    writer = SummaryWriter("./logs/{0}".format(args.task+"_"+args.exp_name))

    if args.task == "cls":
        model = cls_model(num_classes=args.num_cls_class, vanilla=args.vanilla).to(args.device)
    else:
        model = seg_model(num_seg_classes=args.num_seg_class, vanilla=args.vanilla).to(args.device)
    
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
    test_dataloader = get_data_loader(args=args, train=False)

    print ("successfully loaded data")

    best_acc = -1

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{})".format(args.task+"_"+args.exp_name))
    
    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        
        # Test
        current_acc = test(test_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, current_acc))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if (current_acc >= best_acc):
            best_acc = current_acc
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    print ("======== training completes ========")


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model & Data hyper-parameters
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate the model instead of training")
    parser.add_argument("--task", type=str, default="cls", help="The task: cls or seg")
    parser.add_argument("--num_points", type=int, default=8192, help="The number of points per object (default 2048)")
    parser.add_argument("--num_seg_class", type=int, default=6, help="The number of segmentation classes")
    parser.add_argument("--num_cls_class", type=int, default=3, help="The number of classification classes")
    parser.add_argument("--randomization", action="store_true", help="Whether to randomize the point clouds during training")
    parser.add_argument("--vanilla", action="store_true", help="Whether to use vanilla PointNet without rotation prediction")
    
    # Training hyper-parameters
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images in a batch.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of threads to use for the DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (default 0.0001)")

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

    if args.eval:
        test_dataloader = get_data_loader(args=args, train=False)
        
        if args.task == "cls":
            model = cls_model(num_classes=args.num_cls_class, vanilla=args.vanilla).to(args.device)
        else:
            model = seg_model(num_seg_classes=args.num_seg_class, vanilla=args.vanilla).to(args.device)
            
        # Load Checkpoint 
        if args.load_checkpoint:
            model_path = "{}/{}.pt".format(args.checkpoint_dir,args.load_checkpoint)
            with open(model_path, "rb") as f:
                state_dict = torch.load(f, map_location=args.device)
                model.load_state_dict(state_dict)
            print ("successfully loaded checkpoint from {}".format(model_path))
        acc = test(test_dataloader, model, 0, args)
        print(acc)
    else:
        main(args)