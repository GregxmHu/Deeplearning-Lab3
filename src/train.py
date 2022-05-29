from email.policy import strict
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.vgg11 import Vgg11
from models.resnet18 import Resnet18
from dog_datasets.Dog import DogBreed
from tqdm import tqdm
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from util import set_seed
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def dev(dataloader,model):
    model.eval()
    num_correct = 0
    total_num=0
    with torch.no_grad():
        for img,label in tqdm(dataloader,desc="inferencing develop set"):
            img, label = img.to(device), label.to(device)  
            logits = model(img)  
            _, pred = torch.max(logits, dim=1)  
            num_correct += (pred == label).sum().item()  
            total_num+=len(label)
    acc=num_correct / total_num
    model.train()
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str,help="to load data files")
    parser.add_argument("--checkpoint_save_folder", default=None, type=str,help="to save checkpoint")
    parser.add_argument("--pretrained_model_name_or_path", required=True,type=str,default=None)
    parser.add_argument("--log_dir", type=str,default=None)
    parser.add_argument("--results_file",type=str,default=None)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model", default="vgg11", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--pre_process", default="resize", type=str)
    parser.add_argument("--optim", default="SGD", type=str)
    
    seed_list=[13,21,87,100,42]
    args = parser.parse_args()
    # build log file
    if args.log_dir is not None:
        writer = SummaryWriter(args.log_dir)
        tb = writer

    # set random seed to keep the experiment reproduceble
    set_seed(args.seed)
    if args.pre_process == "RandomHorizontalFlip":
        pre_process=transforms.RandomHorizontalFlip(0.5)
    elif args.pre_process == "RandomVerticalFlip":
        pre_process=transforms.RandomVerticalFlip(0.5)
    elif args.pre_process == "RandomRotation":
        pre_process=transforms.RandomRotation(15)
    else:
        pre_process=transforms.Resize([224,224])
    trans=transforms.Compose([
        transforms.Resize([224,224]),
        pre_process,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if "vgg11" in args.model :
        model=Vgg11(3,120)
    elif "resnet18" in args.model:
        model=Resnet18(3,120)
    elif "senet18" in args.model:
        model=Resnet18(3,120,True)
    
    if args.pretrained_model_name_or_path is not None:
        print("load pretrained weight...")
        st=torch.load(args.pretrained_model_name_or_path)
        # can not very strict because the architecture is not the same
        model.load_state_dict(st,strict=False) 
    train_dataset=DogBreed(
        label_file=os.path.join(args.data,"labels.csv"),
        img_dir=os.path.join(args.data,"train/"),
        transform=trans,
        label_transform=torch.tensor
        )
    dev_datasets=[
        DogBreed(
        label_file=os.path.join(args.data,"labels.csv"),
        img_dir=os.path.join(args.data,"dev/{}".format(item)),
        transform=trans,
        label_transform=torch.tensor
        ) for item in seed_list
    ]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=24,collate_fn=train_dataset.collate)
    dev_loaders= [DataLoader(dev_dataset, batch_size=256, shuffle=False, num_workers=10,collate_fn=train_dataset.collate) 
                for dev_dataset in dev_datasets]
    # load model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.optim=="SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
                        weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    global_training_steps=0
    best_acc=0.0
    scheduler=get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epochs*len(train_loader)
    )
    for epoch in range(args.epochs):
        model.train()
        for img,label in tqdm(train_loader,desc="Training Epoch{}".format(epoch)):
            global_training_steps+=1
            img, label = img.to(device), label.to(device)  

            logits = model(img)  

            loss = loss_func(logits, label) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            scheduler.step()
            tb.add_scalar("loss",loss.item(),global_training_steps)

            #_, pred = torch.max(logits, dim=1)  
            #num_correct += (pred == label).sum().item()  


        mean_dev_acc=[dev(d_l,model) for d_l in dev_loaders]
        print("dev_acc ",mean_dev_acc)
        mean_dev_acc=sum(mean_dev_acc)/len(mean_dev_acc)
        print("mean_dev_acc ",mean_dev_acc)
        tb.add_scalar("mean_dev_acc",mean_dev_acc,epoch)
        if mean_dev_acc>best_acc:
        
            best_acc=mean_dev_acc
            print("save model at epoch {}".format(epoch))
            torch.save(model.state_dict(), '{}/{}_epoch{}_bz{}_lr{}_optim{}_aug{}_best.pth'.format(
                args.checkpoint_save_folder,args.model,args.epochs,args.batch_size,args.lr,args.optim,args.pre_process
            ))
