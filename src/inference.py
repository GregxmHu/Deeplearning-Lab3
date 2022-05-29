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
from dog_datasets.testDataset import TestDataSet
from tqdm import tqdm
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from util import set_seed,sum_norm,min_max_norm
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn as nn
from util import init_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default=None, type=str,help="to load data files")
    parser.add_argument("--trained_model_path", required=True,type=str,default=None)
    parser.add_argument("--results_file",type=str,default=None)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--model", default="vgg11", type=str)
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--pre_process", default="resize", type=str)
    parser.add_argument("--class_name_file", default=None, type=str)
    parser.add_argument("--score_norm", default="softmax", type=str)
    
    args = parser.parse_args()


    # set random seed to keep the experiment reproduceble
    set_seed(args.seed)
    if args.pre_process == "resize":
        pre_process=transforms.Resize([224,224])
    trans=transforms.Compose([
        pre_process,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if  "vgg11" in args.model:
        model=Vgg11(3,120)
    elif  "resnet18" in args.model :
        model=Resnet18(3,120)
    elif "senet18" in args.model :
        model=Resnet18(3,120,True)
    st=torch.load(args.trained_model_path)
    # can not very strict because the architecture is not the same
    model.load_state_dict(st,strict=True)

    test_dataset=TestDataSet(
        img_dir=os.path.join(args.test_data,"test/"),
        transform=trans,
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,collate_fn=test_dataset.collate)

    # load model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.score_norm=="softmax":
        stan_func=nn.Softmax(dim=1)
    elif args.score_norm == "sum":
        stan_func=sum_norm
    elif args.score_norm == "minmax":
        stan_func=min_max_norm
    model = model.to(device)
    model.eval()
    temp_results=[]
    init_csv(args.results_file,args.class_name_file)
    with torch.no_grad():
        for index,img in tqdm(test_loader,desc="Inferencing..."):
            img=img.to(device)
            logits = model(img)
            if args.score_norm=="softmax":  
                probilities=stan_func(logits).tolist()
            else:
                probilities=stan_func(logits)
            for i in range(len(img)):
                item=index[i]+','+','.join(
                    [str(prob) for prob in probilities[i]]
                )+'\n'
                temp_results.append(item)
            with open(args.results_file,'a+') as f:
                for s_results in temp_results:
                    f.write(s_results)
            temp_results=[]
