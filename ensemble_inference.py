import os
import configparser
import torchvision
import csv
import yaml
import argparse
import os.path as osp
import pickle
from PIL import Image
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T
from datasets.MOT import MOT17ObjDetect
from datasets.KITTI import KITTIObjectDetect as KITTI
from datasets.BDD import BDDObjectDetect as BDD
from tools.engine import train_one_epoch, evaluate,evaluate_map,evaluate_triple,evaluate_montecarlo
import torch.nn as nn
from utils.seed import set_seed, setup_cudnn
from models.faster_rcnn import FRCNN_FPN
from torchmetrics.functional import calibration_error
from utils.calibration_utils import get_batch_statistics
from ensemble_suite import NaiveAveragger,weighted_boxes_fusion
from torch.nn.functional import normalize

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/MOT17/ensemble.cfg',
                        help='config file. see readme')
    parser.add_argument('--data', type=str, default='MOT17DET',
                        help='config file. see readme')
    return parser.parse_args()


'''
Configuration of model parameters
'''
seed =12345
args =parse_args()
data =args.data
with open(args.cfg, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)

print(args)

# Set seeds to ensure reproducibilty
args=args[data]
set_seed(seed=args['SEED'])
setup_cudnn(deterministic=args['DETERMINISTIC'])

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor    
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
        
    return T.Compose(transforms)

# use our dataset and defined transformations
if data =='MOT17DET':
    dataset = MOT17ObjDetect(path=args['DATA_PATH'], split='train',transforms = get_transform(train=False))
elif data=='KITTI':
     dataset = KITTI(root_dir=args['DATA_PATH'], image_set="train",transforms =get_transform(train=False))
elif data=='BDD':
    dataset = BDD(root_dir=args['DATA_PATH'], image_set="val",transforms =get_transform(train=False))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# dataloader for test
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,
    collate_fn=dataset.collate_fn)
model_loaders=[]

nums =args['NUM_ENSEMBLE']
for i in range(args['NUM_ENSEMBLE']):
    model_loaders.append(FRCNN_FPN(num_classes=args['NUM_CLASSES'],use_dropout=args['DROPOUT']))


def load_ensemble_models(model_loaders, data):
    models_checkpoints =[]
    models_weights=[]
   
    models =[args['model_1_path'],args['model_2_path'],args['model_3_path']]
    for model_path in models:
        path = os.path.join(args['ENSEMBLE_MODELS'],model_path)
        models_checkpoints.append(torch.load(path, map_location='cpu'))
    nums =args['NUM_ENSEMBLE']
    for i in range(nums):
        if data =='KITTI':
            model_loaders[i].load_state_dict(models_checkpoints[i]['model'],strict=False)
        elif data=='BDD':
            model_loaders[i].load_state_dict(models_checkpoints[i]['model'])
        else:
            model_loaders[i].load_state_dict(models_checkpoints[i])       

        print("CUDA::",torch.cuda.is_available())
        if torch.cuda.is_available():
            model_loaders[i].cuda()
            models_weights.append(model_loaders[i])
    return models_weights


models =load_ensemble_models(model_loaders, data)

model1 =models[0]
model2 =models[1]
model3 =models[2]
if args['INFERENCE_TYPE']=='Single':
    del model1
    del model2
    if args['DROPOUT']:
        evaluate_montecarlo(model3,data_loader,device,args['IOU'],args['NMS'], args['ENSEMBLE_TYPE'],args['NMS_TYPE'])
    else:
        evaluate(model3,data_loader,device,args['IOU'])
else:
    print("Evaluating the  COCO metrics") 
    evaluate_triple(model1,model2,model3,data_loader,device,args['ENSEMBLE_TYPE'],args['IOU'],args['NMS'],args['ensembleMethod'],args['WEIGHTS'],args['NMS_TYPE'])

