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
import scipy
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as T
from datasets.MOT import MOT17ObjDetect
from datasets.KITTI import KITTIObjectDetect as KITTI
from datasets.BDD import BDDObjectDetect as BDD
import torch.nn as nn
from utils.utils import *
from utils.seed import set_seed, setup_cudnn
from models.faster_rcnn import FRCNN_FPN
from torchmetrics.functional import calibration_error
from utils.calibration_utils import get_batch_statistics
from mmcv.ops.nms import nms_match, soft_nms
from ensemble_suite import NaiveAveragger,weighted_boxes_fusion
from torch.nn.functional import normalize
from utils.detection_utils import visualize_boxes_and_labels_on_image_array
import uuid
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/ensemble.cfg',
                        help='config file. see readme')
    parser.add_argument('--data', type=str, default='MOT17DET',
                        help='config file. see readme')
    parser.add_argument('--model_type', type=str, default='NONE',
                        help='config file. see readme')
    return parser.parse_args()


'''
Configuration of model parameters
'''
seed =12345
args =parse_args()
data =args.data
model_type =args.model_type
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
model1.eval()
model2.eval()
model3.eval()

KITTI_DICT ={1: 'Car', 2: 'Cyclist',3: 'Dontcare',4: 'Misc',5: 'Pedestrian', 6: 'Person_sitting', 7:'Tram',8: 'Truck', 9: 'Van', 10: 'Background', 11: 'bicycle',12: 'traffic light', 13: 'traffic sign'}
BDD_DICT ={1: 'Pedestrain',2: 'rider',3:'car', 4: 'truck', 5: 'bus',6: 'train',7: 'motorcycle',8:'bicycle',9:'traffic light',10: 'traffic sign'}
MOT_DICT ={1: 'Pedestrain'}
isVisualise=True
count=0
for images, targets in data_loader:
    image = list(img.to(device) for img in images)
    images =list(img.to(device) for img in images)
    with torch.no_grad():
        img = image[0].unsqueeze_(0)
        model1.load_image(img)
        model2.load_image(img)
        model3.load_image(img)
        if args['ENSEMBLE_TYPE'] =='Triple':
            boxes1,scores1,labels1 =model1.detect(img) 
            boxes2,scores2,labels2=model2.detect(img)
            boxes3,scores3,labels3 =model3.detect(img)
        elif  args['ENSEMBLE_TYPE'] =='Single':          
            features, proposals = model1.get_proposal()
            boxes1,scores1,labels1 =model1.FastRCNN_prediction_head(proposals)
            boxes2,scores2,labels2 = model2.FastRCNN_prediction_head(proposals)
            boxes3,scores3,labels3 = model3.FastRCNN_prediction_head(proposals)

        if args['ensembleMethod'] !='wbf':
                boxes =torch.cat((boxes1,boxes2,boxes3),dim=0)
                scores =torch.cat((scores1,scores2,scores3),dim=0)
                labels_t =torch.cat((labels1,labels2,labels3),dim=0)
                scores,boxes, variance,labels = NaiveAveragger(boxes=boxes.tolist(),scores=scores.tolist(),labels=labels_t.tolist(),iou_threshold=args['IOU'],nms_method="match")
                if isVisualise:
                    image =images[0].permute(1, 2, 0).cpu().numpy()
                    boxes =boxes.cpu().detach().numpy()
                    scores =scores.cpu().detach().numpy()
                    classes=labels.cpu().detach().numpy()
                    variance =np.round(variance.cpu().detach().numpy(),2)
                    plots =visualize_boxes_and_labels_on_image_array(image, 
                                                        boxes, 
                                                        classes, 
                                                        scores=scores, 
                                                        variance =variance,
                                                        label_map=KITTI_DICT)

                    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                    width, height = plots.size
                    fig.set_size_inches(width / 80, height / 80)
                    ax.imshow(plots)
                    plt.axis('off')
                    filename = str(uuid.uuid4())
                    plt.savefig("./images_kitti/"+filename+".png")
        else:
            boxes_list =[boxes1.tolist(),boxes2.tolist(),boxes3.tolist()]
            scores_list =[scores1.tolist(),scores2.tolist(),scores3.tolist()]
            labels_list  =[labels1.tolist(),labels2.tolist(),labels3.tolist()]
        
        
            boxes, scores, labels,variance = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=args['WEIGHTS'], iou_thr=args['IOU'], skip_box_thr=0.20)
            boxes =np.array(boxes,dtype=np.float32)
            scores =np.array(scores,dtype=np.float32)
            classes =np.array(labels,dtype=np.int16)
            variance =np.array(variance,dtype=np.float32)

            boxes =torch.from_numpy(boxes)
            scores =torch.from_numpy(scores)
            variance =torch.from_numpy(variance)
            labels =torch.from_numpy(classes)
    
            boxes =boxes.cpu().detach().numpy()
            scores =scores.cpu().detach().numpy()
            classes =labels.cpu().detach().numpy()
            variance = np.round(variance.cpu().detach().numpy(),2)
            if isVisualise:
                image =images[0].permute(1, 2, 0).cpu().numpy()
                    
                plots =visualize_boxes_and_labels_on_image_array(image, 
                                                        boxes, 
                                                        classes, 
                                                        scores=scores, 
                                                        variance =variance,
                                                        label_map=MOT_DICT)

                fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                width, height = plots.size
                fig.set_size_inches(width / 80, height / 80)
                ax.imshow(plots)
                plt.axis('off')
                filename = str(uuid.uuid4())
                plt.savefig("./images_mot/"+filename+".png") 
        count+=1 
        if count==10:
            break        
