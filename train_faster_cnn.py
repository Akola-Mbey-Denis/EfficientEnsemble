from __future__ import division
import os
import argparse
import yaml
import random
import torch
from utils.seed import set_seed, setup_cudnn
from utils.plot import plot_loss_and_lr, plot_map
from datasets.MOT import MOT17ObjDetect
from datasets.KITTI import KITTIObjectDetect as KITTI
from datasets.BDD import BDDObjectDetect as BDD
from torch.autograd import Variable
import torch.optim as optim
from models.faster_rcnn import FRCNN_FPN
from tools.engine import train_one_epoch,evaluate_map
import torchvision.transforms as T
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolo_config.cfg',
                        help='config file. see readme')
    parser.add_argument('--data', type=str, default='MOT17DET',
                        help='config file. see readme')
    return parser.parse_args()

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
        transforms.append(T.RandomHorizontalFlip(0.5))
        
    return T.Compose(transforms)

def write_tb(writer, num, info):
    for item in info.items():
        writer.add_scalar(item[0], item[1], num)


# use our dataset and defined transformations
if data =='MOT17DET':
    dataset = MOT17ObjDetect(path=args['TRAIN_PATH'], split='train',transforms = get_transform(train=True))
    dataset_validation =MOT17ObjDetect(path=args['TEST_PATH'], split ='train', transforms = get_transform(train=False))
elif data=='KITTI':
    dataset = KITTI(root_dir=args['DATA_PATH'], image_set="train",transforms = get_transform(train=True)) 
elif data=='BDD':
    dataset = BDD(root_dir=args['DATA_PATH'], image_set="train",transforms = get_transform(train=True))
    dataset_validation= BDD(root_dir=args['DATA_PATH'], image_set="val",transforms = get_transform(train=False))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if data =='KITTI':
    dataset_size = len(dataset)
    validation_split =args['VALIDATION_SPLIT']
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,
    collate_fn=dataset.collate_fn, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,
    collate_fn=dataset.collate_fn, sampler=valid_sampler)
else:
    # dataloader for training
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,
    collate_fn=dataset.collate_fn)
    validation_loader=torch.utils.data.DataLoader(dataset_validation, batch_size=args['BATCH_SIZE'], shuffle=False, num_workers=4,
    collate_fn=dataset.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Instantiate model
model = FRCNN_FPN(num_classes =args['NUM_CLASSES'])
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args['LR'], momentum=args['MOMENTUM'], weight_decay=args['WEIGHT_DECAY'])

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args['STEPSIZE'],gamma=0.33)

train_loss = []
learning_rate = []
train_mAP_list = []
val_mAP = []

best_mAP = 0
writer = SummaryWriter(os.path.join(args['OUTPUT_DIR'], 'epoch_log'))

for epoch in range(0, args['EPOCHS']):
    loss_dict, total_loss = train_one_epoch(model, optimizer, train_loader,
                                                device, epoch, train_loss=train_loss, train_lr=learning_rate,
                                                print_freq=100, warmup=False)

    lr_scheduler.step()
    _, mAP = evaluate_map(model, validation_loader, device=device, mAP_list=val_mAP)
    print('validation mAp is {}'.format(mAP))
    print('best mAp is {}'.format(best_mAP))
    board_info = {'lr': optimizer.param_groups[0]['lr'],
                    #   'train_mAP': train_mAP,
                      'val_mAP': mAP
                }

    for k, v in loss_dict.items():
        board_info[k] = v.item()
        board_info['total loss'] = total_loss.item()
        write_tb(writer, epoch, board_info)

    if mAP > best_mAP:
        best_mAP = mAP
        # save weights
        save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
        model_save_dir = args['OUTPUT_DIR']
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            torch.save(save_files,
                       os.path.join(model_save_dir, "{}-model-{}-mAp-{}.pth".format('resnet101'+data, epoch, mAP)))
writer.close()
# plot loss and lr curve
if len(train_loss) != 0 and len(learning_rate) != 0:
    plot_loss_and_lr(train_loss, learning_rate, OUTPUT_DIR)

# plot mAP curve
if len(val_mAP) != 0:
    plot_map(val_mAP, OUTPUT_DIR)

