from __future__ import division
from utils.utils import *
from utils.seed import set_seed, setup_cudnn
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.parse_yolo_weights import parse_yolo_weights
from datasets.MOT_YOLO import MOT17ObjDetectYOLO 
from datasets.KITTI_YOLO import KITTIObjectDetectYOLO
from datasets.BDD_YOLO import BDDObjectDetectYOLO
from models.yolov3 import *
from torchmetrics.functional import calibration_error
from utils.calibration_utils import get_batch_statistics
import time
import os
import argparse
import yaml
import random
import torch
from torch.autograd import Variable
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/yolo_eval.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path', type=str,
                        default=None, help='darknet weights file')
    parser.add_argument('--data', type=str,
                        default='MOT17DET', help='dataset to use')
    parser.add_argument('--n_cpu', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--checkpoint_interval', type=int,
                        default=1000, help='interval between saving checkpoints')
    parser.add_argument('--eval_interval', type=int,
                            default=2500, help='interval between evaluations')
    parser.add_argument('--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument(
        '--tfboard_dir', help='tensorboard path for logging', type=str, default=None)
    return parser.parse_args()

def yolobox2label_t(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
 
    h, w, nh, nw, dx, dy = info_img
    _,y1, x1, y2, x2 = box
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [y1, x1, y1 + box_h, x1 + box_w]

   
    return label

def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # print("Box is ",box)
    h, w, nh, nw, dx, dy = info_img
    
    y1, x1, y2, x2 = box
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    dataset_type =args.data
    print("Setting Arguments.. : ", args)

    cuda = torch.cuda.is_available() and args.use_cuda
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
 
    cfg =cfg[dataset_type]
    print("successfully loaded config file: ", cfg)
    print(cfg['TRAIN'])

    momentum = cfg['TRAIN']['MOMENTUM']
    decay = cfg['TRAIN']['DECAY']
    burn_in = cfg['TRAIN']['BURN_IN']
    iter_size = cfg['TRAIN']['MAXITER']
    steps = eval(cfg['TRAIN']['STEPS'])
    batch_size = cfg['TRAIN']['BATCHSIZE']
    subdivision = cfg['TRAIN']['SUBDIVISION']
    ignore_thre = cfg['TRAIN']['IGNORETHRE']
    random_resize = cfg['AUGMENTATION']['RANDRESIZE']
    base_lr = cfg['TRAIN']['LR'] / batch_size / subdivision
    gradient_clip = cfg['TRAIN']['GRADIENT_CLIP']
    print("Batch size :::",batch_size)

    print('effective_batch_size = batch_size * iter_size = %d * %d' %
          (batch_size, subdivision))

    # Make trainer behavior deterministic
    set_seed(seed=0)
    setup_cudnn(deterministic=True)

    # Learning rate setup
    def burnin_schedule(i):
        if i < burn_in:
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # Initiate model
    model = YOLOv3(cfg['MODEL'], ignore_thre=ignore_thre)

    if args.weights_path:
        print("loading darknet weights....", args.weights_path)
        parse_yolo_weights(model, args.weights_path)
    if args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    if cuda:
        print("using cuda") 
        model = model.cuda()

    if args.tfboard_dir:
        print("using tfboard")
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(args.tfboard_dir)

    model.eval()


    imgsize = cfg['TRAIN']['IMGSIZE']
    if dataset_type=='MOT17DET':
        dataset = MOT17ObjDetectYOLO(model_type=cfg['MODEL']['TYPE'],
                  data_dir=cfg['TRAIN_PATH'],
                  img_size=imgsize,
                  augmentation=cfg['AUGMENTATION'],
                  debug=args.debug,evaluate=True)
    elif dataset_type=='KITTI':
        dataset = KITTIObjectDetectYOLO(model_type=cfg['MODEL']['TYPE'],
                  data_dir=cfg['TRAIN_PATH'],
                  img_size=imgsize,
                  augmentation=cfg['AUGMENTATION'],
                  debug=args.debug,evaluate=True)
    elif dataset_type=='BDD':
        dataset =BDDObjectDetectYOLO(model_type=cfg['MODEL']['TYPE'],
                  data_dir=cfg['TRAIN_PATH'],
                  img_size=imgsize,
                  augmentation=cfg['AUGMENTATION'],
                  debug=args.debug,evaluate=True)


    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=args.n_cpu)
    dataiterator = iter(dataloader)
    if dataset_type=='MOT17DET':
        evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir=cfg['TEST_PATH'],
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    eval_labels=None,
                    num_classes = cfg['MODEL']['N_CLASSES'],
                    batch_size = cfg['TEST']['BATCH_SIZE'],
                    nmsthre = cfg['TEST']['NMSTHRE'],
                    data = dataset_type
                    )
    elif dataset_type=='KITTI':
        evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir=cfg['TEST_PATH'],
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    eval_labels='labels.json',
                    num_classes=cfg['MODEL']['N_CLASSES'],
                    batch_size=cfg['TEST']['BATCH_SIZE'],
                    nmsthre=cfg['TEST']['NMSTHRE'],
                    data= dataset_type
                    )
    elif dataset_type=='BDD':
        evaluator = COCOAPIEvaluator(model_type=cfg['MODEL']['TYPE'],
                    data_dir=cfg['TEST_PATH'],
                    img_size=cfg['TEST']['IMGSIZE'],
                    confthre=cfg['TEST']['CONFTHRE'],
                    eval_labels='labels_coco.json',
                    num_classes=cfg['MODEL']['N_CLASSES'],
                    batch_size=cfg['TEST']['BATCH_SIZE'],
                    nmsthre=cfg['TEST']['NMSTHRE'],
                    data= dataset_type
                    )

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    start = time.time()
    ap = evaluator.evaluate(model)
    time_total = time.time() - start
    print("Total Time taken::: ",time_total)
    model.eval()
    sample_metrics = []
    print("ECE computing")
    for img, targets,  info_img, _ in dataloader: 
        target=[]
        for i in range(0,len(targets[0])):
            bbox = targets[0][i][1:].tolist()
            # print("Box::: ",bbox)
            target.append(bbox)
      
        out = [dataset.class_ids[int(x[0])] for x in targets[0].tolist()]
         
        targets =np.array(target,dtype=np.float32)
        labels =np.array(out,dtype=np.int16)
        target_dict =[]
        target_dict.append({"labels":torch.from_numpy(labels),"boxes":torch.from_numpy(targets)} )        
        img = Variable(img.type(dtype))
       
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, cfg['MODEL']['N_CLASSES'],cfg['TEST']['CONFTHRE'],cfg['TEST']['NMSTHRE'])
            if outputs[0] is None:
                continue
            outputs = outputs[0].cpu().data
            output_boxes=[]
            scores_output =[]
            labels_output =[]
            for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                label = dataset.class_ids[int(output[6])]
                box = yolobox2label((y1, x1, y2, x2), info_img)
                bbox = [box[1].item(), box[0].item(), box[3].item() - box[1].item(), box[2].item() - box[0].item()]
                score = float(output[4].data.item() * output[5].data.item())
                output_boxes.append(bbox)
                scores_output.append(score)
                labels_output.append(label)
            boxes_pred =torch.asarray(output_boxes,dtype=torch.float32)
            labels_pred =torch.asarray(labels_output,dtype=torch.int16)
            scores_pred =torch.asarray(scores_output,dtype=torch.float32)
            output_dict=[]
            output_dict.append({'boxes':boxes_pred,'labels':labels_pred,'scores':scores_pred})

            
                            
            sample_metrics += get_batch_statistics(output_dict, target_dict, iou_threshold=cfg['IOU'])
    

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [ np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    target_tensor = torch.from_numpy(np.array(true_positives)).to(torch.int32)
    pred_tensor =torch.from_numpy(np.array(pred_scores)).to(torch.float32)

    ece =calibration_error(preds =pred_tensor,target=target_tensor,n_bins=100)
    print('Ensemble ECE  for '+dataset_type+':::',ece)
        
if __name__ == '__main__':
    main()
