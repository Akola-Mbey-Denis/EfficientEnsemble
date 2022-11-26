import math
import sys
import time

import torch
import torchvision
import utils.train_utils as utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from torchvision.ops import nms
import pandas as pd
import statistics
import numpy as np
import torch.distributed as dist
from torchmetrics.functional import calibration_error
from utils.calibration_utils import get_batch_statistics
import torch.distributed as dist
from mmcv.ops.nms import nms_match, soft_nms
from ensemble_suite import NaiveAveragger,weighted_boxes_fusion
from torch.nn.functional import normalize
def is_dist_avail_and_initialized():
  
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    train_loss=None, train_lr=None, warmup=False):
    global loss_dict, losses
    model.train()
    metric_logger =utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    loss_max =0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if isinstance(train_loss, list):
            train_loss.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        if isinstance(train_lr, list):
            train_lr.append(now_lr)

    return loss_dict, losses


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types    

@torch.inference_mode()
def evaluate(model, data_loader, device,IOU):
    sample_metrics=[]
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    start = time.time()
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=IOU)

        outputs =  [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    time_total = time.time() - start
    print("Total Time taken::: ",time_total)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [ np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    target_tensor = torch.from_numpy(np.array(true_positives)).to(torch.int32)
    pred_tensor =torch.from_numpy(np.array(pred_scores)).to(torch.float32)
    ece =calibration_error(preds =pred_tensor,target=target_tensor,n_bins=100)
    print('Model  ECE for ',ece)
    return coco_evaluator

@torch.inference_mode()
def evaluate_montecarlo(model, data_loader, device,IOU,NMS,ensembleMethod,NMS_TYPE):
    sample_metrics=[]
    outputs_list=[]
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    start = time.time()
    for images, targets in metric_logger.log_every(data_loader, 1, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs_list = [model(images) for i in range(3)]
        #First dropout model
        boxes1 =outputs_list[0][0]['boxes']
        scores1 =outputs_list[0][0]['scores']
        labels1 = outputs_list[0][0]['labels']
        #Second dropout model
        boxes2 =outputs_list[1][0]['boxes']
        scores2 =outputs_list[1][0]['scores']
        labels2 = outputs_list[1][0]['labels']
        #Third dropou model 
        boxes3=outputs_list[2][0]['boxes']
        scores3 =outputs_list[2][0]['scores']
        labels3 = outputs_list[2][0]['labels']
        if ensembleMethod!='wbf':
                boxes =torch.cat((boxes1,boxes2,boxes3),dim=0)
                scores =torch.cat((scores1,scores2,scores3),dim=0)
                labels_t =torch.cat((labels1,labels2,labels3),dim=0)
                scores,boxes, vars_,labels = NaiveAveragger(boxes=boxes.tolist(),scores=scores.tolist(),labels=labels_t.tolist(),iou_threshold=NMS,nms_method=NMS_TYPE)
                outputs=[]
                outputs.append({'boxes':boxes.to(device),'scores':scores.to(device),'labels':labels.to(device)})
        else:
            boxes_list  = [boxes1.tolist(),boxes2.tolist(),boxes3.tolist()]
            scores_list = [scores1.tolist(),scores2.tolist(),scores3.tolist()]
            labels_list = [labels1.tolist(),labels2.tolist(),labels3.tolist()]
            boxes, scores,labels,variance = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=NMS, skip_box_thr=0.20)
            new_boxes = np.array(boxes,dtype=np.float32)
            new_scores =np.array(scores,dtype=np.float32)
            new_labels =np.array(labels,dtype=np.int16)
            new_scores =np.array(new_scores,dtype=np.float32) 
            boxes =torch.from_numpy(new_boxes)
            scores=torch.from_numpy(new_scores)
            labels_t =torch.from_numpy(new_labels)
            outputs=[]
            outputs.append({'boxes':boxes.to(device),'scores': scores.to(device),'labels':labels_t.to(device)})  
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=IOU)

        outputs =  [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    time_total = time.time() - start
    print("Total Time taken::: ",time_total)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [ np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    target_tensor = torch.from_numpy(np.array(true_positives)).to(torch.int32)
    pred_tensor =torch.from_numpy(np.array(pred_scores)).to(torch.float32)
    ece =calibration_error(preds =pred_tensor,target=target_tensor,n_bins=100)
    print('Model  ECE for ',ece)
    return coco_evaluator

@torch.no_grad()
def evaluate_map(model, data_loader, device, mAP_list=None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger =utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    print_txt = coco_evaluator.coco_eval[iou_types[0]].stats
    coco_mAP = print_txt[0]
    voc_mAP = print_txt[1]
    if isinstance(mAP_list, list):
        mAP_list.append(voc_mAP)

    return coco_evaluator, voc_mAP
@torch.no_grad()
def evaluate_triple(model1,model2,model3, data_loader, device,ensemble_type,IOU,NMS,ensembleMethod,weights=None,NMS_TYPE='soft'):
    n_threads = torch.get_num_threads()
    sample_metrics = []
   
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model1.eval()
    model2.eval()
    model3.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model1)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    model_time =0
    if ensemble_type =='Single':
        start =time.time()
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            img = images[0].unsqueeze_(0)  
            model1.load_image(img)                  
            model2.load_image(img)
            model3.load_image(img)           
            features, proposals = model1.get_proposal()
            boxes1,scores1,labels1 =model1.FastRCNN_prediction_head(proposals)
            boxes2,scores2,labels2 = model2.FastRCNN_prediction_head(proposals)
            boxes3,scores3,labels3 = model3.FastRCNN_prediction_head(proposals)
            if ensembleMethod!='wbf':
                boxes =torch.cat((boxes1,boxes2,boxes3),dim=0)
                scores =torch.cat((scores1,scores2,scores3),dim=0)
                labels_t =torch.cat((labels1,labels2,labels3),dim=0)
                scores,boxes, vars_,labels = NaiveAveragger(boxes=boxes.tolist(),scores=scores.tolist(),labels=labels_t.tolist(),iou_threshold=NMS,nms_method=NMS_TYPE)
                outputs=[]
                outputs.append({'boxes':boxes.to(device),'scores':scores.to(device),'labels':labels.to(device)})
            else:
                boxes_list  = [boxes1.tolist(),boxes2.tolist(),boxes3.tolist()]
                scores_list = [scores1.tolist(),scores2.tolist(),scores3.tolist()]
                labels_list = [labels1.tolist(),labels2.tolist(),labels3.tolist()]
                boxes, scores,labels,variance = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=NMS, skip_box_thr=0.20)
                new_boxes = np.array(boxes,dtype=np.float32)
                new_scores =np.array(scores,dtype=np.float32)
                new_labels =np.array(labels,dtype=np.int16)
                new_scores =np.array(new_scores,dtype=np.float32) 
                boxes =torch.from_numpy(new_boxes)
                scores=torch.from_numpy(new_scores)
                labels_t =torch.from_numpy(new_labels)
                outputs=[]
                outputs.append({'boxes':boxes.to(device),'scores': scores.to(device),'labels':labels_t.to(device)})  
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=IOU)
            outputs =  [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        
        # gather the stats from all processes
        time_total = time.time() - start
        print("Total Time taken::: ",time_total)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()           

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [ np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        target_tensor = torch.from_numpy(np.array(true_positives)).to(torch.int32)
        pred_tensor =torch.from_numpy(np.array(pred_scores)).to(torch.float32)
        ece =calibration_error(preds =pred_tensor,target=target_tensor,n_bins=100)
        print('Model  ECE for ',ece)
        return coco_evaluator
    elif ensemble_type=='Triple':
        start =time.time()
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            img = images[0].unsqueeze_(0)
            model2.load_image(img)
            model3.load_image(img)
            boxes1,scores1,labels1 =model1.detect(img) 
            boxes2,scores2,labels2=model2.detect(img)
            boxes3,scores3,labels3 =model3.detect(img)
            if ensembleMethod !='wbf':
                boxes =torch.cat((boxes1,boxes2,boxes3),dim=0)
                scores =torch.cat((scores1,scores2,scores3),dim=0)
                labels_t =torch.cat((labels1,labels2,labels3),dim=0)
                scores,boxes, vars_,labels = NaiveAveragger(boxes=boxes.tolist(),scores=scores.tolist(),labels=labels_t.tolist(),iou_threshold=NMS,nms_method=NMS_TYPE)
                outputs=[]
                outputs.append({'boxes':boxes.to(device),'scores':scores.to(device),'labels':labels.to(device)})
            else:
                boxes_list =[boxes1.tolist(),boxes2.tolist(),boxes3.tolist()]
                scores_list =[scores1.tolist(),scores2.tolist(),scores3.tolist()]
                labels_list  =[labels1.tolist(),labels2.tolist(),labels3.tolist()]
                boxes, scores, labels,variance = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=NMS, skip_box_thr=0.20)
                new_boxes =np.array(boxes,dtype=np.float32)
                new_scores =np.array(scores,dtype=np.float32)
                new_labels =np.array(labels,dtype=np.int16)
                new_scores =np.array(new_scores,dtype=np.float32) 
                boxes =torch.from_numpy(new_boxes)
                scores=torch.from_numpy(new_scores)
                labels_t =torch.from_numpy(new_labels)                   
                outputs=[]
                outputs.append({'boxes':boxes.to(device),'scores':scores.to(device),'labels':labels_t.to(device)})
            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=IOU)
            outputs =  [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        #gather the stats from all processes
        time_total = time.time() - start
        print("Total Time taken::: ",time_total)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [ np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        target_tensor = torch.from_numpy(np.array(true_positives)).to(torch.int32)
        pred_tensor =torch.from_numpy(np.array(pred_scores)).to(torch.float32)
        ece =calibration_error(preds =pred_tensor,target=target_tensor,n_bins=100)
        print('Model  ECE for  :::',ece)
        return coco_evaluator
