import torch
from mmcv.ops.nms import nms_match, soft_nms,nms

def NaiveAveragger(boxes,scores,labels,iou_threshold,nms_method="soft"):
    """This function merges the output of an ensemble by
    iterating through the boxes and averaging boxes that 
    have an overlap equal to the IOU 

     Args:
        boxes:  (Tensor or list): Predicted bounding boxes
        scores: (Tensor or list): Predicted scores
        iou_threshold : mininum acceptable IOU 
    Output:
    average boxes: with an added parameter 
    variance: Variance associated with the averaged boxes
    Average scores: the mean scores of the boxes
    labels: Associated box labels
    
    """
    if nms_method=='soft':
        new_dets, inds = soft_nms(np.array(boxes).astype(np.float32), np.array(scores).astype(np.float32),  iou_threshold, method= 'gaussian',sigma=0.1,min_score=0.20)
        matches = [np.array([ind]) for ind in inds]

        
    elif nms_method =='match':
        X=np.array(boxes)
        scores =np.array(scores).reshape(-1,1)
        boxes=np.column_stack((X, scores)).astype(np.float32)
 
        matches = nms_match(boxes, iou_threshold) 
      
       
    elif nms_method=='nms':
        new_dets, inds = nms(np.array(boxes).astype(np.float32), np.array(scores).astype(np.float32), iou_threshold)
        matches = [np.array([ind]) for ind in inds]
        
        
    box_list = []
    label_list = []
    var_list = []
    scores_list=[]
    if nms_method=='match':
        for match in matches:
            # match[0]:the index of the first box in each matched group, new_boxes are the corresponding box sublist from
            # boxes with indices 'match' in score order.
            # new_boxes: [[x1,y1,x2,y2,s1], [x1,y1,x2,y2,s2], [x1,y1,x2,y2,s3],...]
    
            new_boxes = [boxes[match[0]][:4]]
        
            new_scores =[boxes[match[0]][4:]]
            label = labels[match[0]]
            for id in match[1:]:  
                # the index of remaining boxes in that matched group
                if label == labels[id]:
                    new_boxes.append(boxes[id][:4])
                    new_scores.append(boxes[id][4:])

            #Compute the variance associated with the similar bounding boxes   
            
            vars_ = [np.var([new_box[i] for new_box in new_boxes]) for i in range(4)]
       
            #Compute average bounding boxes
            merged_box = np.mean(new_boxes, axis=0)
   
            #Compute average scores
            merged_score =  np.max(new_scores,axis=0)
 
            #Stack average bounding boxes with associated variance
          
            box_list.append(merged_box.tolist())
            scores_list.append(merged_score.tolist()[0])
            label_list.append(label)
            var_list.append(vars_)
            
    else:
        for match in matches:
            # match[0]:the index of the first box in each matched group, new_boxes are the corresponding box sublist from
            # boxes with indices 'match' in score order.
            # new_boxes: [[x1,y1,x2,y2,s1], [x1,y1,x2,y2,s2], [x1,y1,x2,y2,s3],...]

            new_boxes = [boxes[match[0]]]
        
            new_scores =[scores[match[0]]]
            label = labels[match[0]]
            for id in match[1:]:  
                # the index of remaining boxes in that matched group
                if label == labels[id]:
                    new_boxes.append(boxes[id])
                    new_scores.append(scores[id])

            #Compute the variance associated with the similar bounding boxes   
          
            vars_ = [np.var([new_box[i] for new_box in new_boxes]) for i in range(4)]
            #Compute average bounding boxes
            merged_box = np.mean(new_boxes, axis=0)
           
            #Compute average scores
            merged_score =np.mean(new_scores,axis=0)
        
            box_list.append(merged_box.tolist())
            scores_list.append(merged_score.tolist())
            label_list.append(label)
            var_list.append(vars_)
            #Stack average bounding boxes with associated variance    

    new_boxes =np.array(box_list,dtype=np.float32)
   
    new_scores =np.array(scores_list,dtype=np.float32) 
    labels = np.array(label_list,dtype=np.int16)  
    variance = np.array(var_list,dtype=np.float32)     
    return torch.from_numpy(new_scores), torch.from_numpy(new_boxes),torch.from_numpy(variance), torch.from_numpy(labels)

# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import warnings
import numpy as np


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2,0.0,0.0,0.0,0.0]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """

    box = np.zeros(12, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    vars_ = [np.var([new_box[4+i] for new_box in boxes]) for i in range(4)]
    for b in boxes:
        box[4:8] += (b[1] * b[4:8])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1 # model index field is retained for consistency but is not used.
    box[4:] /= conf
    box[8:]=vars_
  
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """
    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    # print('Old box ----',boxes[:, 4:8])
    # print('New box ----',new_box[4:8])

    ious = bb_iou_array(boxes[:, 4:8], new_box[4:8])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.00,
        conf_type='box_and_model_avg',
        allows_overflow=False
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    :return: variance: box variance
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    # print('Filtered boxes:::',filtered_boxes)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)),np.zeros((0,4))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 12))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            # print('Weigthed boxes ',weighted_boxes)
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
     
    boxes = overall_boxes[:,4:8]
   
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    vars_ =overall_boxes[:,8:12]
    return boxes, scores, labels,vars_