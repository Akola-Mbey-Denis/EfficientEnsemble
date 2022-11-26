from collections import OrderedDict
import torch
import torch.nn.functional as F
# from torchvision.models import  ResNet101_Weights

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
import torch.nn as nn
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Adopted from Pytorch official repo and weight initialisation modified
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes,dropout_precent=0.20,use_dropout=False):
        super().__init__()
        self.drop =dropout_precent
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.activate_dropout =use_dropout       
         

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        if self.activate_dropout:
            scores = F.dropout(self.cls_score(x),p=self.drop,training=True)
            bbox_deltas = F.dropout(self.bbox_pred(x),p=self.drop,training=True)
        else:
            scores  = self.cls_score(x)
            bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes,dropout=0.20,use_dropout=False):
        backbone = resnet_fpn_backbone('resnet101', True)
        super(FRCNN_FPN, self).__init__(backbone, num_classes, box_detections_per_img=300)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None
        self.num_classes =num_classes
        self.drop_out=dropout
        self.activate_dropout =use_dropout
        self.roi_heads.nms_thresh = 0.50
        if self.activate_dropout:
            self.in_features = self.roi_heads.box_predictor.cls_score.in_features
            self.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes,self.drop_out,self.activate_dropout)

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach(), detections['labels'].detach()

    def get_proposal(self):
        'Proposals return from this end is shared by all models in the ensemble'
        proposals, _ =self.rpn(images=self.preprocessed_images,features=self.features,targets=None)
        return self.features, proposals
    
    def FastRCNN_prediction_head(self,proposals):
        'This method passes the proposals to the Fast RCNN layer of each of the other models in the ensemble'
        detections, _ =self.roi_heads(features=self.features,proposals=proposals,  image_shapes=self.preprocessed_images.image_sizes, targets=None)
        detections = self.transform.postprocess(result=detections,image_shapes=self.preprocessed_images.image_sizes,
                                         original_image_sizes=self.original_image_sizes)
        return detections[0]['boxes'].detach(), detections[0]['scores'].detach(), detections[0]['labels'].detach()
   
 
    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
