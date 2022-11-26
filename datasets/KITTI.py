from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms as T
class KITTIObjectDetect(Dataset):
    '''
    https://github.com/AlphaJia/pytorch-faster-rcnn

    '''
    def __init__(self, root_dir, image_set, transforms=None):

        self._root_dir = root_dir
        self._image_set = image_set
        self._data_name = image_set
        self._json_path = self._get_ann_file()
        self.transform =transforms

        # load COCO API
        self._COCO = COCO(self._json_path)

        with open(self._json_path) as anno_file:
            self.anno = json.load(anno_file)

        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
    
       

        self.classes = self._classes
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                                   self._COCO.getCatIds())))

        self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                               self._class_to_ind[cls])
                                              for cls in self._classes[1:]])

    def __len__(self):
        return len(self.anno['images'])

    def _get_ann_file(self):
        return os.path.join(self._root_dir,self._image_set, 'labels.json')

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        file_name = (str(index).zfill(6) + '.png')

        image_path = os.path.join(self._root_dir, self._data_name, 'data',file_name)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def __getitem__(self, idx):
        a = self.anno['images'][idx]
        image_idx = a['id']
        
        image_idx_file =image_idx-1
        img_path =  self._image_path_from_index(image_idx_file)
        image = Image.open(img_path).convert("RGB")
        #image =T.ToTensor()(image)

        width = a['width'] 
        height = a['height']

        annIds = self._COCO.getAnnIds(imgIds=image_idx, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)

        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        iscrowd = []
        for ix, obj in enumerate(objs):
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            iscrowd.append(int(obj["iscrowd"]))

        # convert everything into a torch.Tensor
        image_id = torch.tensor([image_idx])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": gt_classes, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transform is not None:
            img= self.transform(image)

        return img, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @property
    def class_to_coco_cat_id(self):
        return self._class_to_coco_cat_id