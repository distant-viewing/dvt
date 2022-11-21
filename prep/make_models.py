##############################################################################
# Load all required modules here
import torch
import torchvision
from torch.jit import trace
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

import torch.nn as nn
import torchvision.transforms.functional as F

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


##############################################################################
# 0. Load a sample image so that we may trace it
img_input = read_image("dancers.jpg")
img = F.convert_image_dtype(img_input, dtype=torch.float)
img = img.unsqueeze(0) 


##############################################################################
# 1. KeyPoint Detection with keypointrcnn_resnet50_fpn
weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
categories = weights.value.meta['categories']
model = keypointrcnn_resnet50_fpn(weights=weights)
model.eval()
out = model(img)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = model
    def forward(self, x):
        return (
            self.model(x)[0]['boxes'],
            self.model(x)[0]['labels'],
            self.model(x)[0]['scores'],
            self.model(x)[0]['keypoints'],
            self.model(x)[0]['keypoints_scores']
        )


instance = MyModel()
module = torch.jit.trace(instance, img)
module.save("../models/dvt_detect_keypoint.pt")


##############################################################################
# 2. Segmentation with deeplabv3_resnet50
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
categories = weights.value.meta['categories']
model = deeplabv3_resnet50(weights=weights)
model.eval()
out = model(img)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']


instance = MyModel()
module = torch.jit.trace(instance, img)
module.save("../models/dvt_segment.pt")


##############################################################################
# 3. Image embedding
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = efficientnet_v2_s(weights=weights)
model.eval()

import torch 

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = model
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.model.classifier(x)
        return x


instance = MyModel()
module = torch.jit.trace(instance, img)
module.save("../models/dvt_embed.pt")


##############################################################################
# 4. Face Detection
import torch
from dvt_face import MTCNN

# Load and save MTCNN
mtcnn = MTCNN(
    model_path_p = "data/pnet.pt",
    model_path_r = "data/rnet.pt",
    model_path_o = "data/onet.pt"
)
torch.save(mtcnn.state_dict(), "../models/dvt_detect_face.pt")

