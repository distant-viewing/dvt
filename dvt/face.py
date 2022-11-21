# -*- coding: utf-8 -*-
"""Annotators for detecting and recognizing faces.
"""

import cv2
import numpy as np
import torch

from torch import nn
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms

from .utils import _download_file


class AnnoFaces:
    """Detect faces in an image and embed into a space for recognition.

    Uses the MTCNN algorithm to detect faces and FaceNet to do the face
    embedding algorithm.

    Attributes:
        model_path_mtcnn (str): location of the saved pytorch weight for the
            detection algorithm; set to None to download from the DVT repository.
        model_path_vggface (str): location of the saved pytorch weight for the
            embedding algorithm; set to None to download from the DVT repository.
    """
    def __init__(self, model_path_mtcnn=None, model_path_vggface=None):

        if not model_path_mtcnn:
            model_path_mtcnn = _download_file("dvt_detect_face.pt")

        if not model_path_vggface:
            model_path_vggface = _download_file("dvt_face_embed.pt")

        self.mtcnn = _MTCNN()
        self.mtcnn.load_state_dict(torch.load(model_path_mtcnn))
        self.mtcnn.eval()

        self.vggface = InceptionResnetV1()
        self.vggface.load_state_dict(torch.load(model_path_vggface))
        self.vggface.eval()

    def run(self, img: np.ndarray, visualize=False, embed=True) -> dict:
        """Run the annotator.

        Args:
            img (np.ndarray): image file, as returned by dvt.load_image,
                or dvt.yield_video (for frame from a video file).
            visualize (bool): should a visualization of the output be
                returned as well? Default is False.
            embed (bool): should an embedding of the faces be returned?
                Default is True.

        Returns:
            A dictionary giving the annotation.
        """
        img_t = np.transpose(img, (2, 0, 1))
        img_t = torch.Tensor(img_t).unsqueeze(0)

        # detect faces
        batch_boxes, batch_points = self.mtcnn(img_t)
        if not len(batch_boxes[0]):
            return {}

        # take subsets of the image corresponding to each face
        faces = [_extract_face(img, x) for x in batch_boxes[0]]
        faces = [np.transpose(x, (2, 0, 1)) for x in faces]
        faces = np.stack(faces)
        faces = (faces - 127.5) / 128.0
        faces = torch.Tensor(faces)

        # embed the faces
        if embed:
            vects = self.vggface(faces).detach().numpy()
            norms = np.linalg.norm(vects, axis=1, ord=2, keepdims=True)
            vects = vects / norms
        else:
            vects = None

        # clean up
        nface = batch_boxes.shape[1]
        boxes = {
            "face_id": np.arange(nface),
            "x": np.int32(batch_boxes[0][:, 0]),
            "xend": np.int32(batch_boxes[0][:, 2]),
            "y": np.int32(batch_boxes[0][:, 1]),
            "yend": np.int32(batch_boxes[0][:, 3]),
            "prob": batch_boxes[0][:, 4],
        }
        kpnts = {
            "face_id": np.repeat(np.arange(nface), 5),
            "kpnt_id": np.tile(np.arange(5), nface),
            "x": np.int32(batch_points[0][:, :, 0].flatten()),
            "y": np.int32(batch_points[0][:, :, 1].flatten()),
        }

        # visualize?
        if visualize:
            img_out = img.copy()
            yellow = [250, 189, 47]
            for index in range(nface):
                img_out[
                    boxes["y"][index], boxes["x"][index]:boxes["xend"][index], :
                ] = yellow
                img_out[
                    boxes["yend"][index], boxes["x"][index]:boxes["xend"][index], :
                ] = yellow
                img_out[
                    boxes["y"][index]:boxes["yend"][index], boxes["x"][index], :
                ] = yellow
                img_out[
                    boxes["y"][index]:boxes["yend"][index], boxes["xend"][index], :
                ] = yellow
        else:
            img_out = None

        return {"boxes": boxes, "kpnts": kpnts, "embed": vects, "img": img_out}


#################################################################################
# All of the code below is adapted from the facenet-pytorch library, which is 
# covered under the MIT License, Copyright (c) 2019 Timothy Esler.

def _fixed_batch_process(im_data, model):
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i : (i + batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))


def _detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor):

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []

    scale_picks = []

    all_i = 0
    offset = 0
    for scale in scales:
        im_data = _imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)

        boxes_scale, image_inds_scale = _generateBoundingBox(
            reg, probs[:, 1], scale, threshold[0]
        )
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)

    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = _rerec(boxes)
    y, ey, x, ex = _pad(boxes, w, h)

    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[
                    image_inds[k], :, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]
                ].unsqueeze(0)
                im_data.append(_imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
        out = _fixed_batch_process(im_data, rnet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = _bbreg(boxes, mv)
        boxes = _rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2)
    if len(boxes) > 0:
        y, ey, x, ex = _pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[
                    image_inds[k], :, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]
                ].unsqueeze(0)
                im_data.append(_imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        out = _fixed_batch_process(im_data, onet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = _bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        pick = _batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, "Min")
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    image_inds = image_inds.cpu()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    return batch_boxes, batch_points


def _bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


def _generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def _nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def _batched_nms_numpy(boxes, scores, idxs, threshold, method):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = _nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long)


def _pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def _rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


def _imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def _crop_resize(img: np.ndarray, box, image_size):
    img = img[box[1] : box[3], box[0] : box[2]]
    out = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA).copy()
    return out


def _extract_face(img, box, image_size=160, margin=0):
    """Extract face + margin given bounding box."""
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = img.shape[1::-1]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]
    face = _crop_resize(img, box, image_size)

    return face


class _PNet(nn.Module):
    """MTCNN PNet."""

    def __init__(self, model_path):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False
        if model_path:
            state_dict = torch.load(model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class _RNet(nn.Module):
    """MTCNN RNet."""

    def __init__(self, model_path):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False
        if model_path:
            state_dict = torch.load(model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class _ONet(nn.Module):
    """MTCNN ONet."""

    def __init__(self, model_path):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False
        if model_path:
            state_dict = torch.load(model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class _MTCNN(nn.Module):
    """MTCNN face detection module."""

    def __init__(
        self,
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        model_path_p=None,
        model_path_r=None,
        model_path_o=None,
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor

        self.pnet = _PNet(model_path_p)
        self.rnet = _RNet(model_path_r)
        self.onet = _ONet(model_path_o)

    def forward(self, img):

        # Detect Faces
        with torch.no_grad():
            batch_boxes, batch_points = _detect_face(
                img,
                self.min_face_size,
                self.pnet,
                self.rnet,
                self.onet,
                self.thresholds,
                self.factor,
            )

        return batch_boxes, batch_points


class _BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = _BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            _BasicConv2d(256, 32, kernel_size=1, stride=1),
            _BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            _BasicConv2d(256, 32, kernel_size=1, stride=1),
            _BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            _BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class _Block17(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = _BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            _BasicConv2d(896, 128, kernel_size=1, stride=1),
            _BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            _BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class _Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = _BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            _BasicConv2d(1792, 192, kernel_size=1, stride=1),
            _BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            _BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = _BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            _BasicConv2d(256, 192, kernel_size=1, stride=1),
            _BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            _BasicConv2d(192, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            _BasicConv2d(896, 256, kernel_size=1, stride=1),
            _BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            _BasicConv2d(896, 256, kernel_size=1, stride=1),
            _BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.Sequential(
            _BasicConv2d(896, 256, kernel_size=1, stride=1),
            _BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            _BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights."""

    def __init__(
        self,
        pretrained=None,
        classify=False,
        num_classes=None,
        dropout_prob=0.6,
        device=None,
    ):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        # Define layers
        self.conv2d_1a = _BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = _BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = _BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = _BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = _BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = _BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            _Block35(scale=0.17),
            _Block35(scale=0.17),
            _Block35(scale=0.17),
            _Block35(scale=0.17),
            _Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
            _Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            _Block8(scale=0.20),
            _Block8(scale=0.20),
            _Block8(scale=0.20),
            _Block8(scale=0.20),
            _Block8(scale=0.20),
        )
        self.block8 = _Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        self.logits = nn.Linear(512, 8631)

    def forward(self, x):
        """Calculate embeddings."""
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)

        return x
