import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Normalize
from utils.imutils import crop


def imshow(ax, img):
    ax.imshow(img[:, :, [2, 1, 0]])
    ax.axis("off")
    plt.draw()
    plt.pause(1e-3)

def select_humans(predictions):
    labels = predictions.get_field("labels").tolist()
    boxes = predictions.bbox.tolist()

    human_boxes = [boxes[i] for i in range(len(labels)) if labels[i] == 1]
    human_boxes = torch.Tensor(human_boxes).to(torch.int64)

    return human_boxes

def process_image(img, human_bboxes, img_norm_mean, img_norm_std, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=img_norm_mean, std=img_norm_std)
    img = img[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    det_stack = []
    norm_stack = []

    for bbox in human_bboxes:
        # assuming format xyxy
        ### quickfix copy_paste: change from torch to numpy and back again. not necessary obiously
        det_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # Assume that the person is centerered in the image
        height = det_crop.shape[0]
        width = det_crop.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200

        det_crop = crop(det_crop, center, scale, (input_res, input_res))
        det_crop = det_crop.astype(np.float32) / 255.
        det_crop = torch.from_numpy(det_crop).permute(2,0,1)
        norm_crop = normalize_img(det_crop.clone())[None]

        det_stack.append(det_crop)
        norm_stack.append(norm_crop)

    return det_stack, norm_stack
