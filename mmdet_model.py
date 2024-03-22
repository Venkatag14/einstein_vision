from mmdet.apis import DetInferencer
import torch
from midas_model import LoadImagesFromFolder
import numpy as np
import cv2



def mmdet(images):
    
    Device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    obj_det = []
    
    inferencer = DetInferencer(model = 'rtmdet_tiny_8xb32-300e_coco', device = 'cpu')
    for i,img in enumerate(images):
        result = inferencer(img, show=True)
        # result_img = np.array(result['visualization'])
        # if result_img.size!=0:
        #     cv2.imwrite(f"scene4/front_frames/yolo{i}.jpg", result_img)
        obj_det.append(result)
        
    return obj_det

# imgs = LoadImagesFromFolder("scene4/front_frames/")

# bounding_boxes = mmdet([imgs[-1]])

# print(bounding_boxes)

