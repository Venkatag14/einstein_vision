from midas_model import midas
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
import json
from mmdet_model import mmdet
from dataloaders import LoadImagesFromFolder, read_names_file
from points import xyz
from traffic_color import signal



# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Plot RGB image
# axes[0].imshow(rgb_img)
# axes[0].set_title('RGB Image')

# # Plot depth image
# axes[1].imshow(depth_img, cmap='gray')
# axes[1].set_title('Depth Image')

# # Show plot
# # plt.imshow(depth_img, cmap='gray')
# plt.show()

# implementing mmdet

# implementing midas

img_indx = 23

images = LoadImagesFromFolder("scene4/front_frames/")
depth_images = midas("DPT_Large", [images[img_indx]])
randint = random.randint(0, len(depth_images) - 1)

depth_img = depth_images[0]
depth_img = np.array(depth_img)
depth_img = depth_img.T
shape = depth_img.shape
rgb_img = images[img_indx]

#k extracted from calibration.py
k = np.array([[1.60032130e+03, 0.00000000e+00, 6.37476366e+02],
             [0.00000000e+00, 1.61379527e+03, 4.24541585e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

#assuming car camera at 1.5m height in world frame

R = np.array([[-1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 1.5],
             [0, 0, 0, 1]])



labels = read_names_file("coco.names")

bounding_boxes = mmdet([images[img_indx]])

pred_scores = bounding_boxes[0]['predictions'][0]['scores']
pred_labels = bounding_boxes[0]['predictions'][0]['labels']
pred_bound_boxes = bounding_boxes[0]['predictions'][0]['bboxes']

json_dict = {}

for index,score in enumerate(pred_scores):
    if score > 0.33:
        label_ind = pred_labels[index]
        label = labels[label_ind]
        box = pred_bound_boxes[index]
        if label == 'car' or label == 'truck' or label == 'bus' or label == 'motorcycle' or label == 'bicycle' or label == 'person'  or label == 'stop sign':
            u,v = (box[0]+box[2])/2, (box[1]+box[3])/2
            depth = depth_img[int(u)][int(v)]
            XYZ = xyz(R, k, (u,v), depth)
            T = np.eye(4)
            T[:3,3] = XYZ
            T = T.tolist()
            if label in json_dict:
                # If the key exists, append the value to the existing list
                json_dict[label].append(T)
            else:
                # If the key doesn't exist, create a new key-value pair with a list containing the value
                json_dict[label] = [T]
                
        if label == 'traffic light':
            u,v = (box[0]+box[2])/2, (box[1]+box[3])/2
            depth = depth_img[int(u)][int(v)]
            XYZ = xyz(R, k, (u,v), depth)
            T = np.eye(4)
            T[:3,3] = XYZ
            T = T.tolist()
            color = signal(images[img_indx], box)
            value = [T, color]
            if label in json_dict:
                # If the key exists, append the value to the existing list
                json_dict[label].append(value)
            else:
                # If the key doesn't exist, create a new key-value pair with a list containing the value
                json_dict[label] = [value]
                
                
json_filename = f"json_output{img_indx}.json"
with open(json_filename, "w") as json_file:
    json.dump(json_dict, json_file)
            
            
            
            
            




