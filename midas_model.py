import cv2
import torch
import urllib.request
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import LoadImagesFromFolder




    


def midas(model_type, images):
    
    #creating midas model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #keeping it in eval as we're not training
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True).to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    #loading Images
    depth = []
    
    #sending images through the model
    for img in images:
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        max = np.max(output)
        output = 100 - (100*output/max)
        depth.append(output)
        
    return depth, images
    
    