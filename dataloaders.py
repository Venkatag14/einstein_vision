import cv2
from os import listdir
import numpy as np


def read_names_file(file_path):
    class_names = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading and trailing whitespaces, and append to class_names list
            class_names.append(line.strip())
    return class_names

def LoadImagesFromFolder(folder):
    images = []
    img_files = listdir(folder)
    sorted_file_names = sorted(img_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    for file in sorted_file_names:
        img = cv2.imread(folder + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images