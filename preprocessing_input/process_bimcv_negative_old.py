import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import random
import pickle
import glob

rotate_dict = {}
rotate_dict['sub-S04630_ses-E11071_acq-2_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S05367_ses-E10976_run-1_bp-chest_vp-ap_dx.png'] = 2

# https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def process_xray(data, invert=False, rotate=0):
    if invert:
        data = np.amax(data) - data     
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8) 
    if rotate:
        data = np.ascontiguousarray(np.rot90(data, rotate))      
    return data

# prepare input
image_id_list = []
for root, dirs, files in os.walk('../external_data/BIMCV_negative/BIMCV-COVID19-Negative/COVID19_neg/'):
    for file in files:
        if file.endswith('.png'):
             image_id_list.append(os.path.join(root, file))
print(len(image_id_list))

delete_list = set(pd.read_csv('bimcv_negative_old_delete.csv')['delete_id'].values)
invert_list = set(pd.read_csv('bimcv_negative_old_invert.csv')['invert_id'].values)
print(len(delete_list), len(invert_list))

out_dir = '../processed_input/bimcv_negative_old/train/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for i in tqdm(range(len(image_id_list))):
    if ('.png' in image_id_list[i]) and ('lateral' not in image_id_list[i]) and (image_id_list[i].split('/')[-1] not in delete_list):
        invert = False
        rotate = 0
        img = cv2.imread(image_id_list[i], cv2.IMREAD_UNCHANGED)
        if image_id_list[i].split('/')[-1] in invert_list:
            invert = True
        if image_id_list[i].split('/')[-1] in rotate_dict:
            rotate = rotate_dict[image_id_list[i].split('/')[-1]]
        img = process_xray(img, invert, rotate)
        img = cv2.resize(img, (1280, 1280))
        cv2.imwrite(out_dir+image_id_list[i].split('/')[-1], img)      



