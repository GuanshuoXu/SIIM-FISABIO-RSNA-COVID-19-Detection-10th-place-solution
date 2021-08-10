import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import random
import pickle
import glob

rotate_dict = {}
rotate_dict['sub-S09522_ses-E18974_run-1_bp-chest_vp-pa_dx.png'] = 2
rotate_dict['sub-S09651_ses-E19665_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09789_ses-E18888_run-2_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S03068_ses-E06592_run-1_bp-chest_vp-ap_cr.png'] = 2
rotate_dict['sub-S03936_ses-E08636_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S04190_ses-E08730_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S04275_ses-E08736_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S08047_ses-E16837_run-1_bp-chest_cr.png'] = 2
rotate_dict['sub-S08047_ses-E17511_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09458_ses-E16829_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09458_ses-E21702_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09468_ses-E21004_acq-1_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09468_ses-E21414_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09502_ses-E19994_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S09745_ses-E19438_run-1_bp-chest_vp-ap_dx.png'] = 3
rotate_dict['sub-S10095_ses-E19759_run-1_bp-chest_dx.png'] = 2
rotate_dict['sub-S10310_ses-E18298_acq-1_run-1_bp-chest_vp-ap_dx.png'] = 1
rotate_dict['sub-S10373_ses-E21737_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10377_ses-E21272_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10389_ses-E21478_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10389_ses-E21806_acq-2_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10395_ses-E20973_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10395_ses-E21397_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10405_ses-E21428_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10410_ses-E21444_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10411_ses-E19088_run-1_bp-chest_vp-ap_dx.png'] = 3
rotate_dict['sub-S10411_ses-E21768_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10437_ses-E19015_acq-2_run-1_bp-chest_vp-ap_dx.png'] = 1
rotate_dict['sub-S10540_ses-E22903_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10576_ses-E21051_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S10753_ses-E19892_run-1_bp-chest_vp-pa_cr.png'] = 2
rotate_dict['sub-S10809_ses-E21009_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S11310_ses-E23088_run-1_bp-chest_vp-ap_dx.png'] = 2
rotate_dict['sub-S11766_ses-E22231_run-1_bp-chest_vp-ap_dx.png'] = 2

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
import pickle
df = pd.read_csv('../external_data/BIMCV/BIMCV-COVID19-cIter_1_2/covid19_posi/derivatives/partitions.tsv', sep=' |\t')
image_id_list = df['filepath'].values
print(len(image_id_list))

delete_list = set(pd.read_csv('bimcv_delete.csv')['delete_id'].values)
invert_list = set(pd.read_csv('bimcv_invert.csv')['invert_id'].values)
print(len(delete_list), len(invert_list))

out_dir = '../processed_input/bimcv/train/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for i in tqdm(range(len(image_id_list))):
    if ('.png' in image_id_list[i]) and ('lateral' not in image_id_list[i]) and (image_id_list[i].split('/')[-1] not in delete_list):
        invert = False
        rotate = 0
        img = cv2.imread('../external_data/BIMCV/BIMCV-COVID19-cIter_1_2/covid19_posi'+image_id_list[i][1:], cv2.IMREAD_UNCHANGED)
        if image_id_list[i].split('/')[-1] in invert_list:
            invert = True
        if image_id_list[i].split('/')[-1] in rotate_dict:
            rotate = rotate_dict[image_id_list[i].split('/')[-1]]
        img = process_xray(img, invert, rotate)
        img = cv2.resize(img, (1280, 1280))
        cv2.imwrite(out_dir+image_id_list[i].split('/')[-1], img)      



