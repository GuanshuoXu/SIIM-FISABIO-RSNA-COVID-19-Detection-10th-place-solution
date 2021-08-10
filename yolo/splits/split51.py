import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
import json
import glob
import ast
from shutil import copyfile

image_list = sorted(glob.glob('../../input/train/*/*/*.dcm'))
study_id_list = []
studyid2imageid = {}
imageid2studyid = {}
label_dict = {}
for i in range(len(image_list)):
    study_id = image_list[i].split('/')[-3]
    image_id = image_list[i].split('/')[-1][:-4]
    if study_id not in studyid2imageid:
        study_id_list.append(study_id)
        studyid2imageid[study_id] = [image_id]
    else:
        studyid2imageid[study_id].append(image_id)
    imageid2studyid[image_id] = study_id
    label_dict[image_id] = {
        'bbox_list': [], 
        'original_bbox_list': [], 
        'label_list': [],
        'study_label': -1,
        'is_none': False,
        'img_dir': '../../processed_input/input/train/'+image_id+'.png',
    }  
print(len(image_list), len(study_id_list))

df_study = pd.read_csv('../../input/train_study_level.csv')
study_id_list1 = df_study['id'].values
negative_list = df_study['Negative for Pneumonia'].values
typical_list = df_study['Typical Appearance'].values
indeterminate_list = df_study['Indeterminate Appearance'].values
atypical_list = df_study['Atypical Appearance'].values
for i in range(len(study_id_list1)):
    study_id = study_id_list1[i].split('_')[0]
    if negative_list[i] == 1:
        study_label = 3
    elif typical_list[i] == 1:
        study_label = 4    
    elif indeterminate_list[i] == 1:
        study_label = 5  
    elif atypical_list[i] == 1:
        study_label = 6  
    for image_id in studyid2imageid[study_id]:
        label_dict[image_id]['study_label'] = study_label

df = pd.read_csv('../../input/train_image_level.csv')
image_id_list = df['id'].values
boxes_list = df['boxes'].values

# load lung cooridinates
with open('../../processed_input/input/size_dict_train.pickle', 'rb') as f:
    size_dict_train = pickle.load(f)
print(len(size_dict_train))

for i in tqdm(range(len(image_id_list))):
    image_id = image_id_list[i].split('_')[0]
    if boxes_list[i] is np.nan: # none
        label_dict[image_id]['is_none'] = True
    else:
        box_list = ast.literal_eval(boxes_list[i])
        for bbox in box_list:
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = bbox['x']+bbox['width']
            y2 = bbox['y']+bbox['height']
            h, w = size_dict_train[image_id]
            x11 = max(x1/w*1280, 0)
            y11 = max(y1/h*1280, 0)
            x22 = min(x2/w*1280, 1280)
            y22 = min(y2/h*1280, 1280)
            label_dict[image_id]['bbox_list'].append([int(x11), int(y11), int(x22), int(y22)])
            label_dict[image_id]['original_bbox_list'].append([int(x1), int(y1), int(x2), int(y2)])
            label_dict[image_id]['label_list'].append(1)

# only use positive data for detector training
list_detect = []
for key in label_dict:
    if label_dict[key]['is_none']==False:
        list_detect.append(key)
print(len(list_detect))

# create 5 folds
study_id_list = np.array(study_id_list)
k = 5
split_seed = 8366
train_image_id_cv_list = []
train_image_id_cv_list_detect = []
valid_image_id_cv_list = []
valid_image_id_cv_list_detect = []
kf = KFold(n_splits=k, shuffle=True, random_state=split_seed)

for train_index, valid_index in kf.split(study_id_list):
    train_image_id_cv_list1 = []
    train_image_id_cv_list_detect1 = []
    valid_image_id_cv_list1 = []
    valid_image_id_cv_list_detect1 = []

    train_study_id_list = study_id_list[train_index]
    valid_study_id_list = study_id_list[valid_index]

    for study_id in train_study_id_list:
        all_image_none = True
        for image_id in studyid2imageid[study_id]:   
            if label_dict[image_id]['is_none'] == False:
                all_image_none = False
        if all_image_none:
            train_image_id_cv_list1 += studyid2imageid[study_id]  
        else:
            for image_id in studyid2imageid[study_id]:
                if label_dict[image_id]['is_none'] == False:   
                    train_image_id_cv_list1.append(image_id)
        for image_id in studyid2imageid[study_id]:
            if image_id in list_detect:
                train_image_id_cv_list_detect1.append(image_id)
    train_image_id_cv_list.append(train_image_id_cv_list1)
    train_image_id_cv_list_detect.append(train_image_id_cv_list_detect1)

    for study_id in valid_study_id_list:
        all_image_none = True
        for image_id in studyid2imageid[study_id]:   
            if label_dict[image_id]['is_none'] == False:
                all_image_none = False
        if all_image_none:
            valid_image_id_cv_list1 += studyid2imageid[study_id]  
        else:
            for image_id in studyid2imageid[study_id]:
                if label_dict[image_id]['is_none'] == False:   
                    valid_image_id_cv_list1.append(image_id)
        for image_id in studyid2imageid[study_id]:
            if image_id in list_detect:
                valid_image_id_cv_list_detect1.append(image_id)
    valid_image_id_cv_list.append(valid_image_id_cv_list1)
    valid_image_id_cv_list_detect.append(valid_image_id_cv_list_detect1)

    print(len(train_index), len(valid_index))

print(len(train_image_id_cv_list_detect[0]), len(train_image_id_cv_list[0]), len(valid_image_id_cv_list_detect[0]), len(valid_image_id_cv_list[0]))
print(len(train_image_id_cv_list_detect[1]), len(train_image_id_cv_list[1]), len(valid_image_id_cv_list_detect[1]), len(valid_image_id_cv_list[1]))
print(len(train_image_id_cv_list_detect[2]), len(train_image_id_cv_list[2]), len(valid_image_id_cv_list_detect[2]), len(valid_image_id_cv_list[2]))
print(len(train_image_id_cv_list_detect[3]), len(train_image_id_cv_list[3]), len(valid_image_id_cv_list_detect[3]), len(valid_image_id_cv_list[3]))
print(len(train_image_id_cv_list_detect[4]), len(train_image_id_cv_list[4]), len(valid_image_id_cv_list_detect[4]), len(valid_image_id_cv_list[4]))


########################################################################################################
train_image_dir = 'split51/fold0/images/train/'
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
valid_image_dir = 'split51/fold0/images/valid/'
if not os.path.exists(valid_image_dir):
    os.makedirs(valid_image_dir)
train_label_dir = 'split51/fold0/labels/train/'
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
valid_label_dir = 'split51/fold0/labels/valid/'
if not os.path.exists(valid_label_dir):
    os.makedirs(valid_label_dir)

image_list_valid = valid_image_id_cv_list[0]
for i in tqdm(range(len(image_list_valid))):   
    copyfile(label_dict[image_list_valid[i]]['img_dir'], valid_image_dir+image_list_valid[i]+'.png')
    with open(valid_label_dir+image_list_valid[i]+'.txt', 'w') as f:
        if not label_dict[image_list_valid[i]]['is_none']:
            for j in range(len(label_dict[image_list_valid[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_valid[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_valid[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')
image_list_train = train_image_id_cv_list[0]
for i in tqdm(range(len(image_list_train))):   
    copyfile(label_dict[image_list_train[i]]['img_dir'], train_image_dir+image_list_train[i]+'.png')
    with open(train_label_dir+image_list_train[i]+'.txt', 'w') as f:
        if not label_dict[image_list_train[i]]['is_none']:
            for j in range(len(label_dict[image_list_train[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_train[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_train[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')


train_image_dir = 'split51/fold1/images/train/'
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
valid_image_dir = 'split51/fold1/images/valid/'
if not os.path.exists(valid_image_dir):
    os.makedirs(valid_image_dir)
train_label_dir = 'split51/fold1/labels/train/'
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
valid_label_dir = 'split51/fold1/labels/valid/'
if not os.path.exists(valid_label_dir):
    os.makedirs(valid_label_dir)

image_list_valid = valid_image_id_cv_list[1]
for i in tqdm(range(len(image_list_valid))):   
    copyfile(label_dict[image_list_valid[i]]['img_dir'], valid_image_dir+image_list_valid[i]+'.png')
    with open(valid_label_dir+image_list_valid[i]+'.txt', 'w') as f:
        if not label_dict[image_list_valid[i]]['is_none']:
            for j in range(len(label_dict[image_list_valid[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_valid[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_valid[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')
image_list_train = train_image_id_cv_list[1]
for i in tqdm(range(len(image_list_train))):   
    copyfile(label_dict[image_list_train[i]]['img_dir'], train_image_dir+image_list_train[i]+'.png')
    with open(train_label_dir+image_list_train[i]+'.txt', 'w') as f:
        if not label_dict[image_list_train[i]]['is_none']:
            for j in range(len(label_dict[image_list_train[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_train[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_train[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')


train_image_dir = 'split51/fold2/images/train/'
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
valid_image_dir = 'split51/fold2/images/valid/'
if not os.path.exists(valid_image_dir):
    os.makedirs(valid_image_dir)
train_label_dir = 'split51/fold2/labels/train/'
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
valid_label_dir = 'split51/fold2/labels/valid/'
if not os.path.exists(valid_label_dir):
    os.makedirs(valid_label_dir)

image_list_valid = valid_image_id_cv_list[2]
for i in tqdm(range(len(image_list_valid))):   
    copyfile(label_dict[image_list_valid[i]]['img_dir'], valid_image_dir+image_list_valid[i]+'.png')
    with open(valid_label_dir+image_list_valid[i]+'.txt', 'w') as f:
        if not label_dict[image_list_valid[i]]['is_none']:
            for j in range(len(label_dict[image_list_valid[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_valid[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_valid[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')
image_list_train = train_image_id_cv_list[2]
for i in tqdm(range(len(image_list_train))):   
    copyfile(label_dict[image_list_train[i]]['img_dir'], train_image_dir+image_list_train[i]+'.png')
    with open(train_label_dir+image_list_train[i]+'.txt', 'w') as f:
        if not label_dict[image_list_train[i]]['is_none']:
            for j in range(len(label_dict[image_list_train[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_train[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_train[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')


train_image_dir = 'split51/fold3/images/train/'
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
valid_image_dir = 'split51/fold3/images/valid/'
if not os.path.exists(valid_image_dir):
    os.makedirs(valid_image_dir)
train_label_dir = 'split51/fold3/labels/train/'
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
valid_label_dir = 'split51/fold3/labels/valid/'
if not os.path.exists(valid_label_dir):
    os.makedirs(valid_label_dir)

image_list_valid = valid_image_id_cv_list[3]
for i in tqdm(range(len(image_list_valid))):   
    copyfile(label_dict[image_list_valid[i]]['img_dir'], valid_image_dir+image_list_valid[i]+'.png')
    with open(valid_label_dir+image_list_valid[i]+'.txt', 'w') as f:
        if not label_dict[image_list_valid[i]]['is_none']:
            for j in range(len(label_dict[image_list_valid[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_valid[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_valid[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')
image_list_train = train_image_id_cv_list[3]
for i in tqdm(range(len(image_list_train))):   
    copyfile(label_dict[image_list_train[i]]['img_dir'], train_image_dir+image_list_train[i]+'.png')
    with open(train_label_dir+image_list_train[i]+'.txt', 'w') as f:
        if not label_dict[image_list_train[i]]['is_none']:
            for j in range(len(label_dict[image_list_train[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_train[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_train[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')


train_image_dir = 'split51/fold4/images/train/'
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
valid_image_dir = 'split51/fold4/images/valid/'
if not os.path.exists(valid_image_dir):
    os.makedirs(valid_image_dir)
train_label_dir = 'split51/fold4/labels/train/'
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)
valid_label_dir = 'split51/fold4/labels/valid/'
if not os.path.exists(valid_label_dir):
    os.makedirs(valid_label_dir)

image_list_valid = valid_image_id_cv_list[4]
for i in tqdm(range(len(image_list_valid))):   
    copyfile(label_dict[image_list_valid[i]]['img_dir'], valid_image_dir+image_list_valid[i]+'.png')
    with open(valid_label_dir+image_list_valid[i]+'.txt', 'w') as f:
        if not label_dict[image_list_valid[i]]['is_none']:
            for j in range(len(label_dict[image_list_valid[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_valid[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_valid[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')
image_list_train = train_image_id_cv_list[4]
for i in tqdm(range(len(image_list_train))):   
    copyfile(label_dict[image_list_train[i]]['img_dir'], train_image_dir+image_list_train[i]+'.png')
    with open(train_label_dir+image_list_train[i]+'.txt', 'w') as f:
        if not label_dict[image_list_train[i]]['is_none']:
            for j in range(len(label_dict[image_list_train[i]]['bbox_list'])):
                f.writelines(str(label_dict[image_list_train[i]]['label_list'][j]-1)+' ')
                box = label_dict[image_list_train[i]]['bbox_list'][j][:] # copy by value
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0]+width/2.)/1280.
                y_center = (box[1]+height/2.)/1280.
                width = width/1280.
                height = height/1280.
                f.writelines(str(x_center)+' ')
                f.writelines(str(y_center)+' ')
                f.writelines(str(width)+' ')
                f.writelines(str(height))
                f.write('\n')
########################################################################################################

