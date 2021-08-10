import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
import json
import glob
import ast

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
        'img_dir': '../../../processed_input/input/train/'+image_id+'.png',
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


# create 5 folds
study_id_list = np.array(study_id_list)
k = 5
split_seed = 4
train_image_id_cv_list = []
valid_image_id_cv_list = []
kf = KFold(n_splits=k, shuffle=True, random_state=split_seed)

for train_index, valid_index in kf.split(study_id_list):
    train_image_id_cv_list1 = []
    valid_image_id_cv_list1 = []

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
    train_image_id_cv_list.append(train_image_id_cv_list1)

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
    valid_image_id_cv_list.append(valid_image_id_cv_list1)

    print(len(train_index), len(valid_index))

print(len(train_image_id_cv_list[0]), len(valid_image_id_cv_list[0]))
print(len(train_image_id_cv_list[1]), len(valid_image_id_cv_list[1]))
print(len(train_image_id_cv_list[2]), len(valid_image_id_cv_list[2]))
print(len(train_image_id_cv_list[3]), len(valid_image_id_cv_list[3]))
print(len(train_image_id_cv_list[4]), len(valid_image_id_cv_list[4]))


out_dir = 'split1/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
with open(out_dir+'train_image_id_cv_list.pickle', 'wb') as f:
    pickle.dump(train_image_id_cv_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'valid_image_id_cv_list.pickle', 'wb') as f:
    pickle.dump(valid_image_id_cv_list, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(out_dir+'label_dict.pickle', 'wb') as f:
    pickle.dump(label_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


# save coco
for fold in tqdm(range(k)):
    image_list_valid = valid_image_id_cv_list[fold]

    img_ids = []
    for i in range(len(image_list_valid)):
        img_ids.append(i)
    with open(out_dir+'img_ids_fold{}.pickle'.format(fold), 'wb') as f:
        pickle.dump(img_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    coco_image_list = []
    coco_ann_list = []
    ann_id_count = 0
    for i in range(len(image_list_valid)):
        data_id = image_list_valid[i]
        coco_image_list.append({"file_name": data_id, 
                                "height": 4096, # arbitrary value works?
                                "width": 4096,  # arbitrary value works?
                                "id": img_ids[i]})
        for j in range(len(label_dict[data_id]['original_bbox_list'])):
            # xyxy to xywh
            box = label_dict[data_id]['original_bbox_list'][j][:] # copy by value
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            coco_ann_list.append({"image_id": img_ids[i], 
                                  "bbox": list(box), 
                                  "category_id": label_dict[data_id]['label_list'][j],
                                  "iscrowd": 0,
                                  "area": 6335, # arbitrary value works?
                                  "id": ann_id_count})
            ann_id_count += 1
        if label_dict[data_id]['is_none']:
            coco_ann_list.append({"image_id": img_ids[i], 
                                  "bbox": [0, 0, 1, 1], 
                                  "category_id": 2,
                                  "iscrowd": 0,
                                  "area": 6335, # arbitrary value works?
                                  "id": ann_id_count})
            ann_id_count += 1  
        coco_ann_list.append({"image_id": img_ids[i], 
                              "bbox": [0, 0, 1, 1], 
                              "category_id": label_dict[data_id]['study_label'],
                              "iscrowd": 0,
                              "area": 6335, # arbitrary value works?
                              "id": ann_id_count})
        ann_id_count += 1            
    coco_dict = {
                 "images": coco_image_list, 
                 "annotations": coco_ann_list,
                 "categories": [
                                {"supercategory": "opacity", "id": 1, "name": "opacity"},
                                {"supercategory": "none", "id": 2, "name": "none"},
                                {"supercategory": "negative", "id": 3, "name": "negative"},
                                {"supercategory": "typical", "id": 4, "name": "typical"},
                                {"supercategory": "indeterminate", "id": 5, "name": "indeterminate"},
                                {"supercategory": "atypical", "id": 6, "name": "atypical"},
                               ]
                }

    with open(out_dir+'valid_original_fold{}.json'.format(fold), 'w') as f:
        json.dump(coco_dict, f)


