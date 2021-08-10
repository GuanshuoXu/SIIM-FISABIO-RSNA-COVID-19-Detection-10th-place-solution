import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import timm
import pickle
import time
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def macro_multilabel_map(label, pred):
    aucs = []
    for i in range(4):
        aucs.append(average_precision_score(label[:, i], pred[:, i]))
    return aucs

def macro_multilabel_auc(label, pred):
    aucs = []
    for i in range(4):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    return aucs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StudyLevelDataset(Dataset):
    def __init__(self, id_list, label_dict, target_size, do_fixed_mirror=False):
        self.id_list=id_list
        self.label_dict=label_dict
        self.target_size=target_size
        self.do_fixed_mirror=do_fixed_mirror
    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, index):
        x = cv2.imread(self.label_dict[self.id_list[index]]['img_dir'])
        x = cv2.resize(x, (self.target_size, self.target_size))   
        x = self.fixed_mirror(x)        
        x = x.transpose(2, 0, 1)
        return x
    def fixed_mirror(self, x):
        if self.do_fixed_mirror:
            x = np.ascontiguousarray(x[:, ::-1, ...])
        return x

class StudyLevelEffb8(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = timm.create_model('tf_efficientnet_b8', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = 704
        self.last_linear = nn.Linear(in_features, 4)
        self.conv_mask = nn.Conv2d(248, 1, kernel_size=1, stride=1, bias=True)
    @autocast()
    def forward(self, x):
        x1, x = self.net(x)
        x_mask = self.conv_mask(x1)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x, x_mask

def main():

    start_time = time.time()

    # checkpoint list
    checkpoint_list = [
                       'epoch26_polyak',
                      ]

    train_mean_auc_list = []
    train_negative_auc_list = []
    train_typical_auc_list = []
    train_indeterminate_auc_list = []
    train_atypical_auc_list = []
    valid_mean_auc_list = []
    valid_negative_auc_list = []
    valid_typical_auc_list = []
    valid_indeterminate_auc_list = []
    valid_atypical_auc_list = []
    train_mean_map_list = []
    train_negative_map_list = []
    train_typical_map_list = []
    train_indeterminate_map_list = []
    train_atypical_map_list = []
    valid_mean_map_list = []
    valid_negative_map_list = []
    valid_typical_map_list = []
    valid_indeterminate_map_list = []
    valid_atypical_map_list = []

    oof_pred_prob_list = []
    oof_label_list = []
    for fold in range(5):
        # prepare input

        import pickle
        with open('../../splits/split22/train_image_id_cv_list.pickle', 'rb') as f:
            train_id_list = pickle.load(f)[fold]
        with open('../../splits/split22/valid_image_id_cv_list.pickle', 'rb') as f:
            valid_id_list = pickle.load(f)[fold]
        with open('../../splits/split22/label_dict.pickle', 'rb') as f:
            label_dict = pickle.load(f)
        print(len(train_id_list), len(valid_id_list), len(label_dict))

        # hyperparameters
        batch_size = 144
        image_size = 608

        # start validation
        for ckp in checkpoint_list:

            # build model
            model = StudyLevelEffb8()
            model.load_state_dict(torch.load('../fold{}/weights/'.format(fold)+ckp))
            model.cuda()
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.eval()

            pred_prob_train = np.zeros((len(train_id_list), 4), dtype=np.float32)
            # iterator for validation
            train_datagen = StudyLevelDataset(train_id_list, label_dict, image_size, False)
            train_generator = DataLoader(dataset=train_datagen,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)

            for i, images in enumerate(train_generator):
                with torch.no_grad():
                    start = i*batch_size
                    end = start+batch_size
                    if i == len(train_generator)-1:
                        end = len(train_generator.dataset)
                    images = images.cuda().float() / 255.0
                    with autocast():
                        logits, _ = model(images)
                    pred_prob_train[start:end] += logits.sigmoid().cpu().data.numpy()

            train_datagen = StudyLevelDataset(train_id_list, label_dict, image_size, True)
            train_generator = DataLoader(dataset=train_datagen,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)

            for i, images in enumerate(train_generator):
                with torch.no_grad():
                    start = i*batch_size
                    end = start+batch_size
                    if i == len(train_generator)-1:
                        end = len(train_generator.dataset)
                    images = images.cuda().float() / 255.0
                    with autocast():
                        logits, _ = model(images)
                    pred_prob_train[start:end] += logits.sigmoid().cpu().data.numpy()

            pred_prob_train /= 2.0
            label_list = []
            for i in range(len(train_id_list)):
                temp = np.zeros((4, ), dtype=np.float32)
                temp[int(label_dict[train_id_list[i]]['study_label']-3)] = 1.
                label_list.append(temp)
            label_list = np.array(label_list)
            aucs_train = macro_multilabel_auc(label_list, pred_prob_train)
            train_mean_auc_list.append(np.mean(aucs_train))
            train_negative_auc_list.append(aucs_train[0])
            train_typical_auc_list.append(aucs_train[1])
            train_indeterminate_auc_list.append(aucs_train[2])
            train_atypical_auc_list.append(aucs_train[3])
            maps_train = macro_multilabel_map(label_list, pred_prob_train)
            train_mean_map_list.append(np.mean(maps_train))
            train_negative_map_list.append(maps_train[0])
            train_typical_map_list.append(maps_train[1])
            train_indeterminate_map_list.append(maps_train[2])
            train_atypical_map_list.append(maps_train[3])


            pred_prob_valid = np.zeros((len(valid_id_list), 4), dtype=np.float32)
            # iterator for validation
            valid_datagen = StudyLevelDataset(valid_id_list, label_dict, image_size, False)
            valid_generator = DataLoader(dataset=valid_datagen,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=True)

            for i, images in enumerate(valid_generator):
                with torch.no_grad():
                    start = i*batch_size
                    end = start+batch_size
                    if i == len(valid_generator)-1:
                        end = len(valid_generator.dataset)
                    images = images.cuda().float() / 255.0
                    with autocast():
                        logits, _ = model(images)
                    pred_prob_valid[start:end] += logits.sigmoid().cpu().data.numpy()

            valid_datagen = StudyLevelDataset(valid_id_list, label_dict, image_size, True)
            valid_generator = DataLoader(dataset=valid_datagen,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=True)

            for i, images in enumerate(valid_generator):
                with torch.no_grad():
                    start = i*batch_size
                    end = start+batch_size
                    if i == len(valid_generator)-1:
                        end = len(valid_generator.dataset)
                    images = images.cuda().float() / 255.0
                    with autocast():
                        logits, _ = model(images)
                    pred_prob_valid[start:end] += logits.sigmoid().cpu().data.numpy()

            pred_prob_valid /= 2.0
            label_list = []
            for i in range(len(valid_id_list)):
                temp = np.zeros((4, ), dtype=np.float32)
                temp[int(label_dict[valid_id_list[i]]['study_label']-3)] = 1.
                label_list.append(temp)
            label_list = np.array(label_list)
            aucs_valid = macro_multilabel_auc(label_list, pred_prob_valid)
            valid_mean_auc_list.append(np.mean(aucs_valid))
            valid_negative_auc_list.append(aucs_valid[0])
            valid_typical_auc_list.append(aucs_valid[1])
            valid_indeterminate_auc_list.append(aucs_valid[2])
            valid_atypical_auc_list.append(aucs_valid[3])
            maps_valid = macro_multilabel_map(label_list, pred_prob_valid)
            valid_mean_map_list.append(np.mean(maps_valid))
            valid_negative_map_list.append(maps_valid[0])
            valid_typical_map_list.append(maps_valid[1])
            valid_indeterminate_map_list.append(maps_valid[2])
            valid_atypical_map_list.append(maps_valid[3])

            oof_pred_prob_list.append(pred_prob_valid)
            oof_label_list.append(label_list)

    print()
    print('train_mean_auc: {}, valid_mean_auc: {}'.format(np.mean(train_mean_auc_list), np.mean(valid_mean_auc_list)), flush=True)
    print()
    print('train_negative_auc: {}, valid_negative_auc: {}'.format(np.mean(train_negative_auc_list), np.mean(valid_negative_auc_list)), flush=True)
    print()
    print('train_typical_auc: {}, valid_typical_auc: {}'.format(np.mean(train_typical_auc_list), np.mean(valid_typical_auc_list)), flush=True)
    print()
    print('train_indeterminate_auc: {}, valid_indeterminate_auc: {}'.format(np.mean(train_indeterminate_auc_list), np.mean(valid_indeterminate_auc_list)), flush=True)
    print()
    print('train_atypical_auc: {}, valid_atypical_auc: {}'.format(np.mean(train_atypical_auc_list), np.mean(valid_atypical_auc_list)), flush=True)

    print()
    print('train_mean_map: {}, valid_mean_map: {}'.format(np.mean(train_mean_map_list), np.mean(valid_mean_map_list)), flush=True)
    print()
    print('train_negative_map: {}, valid_negative_map: {}'.format(np.mean(train_negative_map_list), np.mean(valid_negative_map_list)), flush=True)
    print()
    print('train_typical_map: {}, valid_typical_map: {}'.format(np.mean(train_typical_map_list), np.mean(valid_typical_map_list)), flush=True)
    print()
    print('train_indeterminate_map: {}, valid_indeterminate_map: {}'.format(np.mean(train_indeterminate_map_list), np.mean(valid_indeterminate_map_list)), flush=True)
    print()
    print('train_atypical_map: {}, valid_atypical_map: {}'.format(np.mean(train_atypical_map_list), np.mean(valid_atypical_map_list)), flush=True)

    oof_pred_prob_list = np.concatenate(oof_pred_prob_list)
    oof_label_list = np.concatenate(oof_label_list)
    print(oof_pred_prob_list.shape, oof_label_list.shape)
    maps_oof = macro_multilabel_map(oof_label_list, oof_pred_prob_list)
    print()    
    print('oof_mean_map: {}'.format(np.mean(maps_oof)), flush=True)
    print()
    print('oof_negative_map: {}'.format(maps_oof[0]), flush=True)
    print()
    print('oof_typical_map: {}'.format(maps_oof[1]), flush=True)
    print()
    print('oof_indeterminate_map: {}'.format(maps_oof[2]), flush=True)
    print()
    print('oof_atypical_map: {}'.format(maps_oof[3]), flush=True)

    print()
    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
