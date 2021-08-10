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

class NoneDataset(Dataset):
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

class NoneEffb7(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = timm.create_model('tf_efficientnet_b7_ns', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = 640
        self.last_linear = nn.Linear(in_features, 1)
        self.conv_mask = nn.Conv2d(224, 1, kernel_size=1, stride=1, bias=True)
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

    train_none_auc_list = []
    valid_none_auc_list = []

    oof_pred_prob_list = []
    oof_label_list = []
    for fold in range(5):
        # prepare input

        import pickle
        with open('../../splits/split1/train_image_id_cv_list.pickle', 'rb') as f:
            train_id_list = pickle.load(f)[fold]
        with open('../../splits/split1/valid_image_id_cv_list.pickle', 'rb') as f:
            valid_id_list = pickle.load(f)[fold]
        with open('../../splits/split1/label_dict.pickle', 'rb') as f:
            label_dict = pickle.load(f)
        print(len(train_id_list), len(valid_id_list), len(label_dict))

        # hyperparameters
        batch_size = 144
        image_size = 672

        # start validation
        for ckp in checkpoint_list:

            # build model
            model = NoneEffb7()
            model.load_state_dict(torch.load('../fold{}/weights/'.format(fold)+ckp))
            model.cuda()
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.eval()

            pred_prob_train = np.zeros((len(train_id_list), ), dtype=np.float32)
            # iterator for validation
            train_datagen = NoneDataset(train_id_list, label_dict, image_size, False)
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
                    pred_prob_train[start:end] += logits.sigmoid().cpu().data.numpy().squeeze()

            train_datagen = NoneDataset(train_id_list, label_dict, image_size, True)
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
                    pred_prob_train[start:end] += logits.sigmoid().cpu().data.numpy().squeeze()

            pred_prob_train /= 2.0
            label_list = []
            for i in range(len(train_id_list)):
                if label_dict[train_id_list[i]]['is_none']:
                    label_list.append(1)
                else:
                    label_list.append(0)
            label_list = np.array(label_list)
            maps_train = average_precision_score(label_list, pred_prob_train)
            train_none_auc_list.append(maps_train)


            pred_prob_valid = np.zeros((len(valid_id_list), ), dtype=np.float32)
            # iterator for validation
            valid_datagen = NoneDataset(valid_id_list, label_dict, image_size, False)
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
                    pred_prob_valid[start:end] += logits.sigmoid().cpu().data.numpy().squeeze()

            valid_datagen = NoneDataset(valid_id_list, label_dict, image_size, True)
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
                    pred_prob_valid[start:end] += logits.sigmoid().cpu().data.numpy().squeeze()

            pred_prob_valid /= 2.0
            label_list = []
            for i in range(len(valid_id_list)):
                if label_dict[valid_id_list[i]]['is_none']:
                    label_list.append(1)
                else:
                    label_list.append(0)
            label_list = np.array(label_list)
            maps_valid = average_precision_score(label_list, pred_prob_valid)
            valid_none_auc_list.append(maps_valid)

            oof_pred_prob_list.append(pred_prob_valid)
            oof_label_list.append(label_list)

    print()
    print('train_mean_map: {}, valid_mean_map: {}'.format(np.mean(train_none_auc_list), np.mean(valid_none_auc_list)), flush=True)

    oof_pred_prob_list = np.concatenate(oof_pred_prob_list)
    oof_label_list = np.concatenate(oof_label_list)
    print(oof_pred_prob_list.shape, oof_label_list.shape)
    maps_oof = average_precision_score(oof_label_list, oof_pred_prob_list)
    print()    
    print('oof_mean_map: {}'.format(np.mean(maps_oof)), flush=True)

    print()
    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
