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

    for fold in range(2,3):
        # prepare input
        import glob
        files = sorted(glob.glob('../../../processed_input/bimcv/train/*.png'))
        label_dict = {}
        valid_id_list = []
        for i in range(len(files)):
            image_id = files[i].split('/')[-1]
            valid_id_list.append(image_id)
            label_dict[image_id] = {
                'study_label': -1,
                'mask': [],
                'img_dir': files[i],
            }          
        print(len(valid_id_list), len(label_dict))

        # hyperparameters
        batch_size = 144
        image_size = 608

        # start validation
        for ckp in checkpoint_list:

            # build model
            model = StudyLevelEffb8()
            model.load_state_dict(torch.load('../../B8/fold{}/weights/'.format(fold)+ckp))
            model.cuda()
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.eval()

            pred_prob_valid = np.zeros((len(valid_id_list), 4), dtype=np.float32)
            pred_prob_mask_valid = np.zeros((len(valid_id_list), 38, 38), dtype=np.float32)
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
                        logits, logits_mask = model(images)
                    pred_prob_valid[start:end] += logits.sigmoid().cpu().data.numpy()
                    pred_prob_mask_valid[start:end] += logits_mask.sigmoid().cpu().data.numpy().squeeze()

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
                        logits, logits_mask = model(images)
                    pred_prob_valid[start:end] += logits.sigmoid().cpu().data.numpy()
                    pred_prob_mask_valid[start:end] += logits_mask.sigmoid().cpu().data.numpy().squeeze()

            pred_prob_valid /= 2.0
            pred_prob_mask_valid /= 2.0


        for i in range(len(valid_id_list)):
            label_dict[valid_id_list[i]]['study_label'] = pred_prob_valid[i]
            label_dict[valid_id_list[i]]['mask'] = pred_prob_mask_valid[i]

        out_dir = 'pseudo_labels/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir+'id_list_bimcv.pickle', 'wb') as f:
            pickle.dump(valid_id_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(out_dir+'label_dict_bimcv.pickle', 'wb') as f:
            pickle.dump(label_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
