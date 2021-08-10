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
from torch.utils.data.distributed import DistributedSampler
import torch
import timm
import random
import pickle
import albumentations
from torch.cuda.amp import autocast, GradScaler
import time
import copy

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
    def __init__(self, id_list, label_dict, transform):
        self.id_list=id_list
        self.label_dict=label_dict
        self.transform=transform
        self.mask_size = 38
    def __len__(self):
        return len(self.id_list)
    def __getitem__(self, index):
        x = cv2.imread(self.label_dict[self.id_list[index]]['img_dir'])
        y_mask = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32) 
        for box in self.label_dict[self.id_list[index]]['bbox_list']:
            y_mask[box[1]:box[3], box[0]:box[2]] = 1.  
        augmented = self.transform(image=x, mask=y_mask)
        x = augmented['image']
        x = x.transpose(2, 0, 1)
        y_mask = augmented['mask']     
        y_mask = cv2.resize(y_mask, (self.mask_size, self.mask_size))
        y = np.zeros((4, ), dtype=np.float32)
        y[int(self.label_dict[self.id_list[index]]['study_label']-3)] = 1.
        return x, y, y_mask

class StudyLevelEffb8(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = timm.create_model('tf_efficientnet_b8', features_only=True, out_indices=(3, 4), drop_path_rate=0.5, pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = 704
        self.last_linear = nn.Linear(in_features, 4)
        self.conv_mask = nn.Conv2d(248, 1, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        x1, x = self.net(x)
        x_mask = self.conv_mask(x1)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x, x_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 35108
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # prepare input
    fold = 1

    import pickle
    with open('../../splits/split22/train_image_id_cv_list.pickle', 'rb') as f:
        id_list = pickle.load(f)[fold]
    with open('../../splits/split22/label_dict.pickle', 'rb') as f:
        label_dict = pickle.load(f)
    print(len(id_list), len(label_dict))

    # hyperparameters
    learning_rate = 0.00004
    image_size = 608
    num_polyak = 32
    batch_size = 18
    num_epoch = 24

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = StudyLevelEffb8()
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.local_rank == 0:
        model.load_state_dict(torch.load('weights/epoch20'))
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    criterion = nn.BCEWithLogitsLoss()

    # training
    train_transform = albumentations.Compose(
        [
         albumentations.Resize(height=image_size, width=image_size, always_apply=True, p=1.0),
         albumentations.HorizontalFlip(p=0.5),
         albumentations.ShiftScaleRotate(shift_limit=0.4, scale_limit=0.4, rotate_limit=40, p=0.8),
         albumentations.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
         albumentations.CLAHE(clip_limit=(1,4), p=0.4),
         albumentations.OneOf([
             albumentations.GaussNoise(),
             albumentations.GaussianBlur(),
             albumentations.MotionBlur(),
             albumentations.MedianBlur(),
         ], p=0.2),
         albumentations.OneOf([
             albumentations.JpegCompression(),
             albumentations.Downscale(),
         ], p=0.2),
         albumentations.IAAPiecewiseAffine(p=0.2),
         albumentations.IAASharpen(p=0.2),
         #albumentations.Cutout(num_holes=8, max_h_size=int(0.1*image_size[0]), max_w_size=int(0.1*image_size[1]), fill_value=0, always_apply=True, p=0.8),
         albumentations.Cutout(num_holes=1, max_h_size=int(0.4*image_size), max_w_size=int(0.4*image_size), fill_value=0, always_apply=True, p=1.0),
        ]
    )

    # iterator for training
    train_datagen = StudyLevelDataset(id_list, label_dict, train_transform)
    train_sampler = DistributedSampler(train_datagen)
    train_generator = DataLoader(dataset=train_datagen,
                                 sampler=train_sampler,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 pin_memory=True)

    if args.local_rank == 0:
        start_time = time.time()

    scaler = GradScaler()
    for ep in range(21,num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (images, labels, labels_mask) in enumerate(train_generator):
            images = images.to(args.device).float() / 255.0
            labels = torch.from_numpy(np.array(labels)).to(args.device)
            labels_mask = labels_mask.to(args.device)

            with autocast():
                logits, logits_mask = model(images)
                loss = criterion(logits.squeeze(), labels.squeeze())+64*criterion(logits_mask.squeeze(), labels_mask.squeeze())

            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.local_rank == 0 and ep==num_epoch-1:
                if j==len(train_generator)-num_polyak:
                    averaged_model = copy.deepcopy(model.module.state_dict())
                if j>len(train_generator)-num_polyak:
                    for k in averaged_model.keys():
                        averaged_model[k].data += model.module.state_dict()[k].data
                if j==len(train_generator)-1:
                    for k in averaged_model.keys():
                        averaged_model[k].data = averaged_model[k].data / float(num_polyak)

            #if args.local_rank == 0:
            #    print('\r',end='',flush=True)
            #    message = '%s %5.1f %6.1f    |     %0.3f     |' % ("train",j/len(train_generator)+ep,ep,losses.avg)
            #    print(message , end='',flush=True)

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep, losses.avg), flush=True)

    if args.local_rank == 0:
        out_dir = 'weights/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.module.state_dict(), out_dir+'epoch{}'.format(ep))
        torch.save(averaged_model, out_dir+'epoch{}_polyak'.format(ep))

    if args.local_rank == 0:
        end_time = time.time()
        print(end_time-start_time)

if __name__ == "__main__":
    main()
