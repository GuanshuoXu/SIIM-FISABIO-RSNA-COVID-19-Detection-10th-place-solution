import argparse
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import copy
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, load_pretrained
from effdet.efficientdet import HeadNet
from torch.cuda.amp import autocast, GradScaler 
import time

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

class VinDataset(Dataset):
    def __init__(self, label_dict, image_list, target_size, transform, transform_safe=None):
        self.label_dict=label_dict
        self.image_list=image_list
        self.target_size=target_size
        self.transform=transform
        self.transform_safe=transform_safe
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        image_id = self.image_list[index]
        image = cv2.imread(self.label_dict[image_id]['img_dir'])
        
        if len(self.label_dict[image_id]['bbox_list']) == 0:
            boxes = np.array([[self.target_size[1]*0.5-self.target_size[1]*0.25, self.target_size[0]*0.5-self.target_size[0]*0.25, self.target_size[1]*0.5+self.target_size[1]*0.25, self.target_size[0]*0.5+self.target_size[0]*0.25], [self.target_size[1]*0.5-self.target_size[1]*0.25, self.target_size[0]*0.5-self.target_size[0]*0.25, self.target_size[1]*0.5+self.target_size[1]*0.25, self.target_size[0]*0.5+self.target_size[0]*0.25]])
            labels = np.array([0,0], dtype=np.int64)
        else:
            boxes = np.stack(self.label_dict[image_id]['bbox_list'])
            labels = np.array(self.label_dict[image_id]['label_list'])

        if self.transform_safe:
            transform_successful = False
            for i in range(20):
                sample = self.transform(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    boxes = boxes.float()
                    boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]  #yxyx: be warning
                    labels = torch.tensor(sample['labels'], dtype=torch.int64)
                    transform_successful = True
                    break
            if not transform_successful:
                sample = self.transform_safe(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
                image = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                boxes = boxes.float()
                boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]  #yxyx: be warning
                labels = torch.tensor(sample['labels'], dtype=torch.int64)
        else:
            sample = self.transform(**{
                'image': image,
                'bboxes': boxes,
                'labels': labels
            })
            image = sample['image']
            boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            boxes = boxes.float()
            boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]  #yxyx: be warning
            labels = torch.tensor(sample['labels'], dtype=torch.int64)

        if len(self.label_dict[image_id]['bbox_list']) == 0:
            boxes = torch.zeros((1, 4))
            labels = torch.tensor([0], dtype=torch.int64)

        box_study = torch.tensor([[0,0,self.target_size[0],self.target_size[1]]])
        label_study = torch.tensor([self.label_dict[image_id]['study_label']], dtype=torch.int64)
        boxes = torch.cat((boxes, box_study))
        labels = torch.cat((labels, label_study))

        return image, boxes, labels, index

def collate_fn(batch):
    image_list = []
    index_list = []
    boxes_list = []
    labels_list = []
    for i in range(len(batch)):
        image_list.append(batch[i][0])
        boxes_list.append(batch[i][1])
        labels_list.append(batch[i][2])
        index_list.append(batch[i][3])
    image_list = np.stack(image_list)
    image_list = torch.from_numpy(image_list)
    index_list = torch.tensor(index_list, dtype=torch.int64)
    targets = {'bbox': boxes_list, 'cls': labels_list, 'img_idx': index_list}
    return image_list, targets

def get_net(image_size):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.image_size = [image_size[0], image_size[1]]
    config.norm_kwargs = dict(eps=0.001, momentum=0.01)
    net = EfficientDet(config, pretrained_backbone=False)
    load_pretrained(net, config.url)
    net.reset_head(num_classes=6)
    return DetBenchTrain(net, create_labeler=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    seed = 2202
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

    # prepare input
    fold = 2

    import pickle
    with open('../../splits/split1/train_image_id_cv_list.pickle', 'rb') as f:
        image_list_train = pickle.load(f)[fold]
    with open('../../splits/split1/label_dict.pickle', 'rb') as f:
        label_dict = pickle.load(f)

    print(len(image_list_train), len(label_dict))

    # hyperparameters
    learning_rate = 0.0003
    batch_size = 8
    image_size = (1024,1024)
    num_polyak = 32
    num_epoch = 6

    # build model
    if args.local_rank != 0:
        torch.distributed.barrier()
    model = get_net(image_size)
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.local_rank == 0:
        model.load_state_dict(torch.load('weights/epoch2'))
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # training
    train_transform = albumentations.Compose(
        [
         albumentations.Resize(height=image_size[0], width=image_size[1], always_apply=True, p=1.0),
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
         albumentations.Cutout(num_holes=8, max_h_size=int(0.1*image_size[0]), max_w_size=int(0.1*image_size[1]), fill_value=0, always_apply=True, p=0.8),
        ],
        p=1.0, 
        bbox_params=albumentations.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )
    train_transform_safe = albumentations.Compose(
        [
         albumentations.Resize(height=image_size[0], width=image_size[1], always_apply=True, p=1.0),
         albumentations.HorizontalFlip(p=0.5),
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
         albumentations.IAASharpen(p=0.2),
        ],
        p=1.0, 
        bbox_params=albumentations.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

    # iterator for training
    datagen = VinDataset(label_dict=label_dict, 
                         image_list=image_list_train, 
                         target_size=image_size,
                         transform=train_transform,
                         transform_safe=train_transform_safe)
    sampler = DistributedSampler(datagen)
    generator = DataLoader(dataset=datagen, 
                           sampler=sampler,
                           collate_fn=collate_fn, 
                           batch_size=batch_size, 
                           num_workers=6, 
                           pin_memory=False)

    if args.local_rank == 0:
        start_time = time.time()

    scaler = GradScaler()
    for ep in range(3,num_epoch):
        losses = AverageMeter()
        model.train()
        for j, (images, targets) in enumerate(generator):
            optimizer.zero_grad()

            images = images.to(args.device)
            images = images.permute(0,3,1,2).float().div(255)
            boxes = [target_bbox.to(args.device) for target_bbox in targets['bbox']]
            labels = [target_cls.to(args.device) for target_cls in targets['cls']]
            targets['bbox'] = boxes
            targets['cls'] = labels

            with autocast():
                output = model(images, targets)
                loss = output['loss']
            losses.update(loss.item(), images.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.local_rank == 0 and ep==num_epoch-1:
                if j==len(generator)-num_polyak:
                    averaged_model = copy.deepcopy(model.module.state_dict())
                if j>len(generator)-num_polyak:
                    for k in averaged_model.keys():
                        averaged_model[k].data += model.module.state_dict()[k].data
                if j==len(generator)-1:
                    for k in averaged_model.keys():
                        averaged_model[k].data = averaged_model[k].data / float(num_polyak)

            #if args.local_rank == 0:
            #    print('\r',end='',flush=True)
            #    message = '%s %5.1f %6.1f    |     %0.3f     |' % ("train",j/len(generator)+ep,ep,losses.avg)
            #    print(message , end='',flush=True)

        if args.local_rank == 0:
            print('epoch: {}, train_loss: {}'.format(ep,losses.avg), flush=True)

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





