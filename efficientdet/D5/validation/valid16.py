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
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict, load_pretrained
from effdet.data.parsers import CocoParserCfg, create_parser
from effdet.evaluator import CocoEvaluator
from effdet.efficientdet import HeadNet
from ensemble_boxes import weighted_boxes_fusion
import time
from torch.cuda.amp import autocast, GradScaler

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
    def __init__(self, fold, img_ids, label_dict, image_list, target_size):
        parser_config = CocoParserCfg(ann_filename='../../splits/split1/valid_original_fold{}.json'.format(fold))
        self.parser = create_parser('coco', cfg=parser_config)
        self.img_ids=img_ids
        self.label_dict=label_dict
        self.image_list=image_list
        self.target_size=target_size
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self,index):
        image_id = self.image_list[index]
        image = cv2.imread(self.label_dict[image_id]['img_dir'])
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        return image, index

def collate_fn(batch):
    image_list = []
    index_list = []
    for i in range(len(batch)):
        image_list.append(batch[i][0])
        index_list.append(batch[i][1])
    image_list = np.stack(image_list)
    image_list = torch.from_numpy(image_list)
    index_list = torch.tensor(index_list, dtype=torch.int64)
    targets = {'img_idx': index_list}
    return image_list, targets

def get_net(image_size, checkpoint):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.image_size = [image_size[0], image_size[1]]
    config.norm_kwargs = dict(eps=0.001, momentum=0.01)
    net = EfficientDet(config, pretrained_backbone=False)
    net.reset_head(num_classes=6)
    return DetBenchPredict(net)

def main():

    start_time = time.time()

    # checkpoint list
    checkpoint_list = [
                       'epoch50_polyak',
                      ]

    opacity_map_list = []
    none_map_list = []
    negative_map_list = []
    typical_map_list = []
    indeterminate_map_list = []
    atypical_map_list = []
    average_map_list = []
    for fold in range(5):
        # prepare input

        import pickle
        with open('../../splits/split1/valid_image_id_cv_list.pickle', 'rb') as f:
            image_list_valid = pickle.load(f)[fold]
        with open('../../splits/split1/label_dict.pickle', 'rb') as f:
            label_dict = pickle.load(f)
        with open('../../splits/split1/img_ids_fold{}.pickle'.format(fold), 'rb') as f:
            img_ids = pickle.load(f)

        print(len(image_list_valid), len(label_dict))

        # load lung cooridinates
        with open('../../../processed_input/input/size_dict_train.pickle', 'rb') as f:
            size_dict_train = pickle.load(f)
        print(len(size_dict_train))

        # hyperparameters
        batch_size = 8
        image_size = (1024,1024)

        # start validation
        for ckp in checkpoint_list:

            # build model
            model = get_net(image_size, ckp)
            model.load_state_dict(torch.load('../fold{}/weights/'.format(fold)+ckp))
            model.cuda()
            model.eval()

            # iterator for validation
            datagen = VinDataset(fold=fold,
                                 img_ids=img_ids,
                                 label_dict=label_dict, 
                                 image_list=image_list_valid,
                                 target_size=image_size)
            generator = DataLoader(dataset=datagen, 
                                   collate_fn=collate_fn, 
                                   batch_size=batch_size, 
                                   shuffle=False,
                                   num_workers=8, 
                                   pin_memory=False)

            evaluator = CocoEvaluator(datagen)
            for i, (images, targets) in enumerate(generator):
                with torch.no_grad():
                    start = i*batch_size
                    end = start+batch_size
                    if i == len(generator)-1:
                        end = len(generator.dataset)
                    images = images.cuda()
                    images = images.permute(0,3,1,2).float().div(255)
                    with autocast():
                        detections = model(images).cpu().numpy()
                        detections_flip = model(images.flip(3)).cpu().numpy()
                    detections_flip[:,:,[0,2]] = image_size[1] - detections_flip[:,:,[2,0]]

                    detections[:,:,0] /= image_size[1]
                    detections[:,:,2] /= image_size[1]
                    detections[:,:,1] /= image_size[0]
                    detections[:,:,3] /= image_size[0]
                    detections_flip[:,:,0] /= image_size[1]
                    detections_flip[:,:,2] /= image_size[1]
                    detections_flip[:,:,1] /= image_size[0]
                    detections_flip[:,:,3] /= image_size[0]

                    detections_ensemble = np.zeros(detections.shape, dtype=np.float32)
                    for n in range(detections.shape[0]):
                        boxes = [detections[n,:,:4].tolist(), detections_flip[n,:,:4].tolist()]
                        scores = [detections[n,:,4].tolist(), detections_flip[n,:,4].tolist()]
                        labels = [detections[n,:,5].tolist(), detections_flip[n,:,5].tolist()]
                        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.6)

                        h, w = size_dict_train[image_list_valid[targets['img_idx'][n]]]
                        boxes[:,0] *= w
                        boxes[:,2] *= w
                        boxes[:,1] *= h
                        boxes[:,3] *= h

                        if len(boxes)>=99:
                            detections_ensemble[n,:99,:4] = boxes[:99,:]
                            detections_ensemble[n,:99,4] = scores[:99]
                            detections_ensemble[n,:99,5] = labels[:99]
                        else:
                            detections_ensemble[n,:len(boxes),:4] = boxes
                            detections_ensemble[n,:len(boxes),4] = scores
                            detections_ensemble[n,:len(boxes),5] = labels

                        # Only consider top 1 for each of the study level classes.
                        find3 = False
                        find4 = False
                        find5 = False
                        find6 = False
                        for bb in range(detections_ensemble.shape[1]):
                            if detections_ensemble[n,bb,5]==3.0:
                                if find3:
                                    detections_ensemble[n,bb,:] = 0.0      
                                else:   
                                    #find3 = True
                                    detections_ensemble[n,bb,:4] = np.array([0,0,1,1], dtype=np.float32)
                            elif detections_ensemble[n,bb,5]==4.0:
                                if find4:
                                    detections_ensemble[n,bb,:] = 0.0      
                                else:   
                                    #find4 = True
                                    detections_ensemble[n,bb,:4] = np.array([0,0,1,1], dtype=np.float32)
                            elif detections_ensemble[n,bb,5]==5.0:
                                if find5:
                                    detections_ensemble[n,bb,:] = 0.0      
                                else:   
                                    #find5 = True
                                    detections_ensemble[n,bb,:4] = np.array([0,0,1,1], dtype=np.float32)
                            elif detections_ensemble[n,bb,5]==6.0:
                                if find6:
                                    detections_ensemble[n,bb,:] = 0.0      
                                else:   
                                    #find6 = True
                                    detections_ensemble[n,bb,:4] = np.array([0,0,1,1], dtype=np.float32)

                        # Estimate the none class using the topK (K=3) opacity probs.
                        detections_ensemble[n,-1,:4] = np.array([0,0,1,1], dtype=np.float32)
                        non_prob = 1.0
                        count = 0
                        for bb in range(detections_ensemble.shape[1]):
                            if detections_ensemble[n,bb,5]==1.0:
                                non_prob *= (1.0-detections_ensemble[n,bb,4])
                                count += 1
                                if count>=3:
                                    break
                        detections_ensemble[n,-1,4] = non_prob
                        detections_ensemble[n,-1,5] = 2
                    
                    detections_ensemble = torch.tensor(detections_ensemble)

                    evaluator.add_predictions(detections_ensemble, targets)

            MAP = evaluator.evaluate()
            print('{}, map@0.5'.format(ckp), flush=True)
            print('{}, opacity: {}'.format(ckp, MAP[0]), flush=True)
            print('{}, none: {}'.format(ckp, MAP[1]), flush=True)
            print('{}, negative: {}'.format(ckp, MAP[2]), flush=True)
            print('{}, typical: {}'.format(ckp, MAP[3]), flush=True)
            print('{}, indeterminate: {}'.format(ckp, MAP[4]), flush=True)
            print('{}, atypical: {}'.format(ckp, MAP[5]), flush=True)
            print('{}, average: {}'.format(ckp, np.mean(MAP)), flush=True)

            opacity_map_list.append(MAP[0])
            none_map_list.append(MAP[1])
            negative_map_list.append(MAP[2])
            typical_map_list.append(MAP[3])
            indeterminate_map_list.append(MAP[4])
            atypical_map_list.append(MAP[5])
            average_map_list.append(np.mean(MAP))

    print()
    print('opacity_map_list:       {}'.format(opacity_map_list), flush=True)
    print('none_map_list:          {}'.format(none_map_list), flush=True)
    print('negative_map_list:      {}'.format(negative_map_list), flush=True)
    print('typical_map_list:       {}'.format(typical_map_list), flush=True)
    print('indeterminate_map_list: {}'.format(indeterminate_map_list), flush=True)
    print('atypical_map_list:      {}'.format(atypical_map_list), flush=True)
    print('average_map_list:       {}'.format(average_map_list), flush=True)
    print('mean opacity:       {}'.format(np.mean(opacity_map_list)), flush=True)
    print('mean none:          {}'.format(np.mean(none_map_list)), flush=True)
    print('mean negative:      {}'.format(np.mean(negative_map_list)), flush=True)
    print('mean typical:       {}'.format(np.mean(typical_map_list)), flush=True)
    print('mean indeterminate: {}'.format(np.mean(indeterminate_map_list)), flush=True)
    print('mean atypical:      {}'.format(np.mean(atypical_map_list)), flush=True)
    print('mean average_map:   {}'.format(np.mean(average_map_list)), flush=True)

    end_time = time.time()
    print(end_time-start_time)

if __name__ == "__main__":
    main()
