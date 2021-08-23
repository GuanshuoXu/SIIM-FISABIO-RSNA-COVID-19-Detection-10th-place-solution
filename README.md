# SIIM-FISABIO-RSNA-COVID-19-Detection-10th-place-solution

## software installation
pytorch                   1.7.1<br/>
python                    3.8.5<br/>
opencv-python             4.5.1.48<br/>
pycocotools               2.0.2<br/>
numpy                     1.19.2<br/>
imgaug                    0.4.0<br/>
pandas                    1.1.3<br/>
pydicom                   2.1.2<br/>
gdcm<br/>

## software used but not required to install
efficientdet (https://github.com/rwightman/efficientdet-pytorch)<br/>
omegaconf<br/>
albumentations (https://github.com/albumentations-team/albumentations)<br/>
Weighted-Boxes-Fusion (https://github.com/ZFTurbo/Weighted-Boxes-Fusion)<br/>
timm (https://github.com/rwightman/pytorch-image-models)<br/>
yolov5 (https://github.com/ultralytics/yolov5)<br/>

## hardware requirement
rtx A6000  x 3 or equivalent

## run training
1. Copy the competition data into input/, then go to preprocessing_input/ and run process_input_train.py to create resized png training data.
2. For efficientdet training, first create splits by running efficientdet/splits/split1.py and split4.py, then go to efficientdet/D5 and D6 to run the run.sh file.
3. For yolo model training, first create splits by running yolo/splits/split51.py and split61.py, then go to yolo/l6 and x6 to run the run.sh file.
4. For the study-level label training, first create splits by running studylevel/splits/split1.py and split22.py, then go to studylevel/B7 and B8 to run the run.sh file.
5. For the 'none' label training, first create splits by running none/splits/split1.py and split22.py, then go to none/B7 and B8 to run the run.sh file.

## pseudo-labeling
The above models can already reproduce my best private score. My best submissions include an extra step of pseudo-labeling using external data, but it decreased private score a little. If you still want to reproduce the pseudo-labeling process:
1. Download RICORD data from https://www.kaggle.com/raddar/ricord-covid19-xray-positive-tests and extract into external_data/MIDRC/.
2. Download BIMCV-COVID19+ (New iteration 1 + 2) and BIMCV-COVID19- (iteration 1) from https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711 and extract into external_data/BIMCV/ and external_data/BIMCV_negative/ respectively. Then, go to preprocessing_input/ and run process_bimcv.py and process_bimcv_negative_old.py to get the data ready for training. And the last step is to go to none/ and studylevel/ to train the psedo-labeling models.
