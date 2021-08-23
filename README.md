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
