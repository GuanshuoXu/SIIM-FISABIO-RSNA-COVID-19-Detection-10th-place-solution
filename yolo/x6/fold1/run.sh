python -m torch.distributed.launch --nproc_per_node 3 yolov5-5.0/train.py --img-size=1024 --batch-size 24 --epochs 160 --data vin.yaml --weights yolov5x6.pt --sync-bn
