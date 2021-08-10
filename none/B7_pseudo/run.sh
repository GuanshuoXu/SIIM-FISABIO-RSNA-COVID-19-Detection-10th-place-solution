cd fold0
python generate_pseudo_labels_bimcv.py
python generate_pseudo_labels_bimcv_negative_old.py
python generate_pseudo_labels_midrc.py
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold1
python generate_pseudo_labels_bimcv.py
python generate_pseudo_labels_bimcv_negative_old.py
python generate_pseudo_labels_midrc.py
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold2
python generate_pseudo_labels_bimcv.py
python generate_pseudo_labels_bimcv_negative_old.py
python generate_pseudo_labels_midrc.py
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold3
python generate_pseudo_labels_bimcv.py
python generate_pseudo_labels_bimcv_negative_old.py
python generate_pseudo_labels_midrc.py
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold4
python generate_pseudo_labels_bimcv.py
python generate_pseudo_labels_bimcv_negative_old.py
python generate_pseudo_labels_midrc.py
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train1.py > train1.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train1.py > train1.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train1.py > train1.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train1.py > train1.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train1.py > train1.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train2.py > train2.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train2.py > train2.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train2.py > train2.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train2.py > train2.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train2.py > train2.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train3.py > train3.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train3.py > train3.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train3.py > train3.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train3.py > train3.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train3.py > train3.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train4.py > train4.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train4.py > train4.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train4.py > train4.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train4.py > train4.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train4.py > train4.txt
cd ..
cd validation
python valid4.py > valid4.txt
cd ..
