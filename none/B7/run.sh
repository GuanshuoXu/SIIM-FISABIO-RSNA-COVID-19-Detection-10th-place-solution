cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
cd ..
cd fold4
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
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train5.py > train5.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train5.py > train5.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train5.py > train5.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train5.py > train5.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train5.py > train5.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train6.py > train6.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train6.py > train6.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train6.py > train6.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train6.py > train6.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train6.py > train6.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train7.py > train7.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train7.py > train7.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train7.py > train7.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train7.py > train7.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train7.py > train7.txt
cd ..
cd fold0
python -m torch.distributed.launch --nproc_per_node=3 train8.py > train8.txt
cd ..
cd fold1
python -m torch.distributed.launch --nproc_per_node=3 train8.py > train8.txt
cd ..
cd fold2
python -m torch.distributed.launch --nproc_per_node=3 train8.py > train8.txt
cd ..
cd fold3
python -m torch.distributed.launch --nproc_per_node=3 train8.py > train8.txt
cd ..
cd fold4
python -m torch.distributed.launch --nproc_per_node=3 train8.py > train8.txt
cd ..
cd validation
python valid8.py > valid8.txt
cd ..
