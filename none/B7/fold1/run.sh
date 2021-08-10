python -m torch.distributed.launch --nproc_per_node=3 train0.py > train0.txt
python -m torch.distributed.launch --nproc_per_node=3 train1.py > train1.txt
python -m torch.distributed.launch --nproc_per_node=3 train2.py > train2.txt
python -m torch.distributed.launch --nproc_per_node=3 train3.py > train3.txt
python -m torch.distributed.launch --nproc_per_node=3 train4.py > train4.txt
python -m torch.distributed.launch --nproc_per_node=3 train5.py > train5.txt
python -m torch.distributed.launch --nproc_per_node=3 train6.py > train6.txt
