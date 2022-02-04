#!/bin/sh

# GPU ID
GPU=0

### Precomputing Neighbor2Seq
python precompute.py --dataset=Flickr --P=10 --add_self_loop=True
python precompute.py --dataset=Reddit --P=5 --add_self_loop=True
python precompute.py --dataset=Yelp --P=5 --add_self_loop=True
python precompute.py --dataset=ogbn-products --P=10 --add_self_loop=True --transductive=True
python precompute.py --dataset=ogbn-papers100M --P=10 --add_self_loop=True --transductive=True


# echo "======Flickr====="
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --model=posattn --lr=0.002 --K=10 --weight_decay=0.00005 --hidden=256 --dropout=0.5 --batch_size=256 --epochs=200 --pe_drop=0.25 --runs=10 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --model=attn --lr=0.002 --K=10 --weight_decay=0.00005 --hidden=256 --dropout=0.5 --batch_size=256 --epochs=200 --runs=10 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --model=conv --lr=0.0008 --K=10 --weight_decay=0.00005 --hidden=256 --dropout=0.5 --batch_size=24576 --epochs=400 --kernel_size=7 --runs=10 --log_step=1 




# echo "======Reddit====="
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --model=posattn --lr=0.002 --weight_decay=0 --hidden=256 --dropout=0.5 --dataset=Reddit --P=5 --K=3 --epochs=600 --batch_size=32768 --pe_drop=0.5 --runs=10 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --model=attn --lr=0.002 --weight_decay=0 --hidden=256 --dropout=0.5 --dataset=Reddit --P=5 --K=3 --epochs=600 --batch_size=32768 --runs=10 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --model=conv --dataset=Reddit --lr=0.00008 --P=5 --K=3 --weight_decay=0 --hidden=256 --dropout=0.5 --batch_size=32768 --epochs=600 --kernel_size=5 --runs=10 --log_step=1




# echo "======Yelp====="
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --dataset=Yelp --multilabel=True --model=posattn --lr=0.0005 --P=5 --K=2 --weight_decay=0 --dropout=0 --batch_size=8192 --hidden=512 --epochs=500 --runs=10 --pe_drop=0 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --dataset=Yelp --multilabel=True --model=attn --lr=0.0005 --P=5 --K=2 --weight_decay=0 --dropout=0 --batch_size=8192 --hidden=512 --epochs=500 --runs=10 --pe_drop=0 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_inductive.py --dataset=Yelp --multilabel=True --model=conv --kernel_size=3 --lr=0.0005 --P=5 --K=2 --weight_decay=0 --dropout=0 --batch_size=8192 --hidden=512 --epochs=150 --runs=10 --log_step=1




# echo "======ogbn-products====="
CUDA_VISIBLE_DEVICES=${GPU} python main_ogbnproducts.py --transductive=True --model=posattn --lr=0.001 --K=7 --weight_decay=0.00005 --hidden=512 --dropout=0.5 --batch_size=3072 --pe_drop=0.5 --epochs=300 --runs=10 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_ogbnproducts.py --transductive=True --model=attn --lr=0.001 --K=7 --weight_decay=0.00005 --hidden=512 --dropout=0.5 --batch_size=3072 --epochs=300 --runs=10 --log_step=1
CUDA_VISIBLE_DEVICES=${GPU} python main_ogbnproducts.py --transductive=True --model=conv --lr=0.00002 --K=7 --weight_decay=0.00005 --hidden=512 --dropout=0.5 --batch_size=64 --epochs=100 --kernel_size=7 --runs=10 --log_step=1




# echo "======ogbn-papers100M====="
CUDA_VISIBLE_DEVICES=${GPU} python main_ogbnpapers100M.py --model=posattn --lr=0.0005 --P=10 --K=10 --weight_decay=0.000005 --hidden=512 --dropout=0.25 --batch_size=12288 --pe_drop=0 --epochs=300 --runs=10 --log_step=25
CUDA_VISIBLE_DEVICES=${GPU} python main_ogbnpapers100M.py --model=attn --lr=0.0005 --P=10 --K=10 --weight_decay=0.000005 --hidden=512 --dropout=0.25 --batch_size=12288 --pe_drop=0 --epochs=300 --runs=10 --log_step=25
CUDA_VISIBLE_DEVICES=${GPU} python main_ogbnpapers100M.py --model=conv --lr=0.0005 --P=10 --K=5 --weight_decay=0.00005 --hidden=512 --dropout=0.25 --batch_size=12288 --kernel_size=5 --epochs=300 --runs=10 --log_step=25
