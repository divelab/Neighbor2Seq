# Neighbor2Seq: Deep Learning on Massive Graphs by Transforming Neighbors to Sequences
This repository is an official PyTorch implementation of Neighbor2Seq.

[Meng Liu](https://mengliu1998.github.io) and [Shuiwang Ji](http://people.tamu.edu/~sji/). [Neighbor2Seq: Deep Learning on Massive Graphs by Transforming Neighbors to Sequences](https://arxiv.org/abs/2202.03341) [SDM2022].

## Requirements
* PyTorch
* PyTorch Geometric (with 1.6.1-1.7.2 recommended)
* OGB

## Reference
```
@inproceedings{liu2022neighbor2seq,
  title={{Neighbor2Seq}: Deep Learning on Massive Graphs by Transforming Neighbors to Sequences},
  author={Liu, Meng and Ji, Shuiwang},
  booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining},
  year={2022},
  organization={SIAM}
}
```

## Run
All of our running scripts are included in [`run_ours.sh`](https://github.com/divelab/Neighbor2Seq/blob/main/Neighbor2Seq/run_ours.sh). An example on Flickr is as follows.
* Step 1: Precompute Neighbor2Seq
```linux
python precompute.py --dataset=Flickr --P=10 --add_self_loop=True
```
* Step 2: Train and evaluate Neighbor2Seq+Conv or Neighbor2Seq+Attn 
```linux
CUDA_VISIBLE_DEVICES=0 python main_inductive.py --model=conv --lr=0.0008 --K=10 --weight_decay=0.00005 --hidden=256 --dropout=0.5 --batch_size=24576 --epochs=400 --kernel_size=7 --runs=10 --log_step=1 
```
```linux
CUDA_VISIBLE_DEVICES=0 python main_inductive.py --model=posattn --lr=0.002 --K=10 --weight_decay=0.00005 --hidden=256 --dropout=0.5 --batch_size=256 --epochs=200 --pe_drop=0.25 --runs=10 --log_step=1
```

## Results
* Results on inductive tasks: `Reddit`, `Flickr`, and `Yelp`
<img src="https://github.com/mengliu1998/Contents/blob/master/Neighbor2Seq/result_inductive.png" width="600" />

* Results on `ogbn-papers100M`
<img src="https://github.com/mengliu1998/Contents/blob/master/Neighbor2Seq/result_papers100M.png" width="600" />

* Results on `ogbn-products`
<img src="https://github.com/mengliu1998/Contents/blob/master/Neighbor2Seq/result_products.png" width="600" />







