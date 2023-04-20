# GRURippleNet

This repository is a variant of RippleNet ([arXiv](https://arxiv.org/abs/1803.03467)) implemented by  **PyTorch**:
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

For the authors' official TensorFlow implementation, see [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet).

![](https://github.com/hwwang55/RippleNet/blob/master/framework.jpg)

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.

In the transform section, We change a single linear layer into a GRU module to achieve more robust memory for item embedding and hop information.

### Files in the folder

- `data/`
  - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
  - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
  - `kg.txt`: knowledge graph file;

- `src/`: implementations of RippleNet.



### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py 
$ python main.py (note: use -h to check optional arguments)
```
