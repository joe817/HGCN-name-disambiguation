# HGCN-name-disambiguation
this is the code of the following paper:
Qiao, Ziyue, Yi Du, Yanjie Fu, Pengfei Wang, and Yuanchun Zhou. "[Unsupervised Author Disambiguation using Heterogeneous Graph Convolutional Network Embedding.](https://ieeexplore.ieee.org/abstract/document/9005458)" In 2019 IEEE International Conference on Big Data (Big Data), pp. 910-919. IEEE, 2019.

If this code helps you, please cite this paper.

## Basic requirements

* python 3.6.5
* networkx 1.9.1
* gensim 3.4.0
* sklearn 0.20.1
* numpy 1.14.3
* pandas 0.23.0
* tensorflow 1.14.0


## How to run?
you should first unzip the file "experimental-results.zip", then create new folders named "gene" and "result". One pre-trained word2vec model from the python-gensim library is needed, and you should put it into the folder "gene".

 
## Data
you are recommended to use the word2vec model we pre-trained to generate word embeddings of publication titles, you can find it in [OneDrive](https://1drv.ms/u/s!AvNheLYVCGGGayqTjhiXoOgRc9w) (or [BaiduYun](https://pan.baidu.com/s/18nTdRcmZ4sKz7RbmrCIfWA)).  Or you can train your own word vectors(dimension = 100) using the [word2vec method](https://radimrehurek.com/gensim/models/word2vec.html) in gensim library.
