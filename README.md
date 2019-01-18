# AFRIRM Deep learning Track
## Overview
On January 16th,2019 Bhaskar Mitra, Nick Craswell, Emine Yilmaz, and Daniel Campos gave lectures and labs about Deep Learning for Information Retrieval and Data Mining at [AFIRM 2019](http://sigir.org/afirm2019/) by SIGIR. This repo contains the lab portion of the deep dive. The are 3 main labs:Word2Vec, LeToR, and Duet. and a few demos. Word2Vec is a pytorch based implementation of Word2Vec where embeddings are trained with MSMARCO queries and these generated embeddings are explored and compared to glove embeddings using tSNE and ANNOY. The Letor Lab goes over Learning To Rank on a toy dataset. The Duet Model is a true deep learning ranking model and is trained and run on the msmarco passage ranking task. There are a few other demos included in the repo to showcase DESM but the code is in C#. 

### Setup
We suggest you use conda to deal with specific enviorments but feel free to ignore.
```bash
#conda create -n afirm python=3.6
pip install annoy torch torchvision sklearn pandas numpy matplotlib tqdm
./prep.sh
```


