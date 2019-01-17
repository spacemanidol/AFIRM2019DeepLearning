# AFRIRM Deep learning Track
## Overview
On January 12 2019 Bhaskar Mitra, Nick Craswell, Emine Yilmaz, and Daniel Campos gave lectures and labs about Deep Learning for Information Retrieval and Data Mining at [AFIRM 2019](http://sigir.org/afirm2019/) by SIGIR. This repo contains the lab portion of the deep dive. 
### Setup
We suggest you use conda to deal with specific enviorments but feel free to ignore.
```bash
conda create -n afirm python=3.6
source activate afirm
pip install -U annoy
pip install -U torch torchvision
pip install -U sklearn
pip install -U pandas
pip install -U numpy
pip install matplotlib
pip install -U tqdm
./prep.sh
```
The prep.sh file will donwload the relevant data.


