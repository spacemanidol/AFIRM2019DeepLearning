wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
sh Anaconda3-5.3.1-Linux-x86_64.sh
conda create -n afirm python=3.6
source activate afirm
pip install -U annoy
pip install -U torch torchvision
pip install -U sklearn
pip install -U pandas
pip install -U numpy
pip install -U spacy
python3 -m spacy download en
python3 -m spacy download en_core_web_lg
python3 -m spacy download en_vectors_web_lg
tar -xzvf queries.tar.gz
tar -xzvf collection.tar.gz