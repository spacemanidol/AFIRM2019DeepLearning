wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
#sh Anaconda3-5.3.1-Linux-x86_64.sh
tar -xzvf queries.tar.gz
tar -xzvf collection.tar.gz
mv queries.*.tsv data/
mv collection.tsv data/
rm *.tar.gz