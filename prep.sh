cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/letor.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/models.tar.gz
wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.6b.zip
unzip glove.twitter.27B.zip
tar -xzvf queries.tar.gz
tar -xzvf collection.tar.gz
tar -xzvf letor.tar.gz
tar -xzvf models.tar.gz
rm *.tar.gz
rm *.zip
