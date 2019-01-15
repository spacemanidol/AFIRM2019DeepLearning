wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir data
mv glove.6B.zip data
unzip data/glove.6b.zip
tar -xzvf queries.tar.gz
tar -xzvf collection.tar.gz
mv queries.*.tsv data/
mv collection.tsv data/
rm *.tar.gz
rm *.zip
