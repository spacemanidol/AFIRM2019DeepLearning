import pickle
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
def main():
    wc = pickle.load(open('data/wc.txt', 'rb'))
    word2idx = pickle.load(open('data/word2idx.txt', 'rb'))
    idx2vec = pickle.load(open('data/idx2vec.txt', 'rb'))
    words = sorted(wc, key=wc.get, reverse=True)[:1000]
    model = TSNE(n_components=2, perplexity=30, init='pca', method='exact', n_iter=5000)
    X = [idx2vec[word2idx[word]] for word in words]
    X = model.fit_transform(X)
    pyplot.figure(figsize=(18, 18))
    for i in range(len(X)):
        pyplot.text(X[i, 0], X[i, 1], words[i], bbox=dict(facecolor='blue', alpha=0.1))
    pyplot.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    pyplot.ylim((np.min(X[:, 1]), np.max(X[:, 1])))
    pyplot.savefig('data/TSNE.png')
if __name__ == '__main__':
    main()