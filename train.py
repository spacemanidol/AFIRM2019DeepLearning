import os
import pickle
import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch import LongTensor as LT
from torch import FloatTensor as FT

DEVICE = torch.device("cuda:0")    # torch.device("cpu")
EMBEDDING_DIMENSION = 300
EPOCHS = 200
MB_SIZE = 2000
SAMPLE_THRESHOLD = 1e-5
VOCAB_SIZE = 10000

class Bundler(nn.Module):
    def forward(self, data):
        raise NotImplementedError
    def forward_i(self, data):
        raise NotImplementedError
    def forward_o(self, data):
        raise NotImplementedError
class Word2Vec(Bundler):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.ivectors = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION, padding_idx=0)
        self.ovectors = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION, padding_idx=0)
        self.ivectors.weight = nn.Parameter(torch.cat([torch.zeros(1, EMBEDDING_DIMENSION), FT(VOCAB_SIZE - 1, EMBEDDING_DIMENSION).uniform_(-0.5 / EMBEDDING_DIMENSION, 0.5 / EMBEDDING_DIMENSION)]))
        self.ovectors.weight = nn.Parameter(torch.cat([torch.zeros(1, EMBEDDING_DIMENSION), FT(VOCAB_SIZE - 1, EMBEDDING_DIMENSION).uniform_(-0.5 / EMBEDDING_DIMENSION, 0.5 / EMBEDDING_DIMENSION)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True
    def forward(self, data):
        return self.forward_i(data)
    def forward_i(self, data):
        return self.ivectors(LT(data).to(DEVICE))
    def forward_o(self, data):
        return self.ovectors(LT(data).to(DEVICE))
class SGNS(nn.Module):
    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        VOCAB_SIZE = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, VOCAB_SIZE - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()
def main():
    wc = pickle.load(open('data/wc.txt', 'rb'))
    word2idx = pickle.load(open('data/word2idx.txt', 'rb'))
    idx2word = pickle.load(open('data/idx2word.txt', 'rb'))
    dataset = pickle.load(open('data/train.txt'),'rb')
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(SAMPLE_THRESHOLD / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    model = Word2Vec()
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=20, weights=None)
    sgns = sgns.to(DEVICE)#Comment if running without cuda
    optim = Adam(sgns.parameters())
    optimpath = 'data/optim'
    for epoch in range(1, EPOCHS + 1):
        dataloader = DataLoader(dataset, batch_size=MB_SIZE, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / MB_SIZE))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump('data/idx2vec.txt', 'wb')
    torch.save(sgns, 'data/model')
if __name__ == '__main__':
    main()