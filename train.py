import os
import pickle
import random
import torch
import numpy as np

from tqdm import tqdm
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda:0")
EMBEDDING_DIMENSION = 300
EPOCHS = 200
MB_SIZE = 2048
SAMPLE_THRESHOLD = 1e-5
VOCAB_SIZE = 100000
NEG_SAMPLES = 20

class Bundler(torch.nn.Module):
    def forward(self, data):
        raise NotImplementedError
    def forward_i(self, data):
        raise NotImplementedError
    def forward_o(self, data):
        raise NotImplementedError
class Word2Vec(Bundler):
    def __init__(self, padding_idx=0):
        super(Word2Vec, self).__init__()
        self.vocab_size = VOCAB_SIZE
        self.embedding_size = EMBEDDING_DIMENSION
        self.ivectors = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ivectors.weight = torch.nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = torch.nn.Parameter(torch.cat([torch.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True
    def forward(self, data):
        return self.forward_i(data)
    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)
    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)
class SGNS(torch.nn.Module):
    def __init__(self, embedding):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = VOCAB_SIZE
        self.n_negs = NEG_SAMPLES
        self.weights = None
    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()
class PermutedSubsampledCorpus(Dataset):
    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)
def main():
    idx2word = pickle.load(open('data/idx2word.txt', 'rb'))
    wc = pickle.load(open('data/wc.txt', 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(SAMPLE_THRESHOLD / wf)
    ws = np.clip(ws, 0, 1)
    model = Word2Vec()
    sgns = SGNS(embedding=model)
    sgns = sgns.to(DEVICE)
    optim = Adam(sgns.parameters())
    for epoch in range(1, EPOCHS + 1):
        dataset = PermutedSubsampledCorpus('data/train.txt')
        dataloader = DataLoader(dataset, MB_SIZE, shuffle=True)
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
        torch.save(sgns.state_dict(), 'data/model.pt')
        torch.save(optim.state_dict(), 'data/model.optim.pt')
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open('data/idx2vec.txt', 'wb'))
if __name__ == '__main__':
    main()