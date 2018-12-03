import torch
import numpy
import sklearn
from torch.autograd import Variable
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, embedding_size, embedding_dimension):
        super(SkipGram), self).__init__()
        self.embedding_size = embedding_size
        self.embedding_dimension = embedding_dimension
        self.u_embeddings = nn.Embedding(embedding_size, embedding_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(embedding_size, embedding_dimension, sparse=True)
        self.create_embeddings()
    def create_embeddings():
        range = 0.5/ self.embedding_dimension
        self.u_embeddings.weight.data.uniform(-range,range)
        self.v_embeddings.weight.data.uniform(-0,0)
    def forward(self, positive_u, potisive_v, negative_u, negative_v):
        embedding_u = self.u_embeddings(positive_u)
        embedding_v = self.u_embeddings(positive_v)
        negative_embedding_v = self.v_embeddings(negative_v)
        negative_embedding_u = self.v_embeddings(negative_u)
        score = F.logsigmoid(torch.sum(torch.mul(embedding_u, embedding_v).squeeze(), dim=1))
        negative_score = F.logsigmoid(-1* torch.bmm(negative_embedding_v, embedding_u.unsqueeze(2)).squeeze())
        return -1 * (torch.sum(score) + torch.sum(negative_score))
    def save_embedding(self, id, filename):
        embedding = self.u_embeddings.weight.data.numpy()
        with open(filename,'w') as w:
            w.write('{}\t{}\n'.format(len(id), self.embedding_dimension))
            for wid, word in id.items():
                e = embedding[wid]
                e = ' '.join(map(lambda x: str(x)), e)
                w.write('{}\t{}\n'.format(word,e))