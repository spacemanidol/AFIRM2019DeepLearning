from __future__ import print_function
import os
import random
import codecs
import re
import datetime
import pickle
regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')
MAX_VOCAB = 100000
WINDOW = 4
def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

def generate_vocabulary():
    print_message('Converting MSMARCO files to corpus and building vocab')
    word_count = {}
    word_count['<UNK>'] = 1
    #files = ['data/queries.dev.tsv','data/queries.eval.tsv', 'data/queries.train.tsv']
    files = ['data/collection.tsv','data/queries.dev.tsv','data/queries.eval.tsv','data/queries.train.tsv'] #warning takes much longer to run
    corpus_length = 0
    for a_file in files:
        print_message("Loading file {}".format(a_file))
        with codecs.open(a_file,'r', encoding='utf-8') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) > 1:
                    l = l[1]
                else:
                    l = l[0]
                for word in regex_multi_space.sub(' ', regex_drop_char.sub(' ', l.lower())).strip().split():
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1
                    corpus_length += 1
    print_message('Done reading vocab.\nThere are {} unique words and the corpus is {} words long'.format(len(word_count), corpus_length))
    return word_count

def cleanup(possible_pairs, sentence_length):
    new_possible_pairs = []
    for v in possible_pairs:
        if v[0] >= 0 and v[1] >= 0:
            if v[1] <= sentence_length:
                new_possible_pairs.append(v)
    return new_possible_pairs

def skipgram(sentence, idx2word):
    possible_pairs = []
    triples = []
    for i in range(0,len(sentence)):
        for j in range(1,WINDOW):
            possible_pairs.append((i,i+j))
            possible_pairs.append((i,i-j))
    possible_pairs = cleanup(possible_pairs, len(sentence)-1)
    for v in possible_pairs:
        triples.append((sentence[v[0]],sentence[v[1]], idx2word[random.randint(0,MAX_VOCAB-1)]))
    return triples
def create_sentence(vocab,sentence):
    output = ''
    for word in sentence.split(' '):
        if word in vocab:
            output += word
        else:
            output += '<UNK>'
        output += ' '
    return output[:-1]
def make_triples(word_count):
    idx2word = ['<UNK>'] + sorted(word_count, key=word_count.get, reverse=True)[:MAX_VOCAB - 1]
    word2idx = {idx2word[idx]: idx for idx, _ in enumerate(idx2word)}
    vocab = set([word for word in word2idx])
    pickle.dump(word_count, open('data/wc.txt', 'wb'))
    pickle.dump(vocab, open('data/vocab.txt', 'wb'))
    pickle.dump(idx2word, open('data/idx2word.txt', 'wb'))
    pickle.dump(word2idx, open('data/word2idx.txt', 'wb'))
    print_message("Creating Train file")
    files = ['data/collection.tsv','data/queries.dev.tsv','data/queries.eval.tsv','data/queries.train.tsv']
    with open('data/triples.txt','w', encoding='utf-8') as w:
        with open('data/corpus.txt','w', encoding='utf-8') as corpus:
            for a_file in files:
                print_message("Loading file {}".format(a_file))
                with codecs.open(a_file,'r', encoding='utf-8') as f:
                    for l in f:
                        l = l.strip().split('\t')
                        if len(l) > 1:
                            l = l[1]
                        else:
                            l = l[0]
                        triples = skipgram(l.split(' '),idx2word)
                        corpus.write(create_sentence(vocab,l))
                        for v in triples:
                            w.write('{}\t{}\t{}\n'.format(v[0],v[1],v[2]))
    print_message('Done')

def preprocess():
    word_count = generate_vocabulary()
    make_triples(word_count)
if __name__ == '__main__':
    preprocess()
