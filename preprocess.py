from __future__ import print_function
import os
import codecs
import re
import datetime
import pickle
regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')
MAX_VOCAB = 100000
WINDOW = 8
def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)
def skipgram(sentence, index):
    left = sentence[max(index - WINDOW, 0): index]
    right = sentence[index + 1: index + 1 + WINDOW]
    return sentence[index], ['<UNK>' for _ in range(WINDOW - len(left))] + left + right + ['<UNK>' for _ in range(WINDOW - len(right))]
def read_files():
    print_message('Converting MSMARCO files to corpus and building vocab')
    word_count = {}
    word_count['<UNK>'] = 1
    files = ['data/collection.tsv','data/queries.dev.tsv','data/queries.eval.tsv','data/queries.train.tsv']
    corpus_length = 0
    with codecs.open('data/corpus.txt','w') as w:
        for a_file in files:
            print_message("Loading file {}".format(a_file))
            with codecs.open(a_file,'r', encoding='utf-8') as f:
                for l in f:
                    for word in regex_multi_space.sub(' ', regex_drop_char.sub(' ', l.lower())).strip().split():
                        if word not in word_count:
                            word_count[word] = 0
                        word_count[word] += 1
                        corpus_length += 1
                        w.write(word + ' ')
    print_message('Done reading vocab.\nThere are {} unique words and the corpus is {} words long\n'.format(len(word_count), corpus_length))
    return word_count
def create_train(word_count):
    idx2word = ['<UNK>'] + sorted(word_count, key=word_count.get, reverse=True)[:MAX_VOCAB - 1]
    word2idx = {idx2word[idx]: idx for idx, _ in enumerate(idx2word)}
    vocab = set([word for word in word2idx])
    pickle.dump(word_count, open('data/word_count.txt', 'wb'))
    pickle.dump(vocab, open('data/vocab.txt', 'wb'))
    pickle.dump(idx2word, open('data/idx2word.txt', 'wb'))
    pickle.dump(word2idx, open('data/word2idx.txt', 'wb'))
    print_message("Creating Train file")
    data = []
    with codecs.open('data/corpus.txt', 'r', encoding='utf-8') as f:
        for l in f:
            sent = []
            for word in l.strip().split():
                if word in vocab:
                    sent.append(word)
                else:
                    sent.append('<UNK>')
            for i in range(len(sent)):
                iword, owords = skipgram(sent, i)
                data.append((word2idx[iword], [word2idx[oword] for oword in owords]))
    pickle.dump(data, open('data/train.txt', 'wb'))
    print_message('Done')
if __name__ == '__main__':
    create_train(read_files())