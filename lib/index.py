import nltk
import torch

def convert_vocab(data):
  word2index = {}
  for d in data:
    token = nltk.word_tokenize(d)
    for vo in token:
      if word2index.get(vo) == None:
        word2index[vo]=len(word2index)
  return word2index


def one_hot_encoding(word, word2index):
  tensor = torch.zeros(len(word2index))
  index = word2index[word]
  tensor[index] = 1.
  return tensor