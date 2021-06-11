import nltk

class Word2idx:
  def __init__(self):
    self.doc = {
      None: 0
    }

  def get_idx_from_word(self, word):
    if self.doc.get(word) == None:
      self.doc[word] = len(self.doc)
    return self.doc.get(word)

  def get_idx_from_sentence(self, sentence):
    token = nltk.word_tokenize(sentence)
    idxs = []
    for vo in token:
      idxs.append(self.get_idx_from_word(vo))
    return idxs