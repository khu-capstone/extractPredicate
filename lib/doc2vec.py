from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk

class Doc2vec:
  def tokenize_sentence(self, sentence):
    return nltk.word_tokenize(sentence)

  def tokenize_sentences(self, sentences):
    tokenized_sentences = []
    for sentence in sentences:
      tokenized_sentences.append(self.tokenize_sentence(sentence))
    return tokenized_sentences

  def get_doc2vec_model(self, sentences):
    tokenized_sentences = self.tokenize_sentences(sentences)
    tagged_data = []
    tagged_data.extend([TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sentences)])
    model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    return model