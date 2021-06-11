from sentence_transformers import SentenceTransformer
import numpy as np

class Bert:
  def get_bert_embedding(self, sentences):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    sentence_embeddings = model.encode(sentences[1:])
    embedding = np.empty((0, 2098176), dtype=float)
    for sentence, embedding in zip(sentences, sentence_embeddings):
      embedding = np.vstack([embedding, sentence_embeddings])
    return embedding

