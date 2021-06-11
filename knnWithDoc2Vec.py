import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from lib.getData import getData
from lib.doc2vec import Doc2vec

from nltk import word_tokenize

x_data, y_data = getData()

ld_x_raw_data = []
ps_x_raw_data = []

# word 생성
doc2vec = Doc2vec()


ld_x_data = doc2vec.get_doc2vec_model(x_data['list-data']).dv.vectors
ps_x_data = doc2vec.get_doc2vec_model(x_data['prev-sentence']).dv.vectors
ol_x_data = pd.DataFrame(x_data['ol'])

training_data, validation_data , training_labels, validation_labels = train_test_split(ps_x_data, y_data, test_size = 0.2, random_state = 100)

k_list = range(1,101)
accuracies = []

for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("List data to predicate")
plt.show()

