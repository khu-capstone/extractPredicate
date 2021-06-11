import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from lib.getData import getData
from lib.word2idx import Word2idx

x_data, y_data = getData()

ld_x_raw_data = []
ps_x_raw_data = []

# word 생성
word2idx = Word2idx()
for idx, data in enumerate(x_data['list-data']):
  ld_x_raw_data.append(word2idx.get_idx_from_sentence(data))
for idx, data in enumerate(x_data['prev-sentence']):
  ps_x_raw_data.append(word2idx.get_idx_from_sentence(data))

ld_x_data = pd.DataFrame(ld_x_raw_data).fillna(0)
ps_x_data = pd.DataFrame(ps_x_raw_data).fillna(0)
ol_x_data = pd.DataFrame(x_data['ol'])

training_data, validation_data , training_labels, validation_labels = train_test_split(ps_x_data, y_data, test_size = 0.2, random_state = 100)

k_list = range(1,101)
accuracies = []

for k in k_list:
  print(k)
  classifier = SVC(kernel='linear', C=k)
  classifier.fit(training_data, training_labels)
  # accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("List data to predicate")
plt.show()

