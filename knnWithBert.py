import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from lib.getData import getData
from lib.bert import Bert

# x_data, y_data = getData()

# ld_x_raw_data = []
# ps_x_raw_data = []

# # word 생성
# bert = Bert()

# ld_x_data = bert.get_bert_embedding(x_data['list-data'].array)
# ps_x_data = bert.get_bert_embedding(x_data['prev-sentence'].array)
# ol_x_data = pd.DataFrame(x_data['ol'])

# print(ld_x_data)

# training_data, validation_data , training_labels, validation_labels = train_test_split(pd.DataFrame(ps_x_data), y_data, test_size = 0.2, random_state = 100)

# k_list = range(1,101)
# accuracies = []

# for k in k_list:
#   classifier = KNeighborsClassifier(n_neighbors = k)
#   classifier.fit(training_data, training_labels)
#   accuracies.append(classifier.score(validation_data, validation_labels))


k_list = range(1,101)

plt.plot(k_list, good2)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("List data to predicate")
plt.show()




# plt.plot(k_list, accuracies)
# plt.xlabel("k")
# plt.ylabel("Validation Accuracy")
# plt.title("List data to predicate")
# plt.show()
